import numpy as np
import pandas as pd
import os
import time
from progressbar import progressbar as pbar
import torch

import visualizer as vis
import optimizer as opt
from JKNet import pMLP
import config as cfg
from calibrator import Calibrator
import maths
import idyntree.bindings as idynbind
from adam.casadi import KinDynComputations as casakin
from adam import Representations


def inertial_step(imus, task, step, calib):
    R"""
    Return three arrays:
        - acc of shape (1, 1, 3*5)
        - ori of shape (1, 1, 9*5)
        - gyro of shape (1, 1, 3*5)
    """
    acc_features, ori_features, gyro_features= 3*5, 9*5, 3*5
    # initialize one-step arrays
    gyro = np.zeros((1, 1, gyro_features))
    acc = np.zeros((1, 1, acc_features))
    ori = np.zeros((1, 1, ori_features))
    for i in range(len(cfg.input_links)):
        # get the key of the link 
        link = cfg.input_links[i]
        node_name = next((k for k, v in cfg.links.items() if v == link), None)
        link_gyro_step = imus[task][node_name][cfg.imu_attris[0]][step:step+1, :, :].reshape((1, 1, 3))
        link_acc_step = imus[task][node_name][cfg.imu_attris[1]][step:step+1, :, :].reshape((1, 1, 3))
        link_ori_step = imus[task][node_name][cfg.imu_attris[2]][step:step+1, :, :].reshape((1, 1, 4))
        # append the gyro, acc into the arrays
        gyro[:, :, i*3:(i+1)*3] = link_gyro_step
        acc[:, :, i*3:(i+1)*3] = link_acc_step
        # calibrate the imu ori
        link_R = maths.quaternion_to_rotation_matrix(link_ori_step.reshape((4,)))
        link_R_calib = calib[node_name] @ link_R @ cfg.link2imu_matrix[link].reshape((1, 1, 9))
        # append the ori into the array
        ori[:, :, i*9:(i+1)*9] = link_R_calib
    return gyro, acc, ori

def base_step(bases, step):
    return

def joint_step(joint_states, step):
    return


if __name__ == '__main__':
    # config
    load_data_dir = "./data/processed"
    urdf_path = "./urdf/humanSubject01_66dof.urdf"
    model_dirs = {
        "forward_walking": "./models/jknet_forward.pt",
        "side_stepping": "./models/jknet_side.pt",
        "forward_walking_clapping_hands": "./models/jknet_clapping.pt",
        "backward_walking": "./models/jknet_backward"
    }
    task = cfg.tasks[0]
    print(f"Task: {task}")

    is_wholebody_task = False
    link_refs = cfg.wholebody_links if is_wholebody_task else cfg.locomotion_links

    pred_dofs = 31
    avatar_dofs = 66
    t_gt = 0
    t_pred = 10 # max 60
    window_size = 10
    t_end = window_size - 1

    # prepare cuda
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_device(device)

    # prepare kindyncomp
    casacomp = casakin(urdfstring=urdf_path, joints_name_list=cfg.joints_31dof)
    casacomp.set_frame_velocity_representation(Representations.MIXED_REPRESENTATION)

    # load the model
    model = pMLP(
        sample=None,
        comp=casacomp,
        use_buffer=True,
        wholebody=is_wholebody_task
    )
    model.load_state_dict(torch.load(model_dirs[task]))
    model.eval()

    # initialize the optimizer
    args, solver = opt.build(casacomp, link_refs, pred_dofs, is_wholebody_task)

    # initialize the visualizer
    visualizer = vis.HumanURDFVisualizer(path=urdf_path, model_names=["gt", "pred"])
    visualizer.load_model(colors=[(0.2 , 0.2, 0.2, 0.6), (1.0 , 0.2, 0.2, 0.3)])
    Hb_gt = np.matrix([[1.0, 0., 0., 0.], [0., 1.0, 0., 0.],
                       [0., 0., 1.0, 0.], [0., 0., 0., 1.0]])
    Hb_pred = np.matrix([[1.0, 0., 0., 0.], [0., 1.0, 0., 0.],
                         [0., 0., 1.0, 0.], [0., 0., 0., 1.0]])
    
    # prepare the offline data
    imus = np.load(os.path.join(load_data_dir, "imu_data.npy"), allow_pickle=True)
    joint_states = np.load(os.path.join(load_data_dir, "joint_data.npy"), allow_pickle=True)
    base_states = np.load(os.path.join(load_data_dir, "base_data.npy"), allow_pickle=True)

    example_imu = imus[task][cfg.nodes[0]][cfg.imu_attris[0]]
    K_steps = example_imu.shape[0]

    # initialize the joint state buffer
    start_point = 300
    s_buffer, sdot_buffer = np.zeros((1, window_size, pred_dofs)), np.zeros((1, window_size, pred_dofs))

    idx_mapping = [cfg.joints_31dof.index(item) for item in cfg.joints]
    print(joint_states[task]['positions'][start_point:start_point+window_size, :, :1].shape)

    for i in range(len(idx_mapping)):
        jpos_gt = joint_states[task]['positions'][start_point:start_point+window_size, :, i:i+1].reshape((1, 10, 1))
        jvel_gt = joint_states[task]['velocities'][start_point:start_point+window_size, :, i:i+1].reshape((1, 10, 1))
        s_buffer[:, :, idx_mapping[i]:idx_mapping[i]+1] = jpos_gt
        sdot_buffer[:, :, idx_mapping[i]:idx_mapping[i]+1] = jvel_gt
    
    """ s_buffer_tensor = torch.from_numpy(np.array(s_buffer)).to(dtype=torch.float64, device=device)
    sdot_buffer_tensor = torch.from_numpy(np.array(sdot_buffer)).to(dtype=torch.float64, device=device) """
    print(f"s buffer shape: {s_buffer.shape}")
    print(f"sdot buffer shape: {sdot_buffer.shape}")

    # initialize the imu buffer
    acc_buffer = np.zeros((1, window_size, 3*5))
    ori_buffer = np.zeros((1, window_size, 9*5))
    counter = 0

    # prepare the calibrator for imu ori
    """
    NOTE:
    1) link2imu: offset from link to sensor (fixed)
    2) link2world: offset from link to world (Identity at T-pose)
    """
    calibrator = Calibrator(cfg.link2imu_matrix, cfg.link2world_matrix, imus, task)
    # take a sequence at T-pose and compute the avg (return quaternion)
    calibrator.avg_raw_imu_ori(start=400, end=450) 
    # compute the offset from imu to imu_world (return rotation matrix)
    calibrator.from_imu_to_imuWorld()
    # compute the offset from imu_world to world
    calibrator.from_imuWorld_to_World()
    # calibrate the world yaw only
    calib_world_yaw_nodes = calibrator.calibrate_world_yaw()

    # start the offline inference through the imu data
    with torch.no_grad():
        print(f"start prediction...")
        for step in pbar(range(start_point, K_steps)):
            #print(f"step: {step}")
            # read current step of imu data (acc, ori_R, gyro)
            r"""
            NOTE:
            1) link linear velocity currently not available
            2) link angular velocity not filtered
            3) link pose currently not available
            TODO: make sure the ori is calibrated before prediction!!
            """
            gyro, acc, ori = inertial_step(imus, task, step, calib_world_yaw_nodes)
            if counter < window_size-1:
                print(f"step {step}: fill the imu buffer")
                # fill the imu buffer until reach the length for first time
                acc_buffer[:, counter:counter+1, :] = acc
                ori_buffer[:, counter:counter+1, :] = ori
                # pass the following steps and jump to the next iteration directly
                counter += 1
                continue
            else:
                print(f"step {step}: update the imu buffer")
                # buffer already full, pop out the left most item
                # ready for prediction
                acc_buffer[:, :-1, :] = acc_buffer[:, 1:, :]
                ori_buffer[:, :-1, :] = ori_buffer[:, 1:, :]
                acc_buffer[:, -1:, :] = acc
                ori_buffer[:, -1:, :] = ori
            #counter += 1

            # read current step base pose and velocity
            # NOTE: base gt from BAF
            pb, rb, vb = base_step()
            pb_tensor = torch.from_numpy(pb).to(dtype=torch.float64, device=device)
            rb_tensor = torch.from_numpy(rb).to(dtype=torch.float64, device=device)
            vb_tensor = torch.from_numpy(vb).to(dtype=torch.float64, device=device)

            # read current step joint state gt
            # NOTE: joint state gt from BAF
            s_gt, sdot_gt = joint_step()
            s_gt_tensor = torch.from_numpy(s_gt).to(dtype=torch.float64, device=device)
            sdot_gt_tensor = torch.from_numpy(sdot_gt).to(dtype=torch.float64, device=device)

            # start prediction (31 dof)
            acc_buffer_tensor = torch.from_numpy(acc_buffer).to(dtype=torch.float64, device=device)
            ori_buffer_tensor = torch.from_numpy(ori_buffer).to(dtype=torch.float64, device=device)
            s_buffer_tensor = torch.from_numpy(np.array(s_buffer)).to(dtype=torch.float64, device=device)
            sdot_buffer_tensor = torch.from_numpy(np.array(sdot_buffer)).to(dtype=torch.float64, device=device)
            s_pred, sdot_pred = model(acc_buffer_tensor, ori_buffer_tensor, s_buffer_tensor, sdot_buffer_tensor)


            # update the joint buffer (31 dof)
            # NOTE: currently update with BAF computation
            

            # (optional) publish to yarp

            # update the visualizer: gt and pred (66 dof)





