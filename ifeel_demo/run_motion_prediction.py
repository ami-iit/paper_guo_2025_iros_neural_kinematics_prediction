import numpy as np
import os
from progressbar import progressbar as pbar
from scipy.ndimage import gaussian_filter1d
from scipy.signal import butter, filtfilt
import torch

import visualizer as vis
import optimizer as opt
from JKNet import pMLP
import config as cfg
from calibrator import Calibrator
import maths

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
        #print(f"calib R shape: {calib[node_name].shape}")
        #print(f"link R shape: {link_R.shape}")
        #print(f"link2imu R shape: {cfg.link2imu_matrix[link].shape}")
        link_R_calib = calib[node_name] @ link_R @ cfg.link2imu_matrix[link]
        # append the ori into the array
        ori[:, :, i*9:(i+1)*9] = link_R_calib.reshape((1, 1, 9))
    return gyro, acc, ori

def base_step(bases, task, step):
    r"""
    Return three arrays:
        - base position of shape (1, 1, 3)
        - base orientation (as R) of shape (1, 1, 9)
        - base velocity of shape (1, 1, 6)
    """
    #pb, rb, vb = np.zeros((1, 1, 3)), np.zeros((1, 1, 9)), np.zeros((1, 1, 6))
    pb_step = bases[task]["base_position"][step:step+1, :, :].reshape((1, 1, 3))
    # handle base velocities
    vb_step = bases[task]["base_linear_velocity"][step:step+1, :, :].reshape((1, 1, 3))
    wb_step = bases[task]["base_angular_velocity"][step:step+1, :, :].reshape((1, 1, 3))
    vb_step_6D = np.concatenate((vb_step, wb_step), axis=-1)
    # handle base orientation
    rb_step_as_euler = bases[task]["base_orientation"][step:step+1, :, :].reshape((3,))
    rb_step = maths.euler_to_rotation_matrix(rb_step_as_euler, False).reshape((1, 1, 9))
    return pb_step, vb_step_6D, rb_step

def joint_step(joint_states, task, step, mapping):
    r"""
    Return two arrays:
        - joint position gt of shape (1, 1, 31)
        - joint velocity gt of shape (1, 1, 31)
    NOTE: need to reorder the joint indices!
    """
    jpos_step, jvel_step = np.zeros((1, 1, 31)), np.zeros((1, 1, 31))
    jpos = joint_states[task]["positions"][step:step+1, :, :].reshape((1, 1, 31))
    jvel = joint_states[task]["velocities"][step:step+1, :, :].reshape((1, 1, 31))

    # reorder the joint index
    for i in range(len(mapping)):
        jpos_step[:, :, mapping[i]] = jpos[:, :, i:i+1]
        jvel_step[:, :, mapping[i]] = jvel[:, :, i:i+1]
    return jpos_step, jvel_step

def extend_joint_state_preds(s, sdot, old_joint_list, new_joint_list):
    r"""Return the extended joint predictions."""
    new_dof = len(new_joint_list)
    s_new, sdot_new = np.zeros(new_dof, ), np.zeros(new_dof, )
    joint_map = {joint: i for i, joint in enumerate(new_joint_list)}
    for i, joint in enumerate(old_joint_list):
        joint_index = joint_map[joint]
        s_new[joint_index] = s[i]
        sdot_new[joint_index] = sdot[i]
    return s_new, sdot_new

def butter_lowpass_filter(data, cutoff=0.1, fs=1.0, order=2):
    nyquist = 0.5*fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return np.apply_along_axis(lambda m: filtfilt(b, a, m), axis=0, arr=data)

if __name__ == '__main__':
    # config
    load_data_dir = "./data/processed"
    urdf_path = "./urdf/humanSubject01_66dof.urdf"
    
    model_dirs = {
        "forward_walking": "./models/jknet_forward.pt",
        "side_stepping": "./models/jknet_side.pt",
        "forward_walking_clapping_hands": "./models/jknet_clapping.pt",
        "backward_walking": "./models/jknet_backward.pt"
    }
    finetuned_model_dirs = {
        "forward_walking": "./models/finetuned/forward/model_epoch16_0.19126.pt",
        "side_stepping": "./models/finetuned/side/model_epoch60_0.21632.pt",
        "forward_walking_clapping_hands": "./models/finetuned/clapping/model_epoch60_0.31320.pt",
        "backward_walking": "./models/finetuned/backward/model_epoch60_0.22817.pt"
    }
    USE_FINETUNED  =True
    task = cfg.tasks[2]
    print(f"Task: {task}")

    loco_task = ["forward_walking", "side_stepping", "backward_walking"]
    if task not in loco_task:
        is_wholebody_task = True
    else:
        is_wholebody_task = False
    link_refs = cfg.wholebody_links if is_wholebody_task else cfg.locomotion_links

    pred_dofs = 31
    avatar_dofs = 66
    t_gt = 0
    t_pred = 30 # max 60
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
    if USE_FINETUNED:
        model.load_state_dict(torch.load(finetuned_model_dirs[task]))
    else:
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

    joint_mapping = [cfg.joints_31dof.index(item) for item in cfg.joints]
    print(joint_states[task]['positions'][start_point:start_point+window_size, :, :1].shape)

    for i in range(len(joint_mapping)):
        jpos_gt = joint_states[task]['positions'][start_point:start_point+window_size, :, i:i+1].reshape((1, 10, 1))
        jvel_gt = joint_states[task]['velocities'][start_point:start_point+window_size, :, i:i+1].reshape((1, 10, 1))
        s_buffer[:, :, joint_mapping[i]:joint_mapping[i]+1] = jpos_gt
        sdot_buffer[:, :, joint_mapping[i]:joint_mapping[i]+1] = jvel_gt
    
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
        for step in pbar(range(start_point, K_steps-t_pred)):
            #print(f"step: {step}")
            # read current step t of imu data (acc, ori_R, gyro)
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
                #print(f"step {step}: update the imu buffer")
                # buffer already full, pop out the left most item
                # ready for prediction
                acc_buffer[:, :-1, :] = acc_buffer[:, 1:, :]
                ori_buffer[:, :-1, :] = ori_buffer[:, 1:, :]
                acc_buffer[:, -1:, :] = acc
                ori_buffer[:, -1:, :] = ori
            #counter += 1
            acc_buffer_tensor = torch.from_numpy(acc_buffer).to(dtype=torch.float64, device=device)
            ori_buffer_tensor = torch.from_numpy(ori_buffer).to(dtype=torch.float64, device=device)

            # read current step t base pose and velocity
            # NOTE: base gt from BAF
            pb, vb, rb = base_step(base_states, task, step)
            pb_tensor = torch.from_numpy(pb).to(dtype=torch.float64, device=device)
            vb_tensor = torch.from_numpy(vb).to(dtype=torch.float64, device=device)
            rb_tensor = torch.from_numpy(rb).to(dtype=torch.float64, device=device)

            pb_future, vb_future, rb_future = base_step(base_states, task, step+t_pred)
            pb_future_tensor = torch.from_numpy(pb_future).to(dtype=torch.float64, device=device)
            v_futureb_tensor = torch.from_numpy(vb_future).to(dtype=torch.float64, device=device)
            rb_future_tensor = torch.from_numpy(rb_future).to(dtype=torch.float64, device=device)

            # read current step t joint state gt (31 dof)
            # NOTE: joint state gt from BAF
            s_gt, sdot_gt = joint_step(joint_states, task, step, joint_mapping)
            #print(f"step: {step}, s gt jRightShoulder_roty: {s_gt[:, :, 2]}")
            s_gt_tensor = torch.from_numpy(s_gt).to(dtype=torch.float64, device=device)
            sdot_gt_tensor = torch.from_numpy(sdot_gt).to(dtype=torch.float64, device=device)

            # start prediction t:t+60 (31 dof)
            s_buffer_tensor = torch.from_numpy(np.array(s_buffer)).to(dtype=torch.float64, device=device)
            sdot_buffer_tensor = torch.from_numpy(np.array(sdot_buffer)).to(dtype=torch.float64, device=device)
            s_pred, sdot_pred = model(acc_buffer_tensor, ori_buffer_tensor, s_buffer_tensor, sdot_buffer_tensor)
            # smooth the prediction seq
            cutoff = 20
            s_pred_smoothed = butter_lowpass_filter(
                s_pred.cpu().detach().numpy()[0], cutoff=cutoff, fs=60
            )
            sdot_pred_smoothed = butter_lowpass_filter(
                sdot_pred.cpu().detach().numpy()[0], cutoff=cutoff, fs=60
            )

            # pick the desired one-step prediction
            USE_SMOOTHED_PRED = True
            if USE_SMOOTHED_PRED:
                s_pred_step = s_pred_smoothed[t_pred, :]
                sdot_pred_step = sdot_pred_smoothed[t_pred, :]
            else:
                s_pred_step = s_pred.cpu().detach().numpy()[0][t_pred, :]
                sdot_pred_step = sdot_pred.cpu().detach().numpy()[0][t_pred, :]
            #print(f"s pred step shape: {s_pred_step.shape}")

            # extend the one-step joint state (gt and pred) to 66 dof
            s_pred_step_new, sdot_pred_step_new = extend_joint_state_preds(
                s_pred_step,
                sdot_pred_step,
                cfg.joints_31dof, cfg.joints_66dof
            )
            s_gt_step_new, sdot_gt_step_new = extend_joint_state_preds(
                s_gt.reshape((31, )), 
                sdot_gt.reshape((31, )),
                cfg.joints_31dof, cfg.joints_66dof
            )

            # update the joint buffer arrays (31 dof)
            # NOTE: currently update with BAF computation
            # TODO: we should repalce it with a refinement process!
            s_buffer[:, :t_end, :] = s_buffer[:, 1:, :]
            sdot_buffer[:, :t_end, :] = sdot_buffer[:, 1:, :]
            s_buffer[:, t_end:, :] = s_gt
            sdot_buffer[:, t_end:, :] = sdot_gt

            # (optional) publish to yarp

            # update the visualizer: gt and pred (66 dof)
            Hb_gt[:3, :3] = rb.reshape((3, 3))
            Hb_gt[:3, 3] = pb.reshape((3, 1))
            Hb_pred[:3, :3] = rb_future.reshape((3, 3))
            #Hb_pred[:3, 3] = pb_future.reshape((3, 1)) + np.array([0, 1, 0]).reshape((3, 1))
            Hb_pred[:3, 3] = pb_future.reshape((3, 1))

            """ j_name = ["jRightElbow_roty", "jRightElbow_rotz"]
            j_index = [cfg.joints_66dof.index(item) for item in j_name]
            for idx in range(len(j_name)):
                print(f"step:{step}, s_gt {j_name[idx]}: {s_gt_step_new[j_index[idx]]}") """
          
            visualizer.update(
                [s_gt_step_new, s_pred_step_new],
                [Hb_gt, Hb_pred],
                False, None, step
            )
            visualizer.run()








