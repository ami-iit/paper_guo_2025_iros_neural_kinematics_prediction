import os
import numpy as np
import torch
import maths
import zenoh
import time
import logging
import struct
from typing import Dict, List
from scipy.signal import butter, filtfilt
import argparse

import visualizer as vis
import optimizer as opt
from JKNet import pMLP
import config as cfg
import online_calibrator as oc

from adam.casadi import KinDynComputations as casakin
from adam import Representations

#---------------------------FUNCTIONS---------------------------#
msg_dict: Dict[str, List[float]] = {}
def listener(sample) -> None:
    r"""Process incoming data and store it in the global dict."""
    global msg_dict
    payload = sample.payload
    key = sample.key_expr

    if len(payload) % 8 != 0:
        print(f"Error: payload length {len(payload)} is not a multiple of 8 (size of float64).")
        return
    float_values = []
    for i in range(0, len(payload), 8):
        value = struct.unpack('<d', payload[i:i+8])[0]
        float_values.append(value)
    msg_dict[key] = float_values

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

def from_numpy_to_tensor(arr, device):
    return torch.from_numpy(arr).to(dtype=torch.float64, device=device)

if __name__ == '__main__':
    #---------------------------CONFIG---------------------------#
    parser= argparse.ArgumentParser()
    parser.add_argument("--task", type=int, default=0, help="Task index,default is forward walking.")
    args = parser.parse_args()

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
    # zenoh params
    ifeel_port_name = "tcp/localhost:7447"
    baf_port_name = "tcp/localhost:7448"
    data_logger_name = "iFeel/**"

    # keys to rtrieve the data
    ifeel_keys = {
        "Pelvis": {"acc": "node3/linAcc", "ori": "node3/orientation"}, # node3
        "LeftForeArm": {"acc": "node4/linAcc", "ori": "node4/orientation"}, # node4
        "RightForeArm": {"acc": "node8/linAcc", "ori": "node8/orientation"}, # node8
        "LeftLowerLeg": {"acc": "node10/linAcc", "ori": "node10/orientation"}, # node10
        "RightLowerLeg": {"acc": "node12/linAcc", "ori": "node12/orientation"}, # node12
    }
    joint_state_keys = {
        "s": "iFeel/joint_state/positions", # (1, 31)
        "sdot": "iFeel/joint_state/vlocities" # (1, 31)
    }
    base_keys = {
        "pb":"iFeel/human_state/base_position", # (1, 3)
        "rb": "iFeel/human_state/base_orientation", # (1, 4)
        "vlinear": "iFeel/human_state/base_linear_velocity", # (1, 3)
        "vangular": "iFeel/human_state/base_angular_velocity" # (1, 3)
    }

    # bool params
    USE_FINETUNED = True
    USE_BAF = True
    USE_SMOOTHED_PRED = True
    
    # default params
    pred_dofs = 31
    avatar_dofs = 66
    t_gt = 0
    t_pred = 30 # max 60
    window_size = 10
    t_end = window_size - 1
    cutoff = 20

    # TODO: maybe use human command to change motion mode?
    if args.task is not None:
        task = cfg.tasks[args.task]
    else:
        task = cfg.tasks[0] # manually set the task
    print(f"Current task: {task}")

    # check if wholbody task
    loco_task = ["forward_walking", "side_stepping", "backward_walking"]
    if task not in loco_task:
        is_wholebody_task = True
    else:
        is_wholebody_task = False
    link_refs = cfg.wholebody_links if is_wholebody_task else cfg.locomotion_links

    # prepare cuda
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_device(device)

    # prepare kindyncomp
    casacomp = casakin(urdfstring=urdf_path, joints_name_list=cfg.joints_31dof)
    casacomp.set_frame_velocity_representation(Representations.MIXED_REPRESENTATION)
    
    #---------------------------INITIALIZATION---------------------------#
    # prepare the model
    model = pMLP(
        sample=None,
        comp=casacomp,
        use_buffer=True,
        wholebody=is_wholebody_task
    )
    # load the model weights
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

    ######################################
    ## Prepare the Zenoh data streaming ##
    ######################################
    logging.basicConfig(level=logging.DEBUG)
    # initialize the zenoh config
    zenoh_config = zenoh.Config()
    zenoh_config.insert_json5("mode", '"peer"')
    zenoh_config.insert_json5("connect/endpoints", f'[{ifeel_port_name}, {baf_port_name}]')
    # initialize the zenoh session
    zenoh_session = zenoh.open(zenoh_config)
    # subsribe to the ifeel and baf ports
    sub = zenoh_session.declare_subscriber(f"{data_logger_name}", listener)

    # initialize the counters
    calib_counter = 0
    buffer_counter = 0
    # joint mapping between BAF joint list and my 31-dof joint list
    joint_mapping = [cfg.joints_31dof.index(item) for item in cfg.joints]
    # initialize the joint state buffer with zeros
    s_buffer, sdot_buffer = np.zeros((1, window_size, pred_dofs)), np.zeros((1, window_size, pred_dofs))
    # initialize the inertial buffer
    acc_buffer, ori_buffer = np.zeros((1, window_size, 3*5)), np.zeros((1, window_size, 9*5))
    # initialize the imu ori calibration matrix
    calib_frames = 300 
    raw_imu_ori = {
        "Pelvis": np.zeros((calib_frames, 4)),
        "LeftForeArm": np.zeros((calib_frames, 4)),
        "RightForeArm": np.zeros((calib_frames, 4)),
        "LeftLowerLeg": np.zeros((calib_frames, 4)),
        "RightLowerLeg": np.zeros((calib_frames, 4))
    }
    calib_matrix = None
    # initialize the step inertial data (from ifeel)
    step_acc_nodes, step_ori_nodes = np.zeros((1, 1, 3*5)), np.zeros((1, 1, 9*5))
    # initialize the step joint state data (from baf)
    step_s, step_sdot = np.zeros((1, 1, 31)), np.zeros((1, 1, 31))
    # initialize the base data (from baf)
    step_pb, step_rb = np.zeros((1, 1, 3)), np.zeros((1, 1, 9))
    step_vb_linear, step_vb_angular = np.zeros((1, 1, 3)), np.zeros((1, 1, 3))
    #---------------------------INFERENCE---------------------------#
    try:
        with torch.no_grad():
            print(f"Start online demo.")
            while True:
                # within each iteration, we want to make sure the data read from zenoh is fixed
                # it's like a downsampling of the data stream form zenoh
                zenoh_data_copy = msg_dict.copy()
                #print(f"Please keep T-pose for a while...")
                #time.sleep(1)
                #---------------------------CALIBRATION---------------------------#
                if calib_counter < calib_frames:
                    print(f"{calib_counter}: Reading T-pose data for calibration, pleaase keep still for 5 seconds...")
                    # read a sequence of raw imu ori (as quaternion) of each input node
                    for item in cfg.input_links:
                        raw_imu_ori[item][calib_counter] = np.array(zenoh_data_copy[ifeel_keys[item]["ori"]])
                    calib_counter += 1
                    continue
                elif calib_counter == calib_frames:
                    # calculate the average of the raw imu ori
                    avg_imu_ori_as_q = oc.cal_raw_imu_ori_avg(raw_imu_ori)

                    # calculate the rotation matrix from imu to imuWorld
                    rot_imu_imuWorld = oc.cal_imu_to_imuWorld_R(avg_imu_ori_as_q)

                    # calculate the rotation matrix from imuWoeld to World
                    rot_imuWorld_World = oc.cal_imuWorld_to_World_R(rot_imu_imuWorld)

                    # calculate the calibration matrix of the orientation to the offset in yaw only
                    calib_matrix = oc.calib_World_Yaw(rot_imuWorld_World)
                    print(f"Calibration is done!")
                # check if the calibration matrix is valid
                if calib_matrix is None:
                    raise ValueError(f"Calibration matrix is not valid!")
                #---------------------------INERTIAL DATA---------------------------#
                # read single-step inertial data (acc and ori) from ifeel port
                for i in range(len(cfg.input_links)):
                    link_name = cfg.input_links[i]
                    # acc can be read straight away
                    step_acc_nodes[:, :, i*3:(i+1)*3] = np.array(zenoh_data_copy[ifeel_keys[link_name]["acc"]]).reshap((1, 1, 3))
                    # handle the orientation
                    step_node_quat = np.array(zenoh_data_copy[ifeel_keys[link_name]["ori"]]).reshape((4, ))
                    step_node_R = maths.quaternion_to_rotation_matrix(step_node_quat)
                    step_node_R_calib = calib_matrix[link_name] @ step_node_R @ cfg.link2imu_matrix[link_name]
                    step_ori_nodes[:, :,i*9:(i+1)*9] = step_node_R_calib.reshape((1, 1, 9))
             
                #---------------------------JOINT STATE DATA---------------------------#
                # read single-step joint state data (31-dof) from baf port
                step_s = np.array(zenoh_data_copy[joint_state_keys["s"]]).reshape((1, 1, 31))
                step_sdot = np.array(zenoh_data_copy[joint_state_keys["sdot"]]).reshape((1, 1, 31))

                #---------------------------BASE DATA---------------------------#
                # read single-step base pose and velocity from baf port
                step_pb = np.array(zenoh_data_copy[base_keys["pb"]]).reshape((1, 1, 3))
                # convert rb from euler angles to R
                step_rb_as_euler = np.array(zenoh_data_copy[base_keys["rb"]]).reshape((3, ))
                step_rb = maths.euler_to_rotation_matrix(step_rb_as_euler, False).reshape((1, 1, 9))
                # get vb as a 6D array
                step_vb_linear = np.array(zenoh_data_copy[base_keys["vlinear"]]).reshape((1, 1, 3))
                step_vb_angular = np.array(zenoh_data_copy[base_keys["vangular"]]).reshape((1, 1, 3))
                step_vb = np.concatenate((step_vb_linear, step_vb_angular), axis=-1)

                #---------------------------UPDATE BUFFER---------------------------#
                if buffer_counter < window_size:
                    print(f"step {buffer_counter}: fill the imu and joint state buffer")
                    # fill the imu buffer
                    acc_buffer[:, buffer_counter:buffer_counter+1, :]  = step_acc_nodes
                    ori_buffer[:, buffer_counter:buffer_counter+1, :] = step_ori_nodes

                    # fill the joint state buffer
                    s_buffer[:, buffer_counter:buffer_counter+1, :] = step_s
                    sdot_buffer[:, buffer_counter:buffer_counter+1, :] = step_sdot

                    buffer_counter += 1
                    continue
                else:
                    #print(f"imu and joint state buffers are ready!")
                    # pop out the first element in the original buffer
                    acc_buffer[:, :-1, :] = acc_buffer[:, 1:, :]
                    ori_buffer[:, :-1, :] = ori_buffer[:, 1:, :]
                    # pop in the new element
                    acc_buffer[:, -1:, :] = step_acc_nodes
                    ori_buffer[:, -1:, :] = step_ori_nodes
                    if USE_BAF:
                        #print(f"update joint state buffer with BAF data.")
                        # NOTE: eventually we should update the joint state buffer with refined predictions
                        s_buffer[:, :-1, :] = s_buffer[:, 1:, :]
                        sdot_buffer[:, :-1, :] = sdot_buffer[:, 1:, :]
                        s_buffer[:, -1:, :] = step_s
                        sdot_buffer[:, -1:, :] = step_sdot
             
                #---------------------------PREDICTION---------------------------#
                # convert all data to tensors
                acc_buffer_tensor = from_numpy_to_tensor(acc_buffer, device)
                ori_buffer_tensor = from_numpy_to_tensor(ori_buffer, device)
                s_buffer_tensor = from_numpy_to_tensor(s_buffer, device)
                sdot_buffer_tensor = from_numpy_to_tensor(sdot_buffer, device)

                # get model prediction of 31-dof
                s_pred, sdot_pred = model(
                    acc_buffer_tensor, ori_buffer_tensor, s_buffer_tensor, sdot_buffer_tensor
                )

                # pick the prediction at the specified step
                if USE_SMOOTHED_PRED:
                    s_pred_smoothed = butter_lowpass_filter(
                        s_pred.cpu().detach().numpy()[0], cutoff=cutoff, fs=60
                    )
                    s_pred_step = s_pred_smoothed[t_pred, :]
                    sdot_pred_smoothed = butter_lowpass_filter(
                        sdot_pred.cpu().detach().numpy()[0], cutoff=cutoff, fs=60
                    )
                    sdot_pred_step = sdot_pred_smoothed[t_pred, :]
                else:
                    s_pred_step = s_pred.cpu().detach().numpy()[0][t_pred, :]
                    sdot_pred_step = sdot_pred.cpu().detach().numpy()[0][t_pred, :]

                # extend the one-step gt and prediction to 66-dof
                s_pred_step_new, sdot_pred_step_new = extend_joint_state_preds(
                    s_pred_step, 
                    sdot_pred_step,
                    cfg.joints_31dof, cfg.joints_66dof
                )
                s_gt_step_new, sdot_gt_step_new = extend_joint_state_preds(
                    step_s,
                    step_sdot,
                    cfg.joints, cfg.joints_66dof
                )

                #---------------------------VISUALIZATION---------------------------#
                # update the base pose 
                Hb_gt[:3, :3] = step_rb.reshape((3, 3))
                Hb_gt[:3, 3] = step_pb.reshape((3, 1))

                # update the visualizer: gt from BAF, pred from nn
                visualizer.update(
                    [s_gt_step_new, s_pred_step_new],
                    [Hb_gt, Hb_gt],
                    False, None
                )
                visualizer.run()
    #---------------------------INTERUPT---------------------------#
    except KeyboardInterrupt:
        pass


