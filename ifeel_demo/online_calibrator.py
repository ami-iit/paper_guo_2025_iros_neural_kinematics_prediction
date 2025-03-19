import numpy as np
from scipy.spatial.transform import Rotation as rot
import config as cfg
import maths

def cal_raw_imu_ori_avg(seqs):
    r"""
    Comput the avg of ori from raw IMU data of each node.
    Args:
        seqs: {"Pelvis": np.array([N, 4]),..., "RightLowerLeg": np.array([]N, 4)}
    """
    avg_ori_as_q = {}
    for i in range(len(cfg.input_links)):
        link_name = cfg.input_links[i]
        single_link_ori = seqs[link_name]
        single_link_ori_mean = np.mean(single_link_ori, axis=0, keepdims=True)
        avg_ori_as_q[link_name] = single_link_ori_mean
    return avg_ori_as_q


def cal_imu_to_imuWorld_R(qs):
    r"""Return the imu to imuWorld rotation matrix for each node."""
    avg_ori_as_R = {}
    for i in range(len(cfg.input_links)):
        link_name = cfg.input_links[i]
        single_link_q = qs[link_name]
        single_link_q_new = np.array(
            [single_link_q[1], single_link_q[2], single_link_q[3], single_link_q[0]]
        )
        avg_ori_as_R[link_name] = rot.from_quat(single_link_q_new).as_matrix()
    return avg_ori_as_R

def cal_imuWorld_to_World_R(Rs):
    r"""Return the transformation matrix from imuWorld to World for each node."""
    imuWorld_to_World_Rs = {}
    for i in range(len(cfg.input_links)):
        link_name = cfg.input_links[i]

        link2imu_matrix_inv = np.linalg.inv(cfg.link2imu_matrix[link_name])
        imu2imuWorld_matrix_inv = np.linalg.inv(Rs[link_name])
        link2world_matrix = cfg.link2world_matrix[link_name]

        transformation_matrix = imu2imuWorld_matrix_inv @ link2imu_matrix_inv @ link2world_matrix
        imuWorld_to_World_Rs[link_name] = transformation_matrix
    return imuWorld_to_World_Rs

def calib_World_Yaw(calib_Rs):
    r"""Set the calibration matrix of the orientation to the offset in yaw only."""
    calib_matrices_WorldYaw = {}
    for i in range(len(cfg.input_links)):
        link_name = cfg.input_links[i]

        calib_matrix = calib_Rs[link_name]
        calib_rpy = maths.rotation_matrix_to_euler(calib_matrix)
        calib_rpy_new = np.array([0.0, 0.0, calib_rpy[-1]])
        
        calib_matrices_WorldYaw[link_name] = maths.euler_to_rotation_matrix(calib_rpy_new, True)
    return calib_matrices_WorldYaw