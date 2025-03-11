import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as rot

def quaternion_to_rotation_matrix(q):
    r"""q --> wxyz"""
    q = np.asarray(q)
    if q.shape != (4,):
        raise ValueError(f"Expected quaternion shape (4,), but got {q.shape}")
    reordered_q = np.array([q[1], q[2], q[3], q[0]])
    return rot.from_quat(reordered_q).as_matrix()

def euler_to_rotation_matrix(rpy, flag):
    if flag:
        rpy = np.radians([rpy[0], rpy[1], rpy[2]])
    return rot.from_euler('xyz', rpy).as_matrix()


def rotation_matrix_to_euler(R):
    return rot.from_matrix(R).as_euler('xyz', degrees=True)