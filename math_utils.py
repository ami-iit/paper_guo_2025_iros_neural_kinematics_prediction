import numpy as np
from scipy.spatial.transform import Rotation as Rotation

def convet_q2R(q):
    r"""
    Arg:
        :param q --> quaternion of shape (4, ) as wxyz
    Return:
        Rotation matrix R of shape (3, 3)
    """
    q = np.asarray(q)
    if q.shape != (4, ):
        raise ValueError(f"Expected quaternion shape (4,), but got {q.shape}")
    # change order to xyzw
    reordered_q = np.array([q[1], q[2], q[3], q[0]])
    return Rotation.from_quat(reordered_q).as_matrix()