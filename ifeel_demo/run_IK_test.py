r"""To test the original and retrieved joint & base data via visualization."""
import os
import numpy as np
import pandas as pd
import manifpy as manif
from progressbar import progressbar as pbar
import time
from scipy.spatial.transform import Rotation as rot

import bipedal_locomotion_framework as blf
import idyntree.bindings as idyntree
import config as cfg
import InverseKinematicsSolver as iks
import visualizer as vis

def quaternion_to_rotation_matrix(q):
    r"""q --> wxyz"""
    q = np.asarray(q)
    if q.shape != (4,):
        raise ValueError(f"Expected quaternion shape (4,), but got {q.shape}")
    reordered_q = np.array([q[1], q[2], q[3], q[0]])
    return rot.from_quat(reordered_q).as_matrix()

def euler_to_rotation_matrix(rpy):
    rpy_rad = np.radians([rpy[0], rpy[1], rpy[2]])
    return rot.from_euler('xyz', rpy_rad).as_matrix()

class Calibrator:
    def __init__(self, link2sensor, imu_ori):
        # calibration matrix from link to sensor
        self.link2sensor = link2sensor
        self.imu_ori = imu_ori

    def avg_raw_imu_ori(self):
        r"""Compute a mean matrix (from imu to imu_world) from raw IMU ori."""

        return

    
    def from_imuWorld_to_World(self):
        return
    
    def calibrate_world_yaw(self):
        return

if __name__ == "__main__":
    # config
    load_data_dir = "./data/processed"
    urdf_path = "./urdf/humanSubject01_66dof.urdf"
    ik_config = "./humanIK.toml"
    task = cfg.tasks[0]

    # load the data
    imus = np.load(os.path.join(load_data_dir, "imu_data.npy"), allow_pickle=True)
    joint_states = np.load(os.path.join(load_data_dir, "joint_data.npy"), allow_pickle=True)
    base_states = np.load(os.path.join(load_data_dir, "base_data.npy"), allow_pickle=True)

    example_imu = imus[task][cfg.nodes[0]][cfg.imu_attris[0]]
    K_steps = example_imu.shape[0]

    # TODO: calibrate the imu orientation
    calibrator = Calibrator(cfg.link2imu_matrix)
    calibration_matrices = calibrator.calibrate_world_yaw()

    # initialize the visualizer
    Hb = np.matrix([
                [1.0, 0., 0., 0.],
                [0., 1.0, 0., 0.],
                [0., 0., 1.0, 0.],
                [0., 0., 0., 1.0]
            ])
    visualizer = vis.HumanURDFVisualizer(path=urdf_path, model_names=["huamn66dof"])
    visualizer.load_model(colors=[(0.2, 0.2, 0.2, 0.9)])
    #visualizer.idyntree_visualizer.camera().animator().enableMouseControl()

    # apply blf IK 
    manager = iks.InverseKinematicsSolver(dt=0.01, urdf_path=urdf_path)
    manager.load_urdf()
    manager.init_config()
    manager.init_urdf()
    manager.init_simulator()
    manager.set_IK_params(config_path=ik_config)

    iksolver = manager.build_IK_solver()
    iksolver.get_task("JOINT_REGULARIZATION_TASK").set_set_point(manager.get_s_init())

    for i in pbar(range(K_steps)):
        time.sleep(0.02)
        for idx, key in enumerate(cfg.links):
            step_q = np.array(imus[task][key][cfg.imu_attris[-1]][i].reshape(4,))
            step_R = quaternion_to_rotation_matrix(step_q)
            iksolver.get_task(cfg.IK_tasks[idx]).set_set_point(
                blf.conversions.to_manif_rot(step_R),
                manif.SO3Tangent.Zero()
            )
        if not iksolver.advance():
            raise ValueError(f'Unable to solve the IK problems!')
        manager.set_simulator_control_input(
            iksolver.get_output().base_velocity,
            iksolver.get_output().joint_velocity
        )
        manager.integrate()

        j_vel = iksolver.get_output().joint_velocity
        _, _, j_pos = manager.get_simulation_solutions()
        #j_pos = np.zeros((66,))
        #idx_mapping = [cfg.joints_66dof.index(item) for item in cfg.joints]
        #print(len(idx_mapping))
        #print(joint_states[task]['positions'][i].shape)
        """ for m in range(len(idx_mapping)):
            #print(joint_states[task]['positions'][i])
            j_pos[idx_mapping[m]] = joint_states[task]['positions'][i][0, m] """
        #j_pos[idx_mapping] = joint_states[task]['positions'][i].reshape(66,)

        vb_linear = base_states[task]['base_linear_velocity'][i].reshape(3,)
        vb_angular = base_states[task]['base_angular_velocity'][i].reshape(3,)

        pb = base_states[task]['base_position'][i].reshape(3,)
        rb_as_euler = base_states[task]['base_orientation'][i].reshape(3,)
        rb_as_matrix = euler_to_rotation_matrix(rb_as_euler)

        manager.update_kinDynComp(
            pb,
            rb_as_matrix.reshape((3, 3)),
            j_pos
        )

        # update the visualizer
        Hb[:3, :3] = rb_as_matrix.reshape((3, 3))
        Hb[:3, 3] = pb.reshape((3, 1))
        visualizer.update([j_pos], [Hb], False, None)
        visualizer.run()






    # visualize two avatars
