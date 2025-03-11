r"""To test the original and retrieved joint & base data via visualization."""
import os
import numpy as np
import manifpy as manif
from progressbar import progressbar as pbar
import time

import bipedal_locomotion_framework as blf
import config as cfg
import InverseKinematicsSolver as iks
import visualizer as vis
#import URDFVisualizer as vis
import maths
from calibrator import Calibrator

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

    # calibrate the imu orientation
    calibrator = Calibrator(cfg.link2imu_matrix, cfg.link2world_matrix, imus, task)
    calibrator.avg_raw_imu_ori(start=400, end=450)
    calibrator.from_imu_to_imuWorld()
    calibrator.from_imuWorld_to_World()
    calib_world_yaw_nodes = calibrator.calibrate_world_yaw()

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
            #print(f"node: {key}, link: {cfg.links[key]}")
            step_q = np.array(imus[task][key][cfg.imu_attris[-1]][i].reshape(4,))
            step_R = maths.quaternion_to_rotation_matrix(step_q)
            
            # calibrate the raw imu ori
            step_R_calib = calib_world_yaw_nodes[key] @ step_R @ cfg.link2imu_matrix[cfg.links[key]]
       
            #print(f"ik task: {cfg.IK_tasks[idx]}")
            iksolver.get_task(cfg.IK_tasks[idx]).set_set_point(
                blf.conversions.to_manif_rot(step_R_calib),
                manif.SO3Tangent.Zero()
            )

        if not iksolver.advance():
            raise ValueError(f'Unable to solve the IK problems!')
        
        manager.set_simulator_control_input(
            iksolver.get_output().base_velocity,
            iksolver.get_output().joint_velocity
        )
        manager.integrate()

        j_vel_blf = iksolver.get_output().joint_velocity
        pb_blf, rb_blf, j_pos_blf = manager.get_simulation_solutions()

        j_pos_baf = np.zeros((66,))
        idx_mapping = [cfg.joints_66dof.index(item) for item in cfg.joints]
        for m in range(len(idx_mapping)):
            j_pos_baf[idx_mapping[m]] = joint_states[task]['positions'][i][0, m]
        
        # base from BAF results
        vb_linear_baf = base_states[task]['base_linear_velocity'][i].reshape(3,)
        vb_angular_baf = base_states[task]['base_angular_velocity'][i].reshape(3,)

        pb_baf = base_states[task]['base_position'][i].reshape(3,)
        rb_euler_baf = base_states[task]['base_orientation'][i].reshape(3,)
        rb_matrix_baf = maths.euler_to_rotation_matrix(rb_euler_baf, False)

        use_baf_base = True
        if use_baf_base:
            pb, rb = pb_baf, rb_matrix_baf
        else:
            pb, rb = pb_blf, rb_blf.rotation()
        j_pos = j_pos_blf

        manager.update_kinDynComp(pb, rb, j_pos)
        # update the visualizer
        Hb[:3, :3] = rb.reshape((3, 3))
        Hb[:3, 3] = pb.reshape((3, 1))
        visualizer.update([j_pos], [Hb], False, None)
        visualizer.run()

