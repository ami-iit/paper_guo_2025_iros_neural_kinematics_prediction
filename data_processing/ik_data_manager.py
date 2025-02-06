import os
import time
import numpy as np
from progressbar import progressbar as pbar
import scipy.signal as signal
import matplotlib.pyplot as plt

import bipedal_locomotion_framework as blf
import idyntree.bindings as idyntree
import manifpy as manif
import const as con
import math_utils as maths
import visualizer as vis
import InverseKinematicsSolver as iksolver

##===========================##
## Inverse Kinematics Loader ##
##===========================##
class InverseKinematicsLoader:
    def __init__(self, const, args, 
                 urdf_path, ik_config_path, save_ik_path,
                 link_data, allow_visualizer):
        self.const = const
        self.args = args

        self.urdf_path = urdf_path
        self.ik_config_path = ik_config_path
        self.save_ik_path = save_ik_path

        self.link_data = link_data
        self.allow_visualizer = allow_visualizer
        self.state_list = ["pb", "rb", "vb", "s", "sdot"]


        # initialize the IK solver
        self.IK_manager = iksolver.InverseKinematicsSolver(dt=0.01, urdf_path=self.urdf_path)
        self.IK_manager.load_urdf()
        self.IK_manager.init_config()
        self.IK_manager.init_urdf()
        self.IK_manager.init_simulator()
        self.IK_manager.set_IK_params(config_path=self.ik_config_path)
        self.dofs = self.IK_manager.get_dofs()

        self.solver = self.IK_manager.build_IK_solver()
        self.solver.get_task("JOINT_REGULARIZATION_TASK").set_set_point(self.IK_manager.get_s_init())

        # initialize the visualizer
        if self.allow_visualizer:
            self.H_base = np.matrix([
                [1.0, 0., 0., 0.],
                [0., 1.0, 0., 0.],
                [0., 0., 1.0, 0.],
                [0., 0., 0., 1.0]
            ])
            self.visualizer = vis.HumanURDFVisualizer(path=self.urdf_path, model_names=["huamn66dof"])
            self.visualizer.load_model(colors=[(0.2, 0.2, 0.2, 0.9)])
            self.visualizer.idyntree_visualizer.camera().animator().enableMouseControl()

        self.n_frames = self.link_data["lori"].shape[0]
        self.pb, self.rb, self.vb = np.zeros((self.n_frames, 3)), np.zeros((self.n_frames, 9)), np.zeros((self.n_frames, 6))
        self.s, self.sdot = np.zeros((self.n_frames, self.dofs)), np.zeros((self.n_frames, self.dofs))

    def save_invkin_data(self, ik_data, dname):
        r"""dname(str) --> (raw, processed, filtered)"""
        path = os.path.join(self.save_ik_path, dname)
        os.makedirs(path, exist_ok=True)

        task_name = self.const["tasks"][self.args.task_idx]
        for key in ik_data:
            file_path = os.path.joint(path, f"{key}_{task_name}.npy")
            np.save(file_path, ik_data[key])
        

    def run_first_inverse_kinematics(self):
        print(f"Run the IK process for the first time.")
        for i in pbar(range(self.n_frames)):
            time.sleep(0.02)
            # define and solve the IK problems
            for j in range(len(con.links)):
                step_R = maths.convet_q2R(
                    np.array(self.link_data["lori"][i, j*4:(j+1)*4])
                )
                self.solver.get_task(con.IK_options["full_tasks"][j]).set_set_point(
                    blf.conversions.to_manif_rot(step_R),
                    manif.SO3Tangent.Zero()
                )
            if not self.solver.advance():
                raise ValueError(f"Unable to solve the IK problems!")
            self.IK_manager.set_simulator_control_input(
                self.solver.get_output().base_velocity,
                self.solver.get_output().joint_velocity
            )
            self.IK_manager.integrate()

            # update the joint velocity (from IK computation)
            j_vel = self.solver.get_output().joint_velocity
            self.sdot[i, :] = np.array(j_vel)

            # update the base twist (from xsens data)
            vb_linear = self.link_data["lv"][i, :3].reshape((1, 3))
            vb_angular = self.link_data["lw"][i, :3].reshape((1, 3))
            self.vb[i, :] = np.concatenate((vb_linear, vb_angular), 1)

            # update the joint position (from IK computation)
            _, _, j_pos = self.IK_manager.get_simulation_solutions()
            self.s[i, :] = np.array(j_pos)

            # update the base pose (from xsens data)
            pb_from_xsens = np.array(self.link_data["lpos"][i, :3])
            self.pb[i, :] = pb_from_xsens
            rb_from_xsens = np.array(maths.convet_q2R(np.array(self.link_data["lori"][i, :4])))
            self.rb[i, :] = rb_from_xsens.reshape(9)

            # update the kindyn computation
            self.IK_manager.update_kinDynComp(
                pb_from_xsens,
                rb_from_xsens.reshape((3, 3)),
                j_pos
            )

            # update the visualizer
            if self.allow_visualizer:
                self.H_base[:3, :3] = rb_from_xsens.reshape((3, 3))
                self.H_base[:3, 3] = pb_from_xsens.reshape((3, 1))
                self.visualizer.update([j_pos], [self.H_base], False, None)
                self.visualizer.run()

        self.ik_res = {"pb": self.pb, "rb": self.rb, "vb": self.vb, "s": self.s, "sdot": self.sdot}
        if self.args.save_inv_kin_raw_data:
            self.save_invkin_data(self.ik_res, "raw")
            print(f"Saved raw IK data to: {self.save_ik_path}/raw")
        return self.ik_res

    def load_invkin_data(self):
        self.ik_res = {}
        source_dir = os.path.join(self.save_ik_path, "raw")
        task_name = self.const["tasks"][self.args.task_idx]
        for state in self.state_list:
            try:
                file_path = f"{source_dir}/{state}_{task_name}.npy"
                self.ik_res[state] = np.load(file_path)
            except FileNotFoundError:
                print(f"File not found: {file_path}")
                self.ik_res[state] = None
        return self.ik_res
        
    
    def cut_head_frames(self):
        cut_frames = con.data_preprocessing["cut_frames"]
        self.ik_res_cutted = {
            key: np.delete(data, slice(0, cut_frames), axis=0)
            for key, data in self.ik_res.items()
        }
        print(f"Cut {cut_frames} head frames from raw IK data.")
        return self.ik_res_cutted
    

    def filter_invkin_data(self):
        if not hasattr(self, "ik_res_cutted"):
            raise AttributeError("Not found proper cutted IK data!")
        
        self.ik_res_filtered = {}
        median_window = con.data_preprocessing["median_window"]
        savitzky_window = con.data_preprocessing["savitzky_window"]
        savitzky_order = con.data_preprocessing["savitzky_order"]

        for key, data in self.ik_res_cutted:
            filtered_value = np.apply_along_axis(
                lambda x: signal.savgol_filter(
                    signal.medfilt(x, kernel_size=median_window),
                    window_length=savitzky_window,
                    polyorder=savitzky_order
                ),
                axis=0,
                arr=data
            )
            self.ik_res_filtered[key] = filtered_value
        return self.ik_res_filtered

    def check_jacobians(self, data_type, link_data):
        r"""
        data_type --> either processed or filtered
        link_data is the same.
        """
        def plot_link_twists(v1, v2, name):
            r"""v1 and v2 are of shape (N, 3)"""
            xs = np.arange(0, v1.shape[0])
            fig, axs = plt.subplots(3, figsize=(8, 6))
            fig.suptitle(f"Compare vlink of {name} between measured and computed.")
            
            xlabels = ["x-axis measured", "x-axis computed"]
            axs[0].plot(xs, np.c_[v1[:, 0], v2[:, 0]], label=xlabels)
            axs[0].legend()

            ylabels = ["y-axis measured", "y-axis computed"]
            axs[1].plot(xs, np.c_[v1[:, 1], v2[:, 1]], label=ylabels)
            axs[1].legend()

            zlabels = ["z-axis measured", "z-axis computed"]
            axs[2].plot(xs, np.c_[v1[:, 2], v2[:, 2]], label=zlabels)
            axs[2].legend()

            plt.show()

        
        link_names = ["RightFoot", "LeftFoot", "RightForeArm", "LeftForeArm"]
        save_path = f"{self.save_ik_path}/{data_type}"

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        human_data = self.ik_res_filtered if data_type == "filtered" else self.ik_res_cutted
        for name in link_names:
            print(f"Validate jacobians of link {name}")
            vlink = self.compute_vlink_with_jacobian(human_data, name)
            print(f"The shape of vlink {name}: {vlink.shape}")

            if self.args.save_vlink_from_jacobians:
                np.save(f"{save_path}/vlink_{name}_jacobian.npy", vlink)

            # get the measured link twist
            link_idx = con.min_links.index(name)
            vlink_linear = link_data["lv"][:, 3*link_idx:3*(link_idx+1)]
            vlink_angular = link_data["lw"][:, 3*link_idx:3*(link_idx+1)]

            # show the vlink linear
            plot_link_twists(vlink[:, :3], vlink_linear)

            # show the vlink angular
            plot_link_twists(vlink[:, 3:], vlink_angular)

    @staticmethod
    def compute_vlink_with_jacobian(human_data, link_name):
        comp = idyntree.KinDynComputations()
        model_loader = idyntree.ModelLoader()
        model_loader.loadReducedModelFromFile(con.human_urdf_path, con.joints_66dof, "urdf")
        comp.loadRobotModel(model_loader.model())
        comp.setFloatingBase("Pelvis")

        joint_name2index = [int(model_loader.model().getJointIndex(joint_name)) for joint_name in con.joints_66dof]
        dofs = model_loader.model().getNrOfDOFs()
        
        s = idyntree.VectorDynSize(dofs)
        s.zero()

        sdot = idyntree.VectorDynSize(dofs)
        sdot.zero()

        Hb = idyntree.Transform()
        wb = idyntree.Twist()
        wb.zero()

        gravity = idyntree.Vector3()
        gravity.zero()
        gravity.setVal(2, -9.81)

        data_size = human_data["s"].shape[0]
        vlink_jacob = np.zeros((data_size, 6))

        for i in pbar(range(data_size), desc="Processing Frames"):
            time.sleep(0.02)
            pb_step = human_data["pb"][i, :].reshape((3, 1))
            rb_step = human_data["rb"][i, :].reshape((3, 3))
            vb_step = human_data["vb"][i, :].reshape((6, 1))
            s_step = human_data["s"][i, :].reshape((dofs, 1))
            sdot_step = human_data["sdot"][i, :].reshape((dofs, 1))

            nu_step = np.vstack((vb_step, sdot_step))
            Hb_step = np.matrix([[1.0, 0., 0., 0.], [0., 1.0, 0., 0.],
                                [0., 0., 1.0, 0.], [0., 0., 0., 1.0]])
            Hb_step[:3, :3] = rb_step
            Hb_step[:3, 3] = pb_step
            Hb.fromHomogeneousTransform(idyntree.Matrix4x4(Hb_step))

            for j in range(s_step.shape[0]):
                s.setVal(joint_name2index[j], s_step[j, 0])
                sdot.setVal(joint_name2index[j], sdot_step[j, 0])

            for k in range(6):
                wb.setVal(k, vb_step[k, 0])

            comp.setRobotState(Hb, s, wb, sdot, gravity)

            # compute the link jacobian
            jacob = idyntree.MatrixDynSize(6, dofs+6)
            comp.getFrameFreeFloatingJacobian(link_name, jacob)
            jacob_numerical = np.zeros((6, dofs+6))
            for n_row in range(6):
                for n_col in range(dofs+6):
                    jacob_numerical[n_row, n_col] = jacob.getVal(n_row, n_col)

            # compute the link twist
            vlink_step = jacob_numerical @ nu_step
            vlink_jacob[i, :] = vlink_step.reshape(6)

        return vlink_jacob

        