r"""
Generate target whole-bodt kinematics configuration.
"""
import bipedal_locomotion_framework as blf
import bipedal_locomotion_framework.bindings.parameters_handler as blfph
import idyntree.bindings as idyntree
import numpy as np
import manifpy as manif
import datetime

class InverseKinematicsSolver:
    def __init__(self, dt, urdf_path):
        self.dt = dt
        self.period = datetime.timedelta(seconds=dt)
        self.urdf_path = urdf_path
        self.kindyncomp = idyntree.KinDynComputations()
        self.model_loader = idyntree.ModelLoader()
        self.base = "Pelvis"

    def load_urdf(self):
        self.model_loader.loadModelFromFile(self.urdf_path)
        self.kindyncomp.loadRobotModel(self.model_loader.model())
        self.kindyncomp.setFloatingBase(self.base)
        self.kindyncomp.setFrameVelocityRepresentation(idyntree.MIXED_REPRESENTATION)

    def init_config(self):
        self.n_links = self.kindyncomp.model().getNrOfLinks()
        self.dofs = self.kindyncomp.model().getNrOfDOFs()
        print(f"Loaded URDF has {self.n_links} links and {self.dofs} DoFs.")

        self.s = idyntree.VectorDynSize(self.dofs)
        self.s.zero()
        self.s_init = [0] * self.dofs
        print(f"Joint positions initialized with {self.s}")

        self.ds = idyntree.VectorDynSize(self.dofs)
        self.ds.zero()
        self.ds_init = [0] * self.dofs
        print(f"Joint velocities initialzied with {self.ds}.")

        self.H_B = idyntree.Transform()
        self.H_B = np.matrix([
            [1.0, 0., 0., 0.],
            [0., 1.0, 0., 0.],
            [0., 0., 1.0, 0.],
            [0., 0., 0., 1.0]
        ])

        self.vb = idyntree.Twist()
        self.vb.zero()

        self.gravity = idyntree.Vector3()
        self.gravity.zero()
        self.gravity.setVal(2, -9.81)

    def get_dofs(self):
        return self.dofs
    
    def get_s_init(self):
        return self.s_init
    
    def init_urdf(self):
        self.kindyncomp.setJointPos(self.s_init)

    def init_simulator(self):
        self.system = blf.continuous_dynamical_system.FloatingBaseSystemKinematics()
        self.system.set_state(([0, 0, 0], manif.SO3.Identity(), self.s_init))
        self.integrator = blf.continuous_dynamical_system.FloatingBaseSystemKinematicsForwardEulerIntegrator()
        self.integrator.set_dynamical_system(self.system)
        assert self.integrator.set_integration_step(self.dt)
        self.zero = datetime.timedelta(milliseconds=0)

    def set_IK_params(self, config_path):
        r"""IK problem defined in config_path"""
        self.params = blfph.TomlParametersHandler()
        assert self.params.set_from_file(config_path) 

    def build_IK_solver(self):
        _, _, solver = blf.ik.QPInverseKinematics.build(kin_dyn=self.kindyncomp, param_handler=self.params)
        return solver
    
    def set_simulator_control_input(self, base_vel, joint_vel):
        self.system.set_control_input((base_vel.coeffs(), joint_vel))

    def integrate(self):
        self.integrator.integrate(self.zero, self.dt)

    def get_simulation_solutions(self):
        return self.integrator.get_solution()
    
    def update_kinDynComp(self, base_pos, base_rot, jpos):
        T = idyntree.Transform()
        T.setPosition(base_pos)

        if not isinstance(base_rot, idyntree.Rotation):
            T.setRotation(base_rot)
        else:
            T.setRotation(base_rot.rotation())

        vb = idyntree.Twist()
        vb.zero()

        self.kindyncomp.setRobotState(T, jpos, vb, jpos*0, self.gravity)

    def sleep(self, tic, toc):
        delta_time = toc - tic
        if delta_time < self.period:
            blf.clock().sleep_for(self.period - delta_time)




        



