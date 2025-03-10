import numpy as np

tasks = ['forward_walking', 'side_stepping', 'forward_walking_clapping_hands', 'backward_walking']

nodes = [
    'node1', 'node2', 'node3', 'node4', 'node5', 'node6', 
    'node7', 'node8', 'node9', 'node10', 'node11', 'node12'
]
imu_attris = ['angVel', 'linAcc', 'orientation']

joints = [
    'jT9T8_rotx',
    'jT9T8_rotz',
    'jRightShoulder_rotx',
    'jRightShoulder_roty',
    'jRightShoulder_rotz',
    'jRightElbow_roty',
    'jRightElbow_rotz',
    'jLeftShoulder_rotx',
    'jLeftShoulder_roty',
    'jLeftShoulder_rotz',
    'jLeftElbow_roty',
    'jLeftElbow_rotz',
    'jLeftHip_rotx',
    'jLeftHip_roty',
    'jLeftHip_rotz',
    'jLeftKnee_roty',
    'jLeftKnee_rotz',
    'jLeftAnkle_rotx',
    'jLeftAnkle_roty',
    'jLeftAnkle_rotz',
    'jLeftBallFoot_roty',
    'jRightHip_rotx',
    'jRightHip_roty',
    'jRightHip_rotz',
    'jRightKnee_roty',
    'jRightKnee_rotz',
    'jRightAnkle_rotx',
    'jRightAnkle_roty',
    'jRightAnkle_rotz',
    'jRightBallFoot_roty',
    'jL5S1_roty'
]

joints_66dof = ["jL5S1_rotx" , "jRightHip_rotx" , "jLeftHip_rotx" , "jLeftHip_roty" , "jLeftHip_rotz" , "jLeftKnee_rotx" , "jLeftKnee_roty" ,
                "jLeftKnee_rotz" , "jLeftAnkle_rotx" , "jLeftAnkle_roty" , "jLeftAnkle_rotz" , "jLeftBallFoot_rotx" , "jLeftBallFoot_roty" ,
                "jLeftBallFoot_rotz" , "jRightHip_roty" , "jRightHip_rotz" , "jRightKnee_rotx" , "jRightKnee_roty" , "jRightKnee_rotz" ,
                "jRightAnkle_rotx" , "jRightAnkle_roty" , "jRightAnkle_rotz" , "jRightBallFoot_rotx" , "jRightBallFoot_roty" , "jRightBallFoot_rotz" ,
                "jL5S1_roty" , "jL5S1_rotz" , "jL4L3_rotx" , "jL4L3_roty" , "jL4L3_rotz" , "jL1T12_rotx" , "jL1T12_roty" , "jL1T12_rotz" ,
                "jT9T8_rotx" , "jT9T8_roty" , "jT9T8_rotz" , "jLeftC7Shoulder_rotx" , "jT1C7_rotx" , "jRightC7Shoulder_rotx" , "jRightC7Shoulder_roty" ,
                "jRightC7Shoulder_rotz" , "jRightShoulder_rotx" , "jRightShoulder_roty" , "jRightShoulder_rotz" , "jRightElbow_rotx" , "jRightElbow_roty" ,
                "jRightElbow_rotz" , "jRightWrist_rotx" , "jRightWrist_roty" , "jRightWrist_rotz" , "jT1C7_roty" , "jT1C7_rotz" , "jC1Head_rotx" ,
                "jC1Head_roty" , "jC1Head_rotz" , "jLeftC7Shoulder_roty" , "jLeftC7Shoulder_rotz" , "jLeftShoulder_rotx" , "jLeftShoulder_roty" ,
                "jLeftShoulder_rotz" , "jLeftElbow_rotx" , "jLeftElbow_roty" , "jLeftElbow_rotz" , "jLeftWrist_rotx" , "jLeftWrist_roty" ,
                "jLeftWrist_rotz"]

joint_attris = ['positions', 'velocities']
base_attris = [
    'base_linear_velocity',
    'base_position',
    'base_angular_velocity',
    'base_orientation'
]
dynamics_attris = ['joint_torques', 'wrenches']

links = {
    "node1": "LeftFoot",
    "node2": "RightFoot",
    "node3": "Pelvis",
    "node4": "LeftForeArm",
    "node5": "LeftUpperArm",
    "node6": "T8",
    "node7": "RightUpperArm",
    "node8": "RightForeArm",
    "node9": "LeftUpperLeg",
    "node10": "LeftLowerLeg",
    "node11": "RightUpperLeg",
    "node12": "RightLowerLeg"
}

IK_tasks = [
    "PELVIS_TASK",
    "LEFT_UPPER_LEG_TASK",
    "RIGHT_UPPER_LEG_TASK",
    "LEFT_LOWER_LEG_TASK",
    "RIGHT_LOWER_LEG_TASK",
    "LEFT_FOOT_TASK",
    "RIGHT_FOOT_TASK",
    "LEFT_UPPER_ARM_TASK",
    "RIGHT_UPPER_ARM_TASK",
    "LEFT_FORE_ARM_TASK",
    "RIGHT_FORE_ARM_TASK",
    "T8_TASK"
]

link2imu_matrix = {
    "Pelvis": np.matrix([[0.0, 1.0, 0.0,], [0.0, 0.0, -1.0], [-1.0, 0.0, 0.0]]),
    "T8": np.matrix([[0.0, 1.0, 0.0,], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]),
    "RightUpperArm": np.matrix([[1.0, 0.0, 0.0,], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
    "RightForeArm": np.matrix([[1.0, 0.0, 0.0,], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
    "LeftUpperArm": np.matrix([[1.0, 0.0, 0.0,], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
    "LeftForeArm": np.matrix([[1.0, 0.0, 0.0,], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
    "RightUpperLeg": np.matrix([[1.0, 0.0, 0.0,], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]]),
    "RightLowerLeg": np.matrix([[1.0, 0.0, 0.0,], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]]),
    "LeftUpperLeg": np.matrix([[1.0, 0.0, 0.0,], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]),
    "LeftLowerLeg": np.matrix([[1.0, 0.0, 0.0,], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]),
    "RightFoot": np.matrix([[0.0, 1.0, 0.0,], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
    "LeftFoot": np.matrix([[0.0, 1.0, 0.0,], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
}