# vlinks in xsens mocap dataset
links = [
    "Pelvis", # :3, :4
    "LeftUpperLeg", # 3:6, 4:8
    "RightUpperLeg", # 6:9, 8:12
    "LeftLowerLeg", # 9:12, 12:16
    "RightLowerLeg", # 12:15, 16:20
    "LeftFoot", # 15:18, 20:24
    "RightFoot", # 18:21, 24:28
    "Neck", # 21:24, 28:32
    "LeftShoulder", # 24:27, 32:36
    "RightShoulder", # 27:30, 36:40
    "Head", # 30:33, 40:44
    "LeftUpperArm", # 33:36, 44:48
    "RightUpperArm", # 36:39, 48:52
    "LeftForeArm", #39:42, 52:56
    "RightForeArm", # 42:45, 56:60
    "LeftHand", # 45:48, 60:64
    "RightHand" # 48:51, 64:68
    ]

# each index is a link, index order the same as the list "links"
idx_mapping_4d = [
    1, 77, 61, 81, 65, 
    85, 69, 21, 45, 29,
    25, 49, 33, 53, 37, 57, 41
    ] # for link orientation
idx_mapping_3d = [
    1, 58, 46, 61, 49,
    64, 52, 16, 34, 22,
    19, 37, 25, 40, 28, 43, 31
    ] # for link acc, pos, linear/angular vel

# only consider a minimum set of imus as inputs
min_links = [
    "Pelvis", 
    "LeftLowerLeg", 
    "RightLowerLeg", 
    "LeftForeArm", 
    "RightForeArm",
    "LeftFoot", 
    "RightFoot"
    ]

min_links_4d_index = [0, 12, 16, 52, 56, 20, 24]
min_links_3d_index = [0, 9, 12, 39, 42, 15, 18]

joints_66dof = [
    "jL5S1_rotx" , "jRightHip_rotx" , "jLeftHip_rotx" , "jLeftHip_roty" , "jLeftHip_rotz" , "jLeftKnee_rotx" , "jLeftKnee_roty" ,
    "jLeftKnee_rotz" , "jLeftAnkle_rotx" , "jLeftAnkle_roty" , "jLeftAnkle_rotz" , "jLeftBallFoot_rotx" , "jLeftBallFoot_roty" ,
    "jLeftBallFoot_rotz" , "jRightHip_roty" , "jRightHip_rotz" , "jRightKnee_rotx" , "jRightKnee_roty" , "jRightKnee_rotz" ,
    "jRightAnkle_rotx" , "jRightAnkle_roty" , "jRightAnkle_rotz" , "jRightBallFoot_rotx" , "jRightBallFoot_roty" , "jRightBallFoot_rotz" ,
    "jL5S1_roty" , "jL5S1_rotz" , "jL4L3_rotx" , "jL4L3_roty" , "jL4L3_rotz" , "jL1T12_rotx" , "jL1T12_roty" , "jL1T12_rotz" ,
    "jT9T8_rotx" , "jT9T8_roty" , "jT9T8_rotz" , "jLeftC7Shoulder_rotx" , "jT1C7_rotx" , "jRightC7Shoulder_rotx" , "jRightC7Shoulder_roty" ,
    "jRightC7Shoulder_rotz" , "jRightShoulder_rotx" , "jRightShoulder_roty" , "jRightShoulder_rotz" , "jRightElbow_rotx" , "jRightElbow_roty" ,
    "jRightElbow_rotz" , "jRightWrist_rotx" , "jRightWrist_roty" , "jRightWrist_rotz" , "jT1C7_roty" , "jT1C7_rotz" , "jC1Head_rotx" ,
    "jC1Head_roty" , "jC1Head_rotz" , "jLeftC7Shoulder_roty" , "jLeftC7Shoulder_rotz" , "jLeftShoulder_rotx" , "jLeftShoulder_roty" ,
    "jLeftShoulder_rotz" , "jLeftElbow_rotx" , "jLeftElbow_roty" , "jLeftElbow_rotz" , "jLeftWrist_rotx" , "jLeftWrist_roty" ,
    "jLeftWrist_rotz"
    ]


joints_31dof = [
    "jT9T8_rotx", "jT9T8_rotz", # T9T8
    "jRightShoulder_rotx", "jRightShoulder_roty", "jRightShoulder_rotz", # right shoulder
    "jRightElbow_roty", "jRightElbow_rotz", # right elbow
    "jLeftShoulder_rotx", "jLeftShoulder_roty", "jLeftShoulder_rotz", # left shoulder
    "jLeftElbow_roty", "jLeftElbow_rotz", # left elbow
    "jLeftHip_rotx", "jLeftHip_roty", "jLeftHip_rotz", # left hip 12:14
    "jLeftKnee_roty", "jLeftKnee_rotz", # left knee 15:16
    "jLeftAnkle_rotx", "jLeftAnkle_roty", "jLeftAnkle_rotz", # left ankle 17:19
    "jLeftBallFoot_roty", # left ball foot 20
    "jRightHip_rotx", "jRightHip_roty", "jRightHip_rotz", # right hip 21:23
    "jRightKnee_roty", "jRightKnee_rotz", # right knee 24:25
    "jRightAnkle_rotx", "jRightAnkle_roty", "jRightAnkle_rotz", # right ankle 26:28
    "jRightBallFoot_roty", # right ball foot 29
    "jL5S1_roty", # L5S1 30
    ]

joints_44dof =  [
    "jL5S1_rotx", "jL5S1_roty", 
    "jT9T8_rotx", "jT9T8_roty", "jT9T8_rotz",
    "jC7RightShoulder_rotx", "jC7LeftShoulder_rotx",
    "jT1C7_rotx", "jT1C7_roty", "jT1C7_rotz",
    "jC1Head_rotx", "jC1Head_roty",
    "jRightShoulder_rotx", "jRightShoulder_roty", "jRightShoulder_rotz",
    "jRightElbow_roty", "jRightElbow_rotz",
    "jRightWrist_rotx", "jRightWrist_rotz",
    "jLeftShoulder_rotx", "jLeftShoulder_roty", "jLeftShoulder_rotz",
    "jLeftElbow_roty", "jLeftElbow_rotz",
    "jLeftWrist_rotx", "jLeftWrist_rotz",
    "jRightHip_rotx", "jRightHip_roty", "jRightHip_rotz",
    "jRightKnee_roty", "jRightKnee_rotz",
    "jRightAnkle_rotx", "jRightAnkle_roty", "jRightAnkle_rotz",
    "jRightBallFoot_roty",
    "jLeftHip_rotx", "jLeftHip_roty", "jLeftHip_rotz",
    "jLeftKnee_roty", "jLeftKnee_rotz",
    "jLeftAnkle_rotx", "jLeftAnkle_roty", "jLeftAnkle_rotz",
    "jLeftBallFoot_roty"
    ]

data_preprocessing = {
    "cut_frames": 500,
    "median_window": 21,
    "savitzky_window": 51,
    "savitzky_order": 5
}