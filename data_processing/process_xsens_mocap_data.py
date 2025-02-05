r"""
Process the raw Xsens MoCap data to have:
    - Proper input data: link inertial quantities
    - Proper output data: joint and base kinematics
"""
import argparse
from arg_utils import add_bool_arg
from inertial_data_manager import InertialDataLoader as inertialLoader
from ik_data_manager import InverseKinematicsLoader as ikLoader


if __name__ == '__main__':
    ## constant config
    constants = {
        "tasks": [
            "forward_walking_normal",
            "forward_walking_fast",
            "backward_walking_normal",
            "side_walking_normal",
            "forward_walking_lifting_arms",
            "forward_walking_waving_arms",
            "forward_walking_clapping_hands",
            "rope_jumping"
        ],
        "sheets": {
            "ori": "Segment Orientation - Quat",
            "acc": "Segment Acceleration", 
            "w": "Segment Angular Velocity",
            "pos": "Segment Position", 
            "v": "Segment Velocity"
        }
    }

    # config variables from user
    parser = argparse.ArgumentParser()

    # group for decision flags
    flag_group = parser.add_argument_group("flags")
    add_bool_arg(flag_group, "allow_link_computation", default=False)
    add_bool_arg(flag_group, "exist_link_data", default=False)
    add_bool_arg(flag_group, "save_link_raw_data", default=False)
    add_bool_arg(flag_group, "save_link_processed_data", default=False)
    add_bool_arg(flag_group, "cut_link_head_frames", default=False)
    add_bool_arg(flag_group, "select_link_minimal", default=False)
    add_bool_arg(flag_group, "convert_link_q2R", default=False)
    add_bool_arg(flag_group, "allow_link_filter", default=False)

    add_bool_arg(flag_group, "allow_inv_kin_computation", default=False)
    add_bool_arg(flag_group, "exist_inv_kin_data", default=False)
    add_bool_arg(flag_group, "save_inv_kin_raw_data", default=False)
    add_bool_arg(flag_group, "save_inv_kin_processed_data", default=False)
    add_bool_arg(flag_group, "cut_inv_kin_head_frames", default=False)
    add_bool_arg(flag_group, "allow_inv_kin_filter", default=False)
    add_bool_arg(flag_group, "save_inv_kin_filtered_data", default=False)
    add_bool_arg(flag_group, "check_jacobians", default=False)
    add_bool_arg(flag_group, "save_vlink_from_jacobians", default=False)

    add_bool_arg(flag_group, "allow_nn_operations", default=False)
    add_bool_arg(flag_group, "align_link_inv_kin_data", default=False)
    add_bool_arg(flag_group, "split_link_inv_kin_data", default=False)

    # group for paths 
    path_group = parser.add_argument_group("paths")
    path_group.add_argument("--task_idx", type=int, default=None)
    path_group.add_argument("--subject", type=str, default="davide")

    args = parser.parse_args()

    # prepare the data paths (for now still kinda hard-coded path)
    load_xsens_file_path = "../ral_code/xsens_data/05_11_cheng_davide/{}/{}/{}.xlsx".format(
        args.subject, 
        constants["tasks"][args.task_idx],
        constants["tasks"][args.task_idx]
    )
    save_link_data_path = "../ral_code/xsens_data/05_11_cheng_davide/{}/{}/link_data".format(
        args.subject,
        constants["tasks"][args.task_idx]
    )
    save_inv_kin_data_path = "../ral_code/xsens_data/05_11_cheng_davide/{}/{}/IK_data".format(
        args.subject,
        constants["tasks"][args.task_idx]
    )
    save_nn_data_path = "../ral_code/xsens_data/05_11_cheng_davide/{}/{}/NN".format(
        args.subject,
        constants["tasks"][args.task_idx]
    )



    

    










        

