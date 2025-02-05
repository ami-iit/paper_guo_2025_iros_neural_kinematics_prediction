r"""
Process the raw Xsens MoCap data to have:
    - Proper input data: link inertial quantities
    - Proper output data: joint and base kinematics
"""
import argparse
from arg_utils import add_bool_arg
from inertial_data_manager import InertialDataLoader as inertialLoader
from ik_data_manager import InverseKinematicsLoader as ikLoader
from nn_data_manager import IODataManager as iomanager
import const as con


if __name__ == '__main__':
    ############
    ## Config ##
    ############
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

    #################
    ## User Config ##
    #################
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

    ##########################
    ## Link Data Processing ##
    ##########################
    inertial_loader = inertialLoader(
        const=constants, 
        args=args, 
        xsens_path=load_xsens_file_path,
        save_link_path=save_link_data_path
    )
    if args.allow_link_computation:
        if not args.exist_link_data:
            # retrieve and save the raw link inertial data when first time running
            # link_data (dict{arrays}) --> {lori, lacc, lw, lpos, lv}
            link_data = inertial_loader.retrieve_raw_link_data()
        else:
            # load existing link inertial data
            link_data = inertial_loader.load_raw_link_data()
        
        plot_raw_link_data = False
        if plot_raw_link_data:
            inertial_loader.plot_link_data_feature("lacc", "Pelvis", link_data, "link_data")

        if args.cut_link_head_frames:
            r"""using raw data as input"""
            link_data_cutted = inertial_loader.cut_link_head_frames()
        
        if args.select_link_minimal:
            r"""using cutted data as input"""
            link_data_reduced = inertial_loader.reduce_link()

        if args.convert_link_q2R:
            r"""using reduced data as input"""
            link_data_reduced_R = inertial_loader.convert_link_q2R()

        if args.save_link_processed_data:
            inertial_loader.save_link_data(link_data_reduced_R, "processed")

        if args.allow_link_filter:
            # automatically saved filtered data
            link_data_filtered = inertial_loader.filter_link_data()

        plot_filtered_link_data = False
        if plot_filtered_link_data:
            inertial_loader.plot_link_data_feature("lacc", "Pelvis", link_data_filtered, "link_data_filtered")
    else:
        print(f"No operation for link inertial data!")



    ########################
    ## IK Data Processing ##
    ########################
    ik_loader = ikLoader(
        const=constants, 
        args=args, 
        urdf_path=con.human_urdf_path, 
        ik_config_path=con.IK_config_path, 
        save_ik_path=save_inv_kin_data_path,
        link_data=1, 
        allow_visualizer=True
    )





    

    










        

