r"""
Process the raw Xsens MoCap data to have:
    - Proper input data: link inertial quantities
    - Proper output data: joint and base kinematics
"""
import argparse
from utils import arg_utils as autils
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
    # link data
    autils.add_bool_arg(flag_group, "allow_link_computation", default=False)
    autils.add_bool_arg(flag_group, "exist_link_data", default=False)
    autils.add_bool_arg(flag_group, "save_link_raw_data", default=False)
    autils.add_bool_arg(flag_group, "save_link_processed_data", default=False)
    autils.add_bool_arg(flag_group, "cut_link_head_frames", default=False)
    autils.add_bool_arg(flag_group, "select_link_minimal", default=False)
    autils.add_bool_arg(flag_group, "convert_link_q2R", default=False)
    autils.add_bool_arg(flag_group, "allow_link_filter", default=False)

    # human ik data
    autils.add_bool_arg(flag_group, "allow_inv_kin_computation", default=False)
    autils.add_bool_arg(flag_group, "exist_inv_kin_data", default=False)
    autils.add_bool_arg(flag_group, "save_inv_kin_raw_data", default=False)
    autils.add_bool_arg(flag_group, "save_inv_kin_processed_data", default=False)
    autils.add_bool_arg(flag_group, "cut_inv_kin_head_frames", default=False)
    autils.add_bool_arg(flag_group, "allow_inv_kin_filter", default=False)
    autils.add_bool_arg(flag_group, "save_inv_kin_filtered_data", default=False)
    autils.add_bool_arg(flag_group, "check_jacobians", default=False)
    autils.add_bool_arg(flag_group, "save_vlink_from_jacobians", default=False)
    
    # nn data
    autils.add_bool_arg(flag_group, "allow_nn_operations", default=False)
    autils.add_bool_arg(flag_group, "align_link_inv_kin_data", default=False)
    autils.add_bool_arg(flag_group, "split_link_inv_kin_data", default=False)

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
    if args.allow_link_computation:
        # initialize the link data processor
        inertial_loader = inertialLoader(
            const=constants, 
            args=args, 
            xsens_path=load_xsens_file_path,
            save_link_path=save_link_data_path
        )
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
    if args.allow_inv_kin_computation:
        # initialize the IK data processor
        ik_loader = ikLoader(
            const=constants, 
            args=args, 
            urdf_path=con.human_urdf_path, 
            ik_config_path=con.IK_config_path, 
            save_ik_path=save_inv_kin_data_path,
            link_data=1, 
            allow_visualizer=True
        )
        if not args.exist_inv_kin_data:
            # savik data defautly for first running
            human_data_raw = ik_loader.run_first_inverse_kinematics()
        else:
            human_data_raw = ik_loader.load_invkin_data()

        if args.cut_inv_kin_head_frames:
            human_data_cutted = ik_loader.cut_head_frames()

        if args.save_inv_kin_processed_data:
            # save the cutted human data here
            ik_loader.save_invkin_data(human_data_cutted, "processed")
        
        if args.allow_inv_kin_filter:
            human_data_filtered = ik_loader.filter_invkin_data()
            if args.save_inv_kin_filtered_data:
                ik_loader.save_invkin_data(human_data_filtered, "filtered")
        
        if args.check_jacobians:
            # defautly save the vlink data computed from Jacobians
            ik_loader.check_jacobians("processed", human_data_cutted)
    else:
        print(f"No operation for human IK data!")

    
    ########################
    ## NN Data Processing ##
    ########################
    if args.allow_nn_operations:
        # initialize the nn data manager
        nn_manager = iomanager(
            link_data_path=args.save_link_data_path, 
            human_data_path=args.save_inv_kin_data_path, 
            save_nn_path=args.save_nn_data_path, 
            operation="processed", 
            task_name=constants["tasks"][args.task_idx]
        )
        if args.align_link_inv_kin_data:
            # defautly save the aligned data
            nn_manager.shift(n_shift=4)
        if args.split_link_inv_kin_data:
            # defautly save the splited train/test subsets
            nn_manager.split(train_ratio=0.8)





    

    










        

