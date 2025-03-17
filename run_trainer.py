import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR
from torch.optim import Adam
import argparse
import sys

from utils import arg_utils as autils
import const as con
from trainer import Trainer
from dataset_factory import dataset_manager as dm
from adam.pytorch import KinDynComputations
from adam import Representations



if __name__ == '__main__':
    #################
    ## User Config ##
    #################
    # change default values if needed
    parser = argparse.ArgumentParser()
    # ------------------ sliding window ----------------- #
    parser.add_argument("--window_size", type=int, default=10)
    parser.add_argument("--prediction_horizon", type=int, default=60)
    parser.add_argument("--stride", type=int, default=1)

    # ------------------ network ----------------- #
    parser.add_argument("--pmlp_layer_dropout", type=float, default=0.0)
    parser.add_argument("--w_s", type=float, default=10.0)
    parser.add_argument("--w_sdot", type=float, default=1.0)
    parser.add_argument("--w_pi", type=float, default=0.1)
    parser.add_argument("--pi_step", type=int, default=0)

    # ------------------ training ----------------- #
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--best_val_loss", type=float, default=float('inf'))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--clip", type=int, default=5)
    parser.add_argument("--dofs", type=int, default=31)
    parser.add_argument("--task_idx", type=int, default=0)
    parser.add_argument("--mode", type=str, default="pre_training")

    # ------------------ lr scheduler and optimizer ----------------- #
    parser.add_argument("--lr_init", type=float, default=1e-3)
    parser.add_argument("--wd_init", type=float, default=5e-4)
    parser.add_argument("--adam_eps", type=float, default=1e-9)
    parser.add_argument("--lr_warmup", type=int, default=5)
    parser.add_argument("--step_lr_size", type=int, default=5)
    parser.add_argument("--step_lr_gamma", type=float, default=0.5)
    
    # ------------------ booleans ----------------- #
    autils.add_bool_arg(parser, "gradien_clipping", default=True)
    autils.add_bool_arg(parser, "gradien_manipulation", default=True)
    autils.add_bool_arg(parser, "use_buffer", default=False)
    autils.add_bool_arg(parser, "is_wholebody", default=False)

    args = parser.parse_args()
    ######################
    ## In-script Config ##
    ######################
    print(f"Train mode: {args.mode}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_device(device)

    comp = KinDynComputations(con.human_urdf_path, con.joints_31dof, root_link="Pelvis")
    comp.set_frame_velocity_representation(Representations.MIXED_REPRESENTATION)

    urdf_path = con.human_urdf_path
    task_name = con.motion_tasks[args.task_idx]
    link_refs = None
    
    #####################
    ## Prepare Dataset ##
    #####################
    if args.mode == "pre_training":
        # train the model with xsens-mocap data
        load_data_dir = None
        ds = dm.XsensDataset(
            data_dir=load_data_dir,
            reduce_dofs=True,
            task=task_name
        )
    else:
        # fine-tune the model with ifeel-mocap data
        load_data_dir = None
        pretrained_model_dir = None



