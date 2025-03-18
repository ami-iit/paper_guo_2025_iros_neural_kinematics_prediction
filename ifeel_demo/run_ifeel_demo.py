import numpy as np
import os
import torch
import visualizer as vis
import optimizer as opt
from JKNet import pMLP
import config as cfg
from calibrator import Calibrator
import maths

from adam.casadi import KinDynComputations as casakin
from adam import Representations

if __name__ == '__main__':
    # urdf and pre-trained models path
    urdf_path = "./urdf/humanSubject01_66dof.urdf"
    model_dirs = {
        "forward_walking": "./models/jknet_forward.pt",
        "side_stepping": "./models/jknet_side.pt",
        "forward_walking_clapping_hands": "./models/jknet_clapping.pt",
        "backward_walking": "./models/jknet_backward.pt"
    }
    finetuned_model_dirs = {
        "forward_walking": "./models/finetuned/forward/model_epoch16_0.19126.pt",
        "side_stepping": "./models/finetuned/side/model_epoch60_0.21632.pt",
        "forward_walking_clapping_hands": "./models/finetuned/clapping/model_epoch60_0.31320.pt",
        "backward_walking": "./models/finetuned/backward/model_epoch60_0.22817.pt"
    }

    # bool params
    USE_FINETUNED = True
    USE_BAF = True
    USE_SMOOTHED_PRED = True
    
    # default params
    pred_dofs = 31
    avatar_dofs = 66
    t_gt = 0
    t_pred = 30 # max 60
    window_size = 10
    t_end = window_size - 1
    cutoff = 20

    # TODO: maybe use human command to change motion mode?
    task = cfg.tasks[0]
    print(f"Current task: {task}")

    # check if wholbody task
    loco_task = ["forward_walking", "side_stepping", "backward_walking"]
    if task not in loco_task:
        is_wholebody_task = True
    else:
        is_wholebody_task = False
    link_refs = cfg.wholebody_links if is_wholebody_task else cfg.locomotion_links

    # prepare cuda
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_device(device)

    # prepare kindyncomp
    casacomp = casakin(urdfstring=urdf_path, joints_name_list=cfg.joints_31dof)
    casacomp.set_frame_velocity_representation(Representations.MIXED_REPRESENTATION)

    # prepare the model
    model = pMLP(
        sample=None,
        comp=casacomp,
        use_buffer=True,
        wholebody=is_wholebody_task
    )
    # load the model weights
    if USE_FINETUNED:
        model.load_state_dict(torch.load(finetuned_model_dirs[task]))
    else:
        model.load_state_dict(torch.load(model_dirs[task]))
    model.eval()

    # initialize the optimizer
    args, solver = opt.build(casacomp, link_refs, pred_dofs, is_wholebody_task)

    # initialize the visualizer
    visualizer = vis.HumanURDFVisualizer(path=urdf_path, model_names=["gt", "pred"])
    visualizer.load_model(colors=[(0.2 , 0.2, 0.2, 0.6), (1.0 , 0.2, 0.2, 0.3)])
    Hb_gt = np.matrix([[1.0, 0., 0., 0.], [0., 1.0, 0., 0.],
                       [0., 0., 1.0, 0.], [0., 0., 0., 1.0]])
    Hb_pred = np.matrix([[1.0, 0., 0., 0.], [0., 1.0, 0., 0.],
                         [0., 0., 1.0, 0.], [0., 0., 0., 1.0]])

    # joint mapping between BAF joint list and my 31-dof joint list
    joint_mapping = [cfg.joints_31dof.index(item) for item in cfg.joints]

    # initialize the joint state buffer

    # initialize the imu buffer

    # prepare the calibrator
    # NOTE: should calculate the calibration matrix from online readings!

    counter = 0
    with torch.no_grad():
        print(f"start online demo...")
        while True:
            print(f"iteration:{counter}")
            counter += 1
            # keep reading single-step inertial data from iFeel port

            # keep reading single-step joint of 31-dof from BAF port

            # keep reading single-step base data from BAF port

            # fill the inertial buffer

            # get model prediction of 31-dof
            s_pred, sdot_pred= model()

            # pick the prediction at th desired step

            # extend the prediction to 66-dof

            # update the joint state buffer of 31-dof

            # update the visualizer: gt from BAF, pred from nn
            # NOTE: here we have the same base pose for both gt and pred

