r"""Finetune the pretrained model with iFeel data."""
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from torch.optim import Adam

from JKNet import pMLP
import config as cfg
from trainer import Trainer
from make_dataset import CustomDataset
from adam.pytorch import KinDynComputations
from adam import Representations

if __name__ == '__main__':
    # paths
    load_data_dir = "./data/processed"
    urdf_path = "./urdf/humanSubject01_66dof.urdf"

    model_dirs = {
        "forward_walking": "./models/jknet_forward.pt",
        "side_stepping": "./models/jknet_side.pt",
        "forward_walking_clapping_hands": "./models/jknet_clapping.pt",
        "backward_walking": "./models/jknet_backward.pt"
    }
    # custom config
    task = cfg.tasks[1]
    print(f"Task: {task}")
    loco_task = ["forward_walking", "side_stepping", "backward_walking"]
    if task not in loco_task:
        is_wholebody_task = True
    else:
        is_wholebody_task = False
    link_refs = cfg.wholebody_links if is_wholebody_task else cfg.locomotion_links

    train_ratio = 0.7
    input_window_size = 10
    stride = 1
    output_steps = 60
    batch_size = 64

    lr_init = 1e-3
    wd_init = 5e-4
    adam_eps = 1e-9
    step_lr_size = 5
    step_lr_gamma = 0.5
    lr_warm_up = 5

    epochs = 60
    gradient_clipping = True
    clip = 5
    best_val_loss = float('inf')
    weights = {'s': 1.0, 'sdot': 1.0}
    
    # set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_device(device)

    comp = KinDynComputations(urdf_path, cfg.joints_31dof, root_link="Pelvis")
    comp.set_frame_velocity_representation(Representations.MIXED_REPRESENTATION)

    # prepare the iFeel dataset
    ds = CustomDataset(
        data_dir=load_data_dir, 
        task=task, 
        dofs=31
    )
    ds.get_features()
    ds.init_calib(start=400, end=450) # prepare the calibration matrix
    ds.calib_link_ori() # calibrate raw ifeel ori data
    ds.transform_base_ori() # transform rb to rotation matrix
    ds.make_feature_dict()
    ds.get_splitted_datasets(ratio=train_ratio)
    ds.generate(window_size=input_window_size, stride=stride, output_steps=output_steps)
    train_iterator, val_iterator = ds.iterate(bs=batch_size)

    sample = next(iter(train_iterator))

    # load the pre-trained model weigths
    # NOTE: we start with finetuning the whole model
    model = pMLP(
        sample=sample,
        comp=comp,
        use_buffer=True,
        wholebody=is_wholebody_task
    )
    model.load_state_dict(torch.load(model_dirs[task]))
    model.to(device)

    # loss function
    criterion = nn.MSELoss(reduction="mean")
    # optimizer
    optimizer = Adam(
        model.parameters(),
        lr=lr_init,
        weight_decay=wd_init,
        eps=adam_eps
    )
    # learning rate scheduler
    lr_scheduler = StepLR(
        optimizer,
        step_size=step_lr_size,
        gamma=step_lr_gamma
    )

    Problem = {
        "model": model,
        "criterion": criterion,
        "optimizer": optimizer,
        "lr_scheduler": lr_scheduler
    }

    Iterators = {"train": train_iterator, "val": val_iterator}
    Devices = {"device": device, "comp": comp}

    # NOTE: we don't use Physics-informed components when finetuning!
    Params = {
        "epochs": epochs,
        "gradient_clipping": gradient_clipping,
        "clip": clip,
        "best_val_loss": best_val_loss,
        "is_wholebody": is_wholebody_task,
        "use_physics": False,
        "links": ["LeftLowerLeg", "RightLowerLeg", "LeftForeArm", "RightForeArm"],
        "lr_warm_up": lr_warm_up,
        "weights": weights
    }

    trainer = Trainer(Problem, Iterators, Devices, Params)
    trainer.run()



    



