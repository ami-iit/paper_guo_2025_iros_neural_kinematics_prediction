import os
import numpy as np

class IODataManager:
    def __init__(self, link_data_path, human_data_path, save_nn_path, operation, task_name):
        r"""load processed (cutted but not filtered) link nad human data"""
        self.link_data_path = link_data_path
        self.human_data_path = human_data_path
        self.save_nn_path = save_nn_path
        self.operation = operation
        self.task_name = task_name

        # retrieve the link data
        self.link_features = ["lori", "lacc", "lpos", "lv", "lw"]
        self.link_data = {
            name: np.load(f"{self.link_data_path}/{self.operation}/{name}_{self.task_name}.npy")
            for name in self.link_features
        }
        # retrieve the human state data
        self.ik_features = ["pb", "rb", "vb", "s", "sdot"]
        self.human_data = {
            name: np.load(f"{self.human_data_path}/{self.operation}/{name}_{self.task_name}.npy")
            for name in self.ik_features
        }
        self.vlink_vars = {
            "vleft_foot": "vlink_LeftFoot_jacobian",
            "vright_foot": "vlink_RightFoot_jacobian",
            "vleft_arm": "vlink_LeftForeArm_jacobian",
            "vright_arm": "vlink_RightForeArm_jacobian"
        }
        self.human_data.update(
            {
                key: np.load(f"{self.human_data_path}/{self.operation}/{filename}.npy")
                for key, filename in self.vlink_vars.items()
            }
        )
    
    def shift(self, n_shift):
        r"""left/right shift link and IK data to align them"""
        self.human_data_shifted = {
            key: np.delete(val, slice(0, n_shift), axis=0)
            for key, val in self.human_data.items()
        }
        self.link_data_shifted = {
            key: val[:-n_shift] for key, val in self.link_data.items()
        }

        # save aligned data
        os.makedirs(f"{self.save_nn_path}/aligned", exist_ok=True)
        np.save(f"{self.save_nn_path}/aligned/link_data.npy", self.link_data_shifted)
        np.save(f"{self.save_nn_path}/aligned/human_data.npy", self.human_data_shifted)
        
    
    def split(self, train_ratio):
        try:
            # compute the split index
            data_length = next(iter(self.link_data_shifted.values())).shape[0]
            split_index = int(data_length*train_ratio)

            # split the link data
            link_data_train = {
                key: val[:split_index] for key, val in self.link_data_shifted.items()
            }
            link_data_test = {
                key: val[split_index:] for key, val in self.link_data_shifted.items()
            }

            # split the human data
            human_data_train = {
                key: val[:split_index] for key, val in self.human_data_shifted.items()
            }
            human_data_test = {
                key: val[split_index:] for key, val in self.human_data_shifted.items()
            }

            # define save paths
            train_path = os.path.join(self.save_nn_path, "training")
            os.makedirs(train_path, exist_ok=True)
            test_path = os.path.join(self.save_nn_path, "testing")
            os.makedirs(test_path, exist_ok=True)

            # save the train/test datasets
            np.save(os.path.join(train_path, "x.npy"), link_data_train)
            np.save(os.path.join(train_path, "y.npy"), human_data_train)

            np.save(os.path.join(test_path, "x.npy"), link_data_test)
            np.save(os.path.join(test_path, "y.npy"), human_data_test)

            print(f"Link and IK data successfully split and saved!")
        
        except KeyError as e:
            print(f"Error: Missing key in the ata dict - {e}")

        except IndexError as e:
            print(f"Error: Data arrays may be too short - {e}")
        
        except Exception as e:
            print(f"Unexpected error: {e}")

        
