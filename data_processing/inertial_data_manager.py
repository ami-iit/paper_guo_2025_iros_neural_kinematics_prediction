import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as signal

import const as con
import math_utils as maths

##===========================##
## Link Inertial Data Loader ##
##===========================##
class InertialDataLoader:
    def __init__(self, const, args, xsens_path, save_link_path):
        r"""
        Args:
            :param const --> constant config 
            :param args --> config args from user commands
            :param xsens_path --> path to load xsens file 
            :param save_link_path --> path to save the link inertial data
        """
        self.const = const
        self.args = args
        self.xsens_path = xsens_path
        self.save_link_path = save_link_path
        self.link_categories = {
            "lori": 4,
            "lacc": 3,
            "lw": 3,
            "lpos": 3,
            "lv": 3
        }
    
    def save_link_data(self, link_data, data_name):
        r"""save the link data if first time retrieved"""
        # create directory to save raw/processed/filtered link data if not exists
        source_dir = f"{self.save_link_path}/{data_name}"
        os.makedirs(source_dir, exist_ok=True)

        # get task identifier
        task_name = self.const["tasks"][self.args.task_idx]

        for key, data in link_data.items():
            file_path = f"{source_dir}/{key}_{task_name}.npy"
            np.save(file_path, data)
        print(f"Saved '{link_data}' to {source_dir}")


    def retrieve_raw_link_data(self):
        r"""obtain the link inertial data for the first time"""
        xsens_file = pd.read_excel(self.xsens_path, engine="openpyxl", sheet_name=None)

        # get the raw segment inertial data
        raw_data = {
            key: xsens_file[self.const["sheets"][key]].to_numpy()
            for key in self.link_categories
        }

        # rearange the link order according to the IK problem config
        data_length = next(iter(raw_data.values())).shape[0]
        structured_data = {
            key: np.zeros((data_length, 17 * dim))
            for key, dim in self.link_categories.items()
        }

        for i, link_name in enumerate(con.links):
            print(f"Processing link {i}: {link_name}")
            for key, dim in self.link_categories.items():
                start_idx = i * dim
                end_idx = start_idx + dim
                source_idx = getattr(con, f"idx_mapping_{dim}d")[i]
                structured_data[key][:, start_idx:end_idx] = raw_data[key][:, source_idx:source_idx+dim]
        self.link_data = structured_data
        if self.args.save_link_raw_data:
            self.save_link_data(self.link_data, "raw")
            print(f"Saved all rearranged raw link data to: {self.save_link_path}/raw")
        return self.link_data


    def load_raw_link_data(self):
        r"""if already exists"""
        task_name = self.const["tasks"][self.args.task_idx]
        self.link_data = {}
        for key, _ in self.link_categories.items():
            file_path = f"{self.save_link_path}/raw/{key}_{task_name}.npy"
            if os.path.exists(file_path):
                self.link_data[key] = np.load(file_path)
            else:
                raise FileNotFoundError("File not found: {file_path}")
        return self.link_data
       
    def plot_link_data_feature(self, feature, link, data, data_type):
        r"""plot the feature of one joint in the raw/processed/filtered link data"""
        # check if link data exists
        if not hasattr(self, str(data_type)):
            raise AttributeError(f"Not found proper link data: '{data_type}'!")
        
        # check if the passed feature exists in link data
        if feature not in data:
            raise KeyError(f"Feature '{feature}' not found in link data!")
        
        # check if the passed link is valid
        if link not in con.links:
            raise ValueError(f"Link '{link}' not found in link list!")
        
        idx = con.links.index(link)
        xs = np.arange(0, data[feature].shape[0])

        # set up the plot
        fig, axs = plt.subplots(3, figsize=(8, 6), sharex=True)
        fig.suptitle(f"Raw Link Data: {feature} for {link}")
        axis_labels = ["x-axis", "y-axis", "z-axis"]

        for i in range(3):
            axs[i].plot(xs, data[feature][:, idx+i], label=axis_labels[i], color=['r', 'g', 'b'][i])
            axs[i].legend()
            axs[i].set_ylabel(axis_labels[i])
        axs[2].set_xlabel('Time Step')
        plt.tight_layout()
        plt.show()

    def cut_link_head_frames(self):
        r"""cut off the first frames (might be noisy)"""
        cut_frames = con.data_preprocessing["cut_frames"]
        self.link_data_cutted = {
            key: np.delete(data, slice(0, cut_frames), axis=0)
            for key, data in self.link_data.items()
        }
        
        print(f"Cut {cut_frames} head frames from link data.")
        return self.link_data_cutted

    def reduce_link(self):
        r"""preserve minimum numbers of links"""
        # check if cutted link data exists
        if not hasattr(self, "link_data_cutted"):
            raise AttributeError("Not found proper cutted link data!")
            
        n_rows = self.link_data_cutted["lori"].shape[0]
        # initialize new reduced arrays
        reduced_data = {
            key: np.zeros((n_rows, len(con.min_links)*dim))
            for key, dim in self.link_categories.items()
        }
        # reduce links by selecting relevant indices
        for i, _ in enumerate(con.min_links):
            for key, dim in self.link_categories.items():
                start_idx = i * dim
                end_idx = start_idx + dim
                # dynamically get index in the original array
                source_idx = getattr(con, f"min_links_{dim}d_index")[i]
                reduced_data[key][:, start_idx:end_idx] = self.link_data_cutted[key][:, source_idx:source_idx+dim]
        self.link_data_reduced = reduced_data
        return self.link_data_reduced

    def convert_link_q2R(self):
        r"""Convert the link quaternion to rotation matrix"""
        # check if reduced link data (cut head frames + min links) available
        if not hasattr(self, "link_data_reduced"):
            raise AttributeError("Not found proper reduced link data!")
        
        lori_as_q = self.link_data_reduced["lori"]
        num_steps, num_links = lori_as_q.shape[0], len(con.min_links)
        lori_as_R = np.zeros((num_steps, num_links*9))

        for i in range(num_steps):
            for j in range(num_links):
                step_q = np.array(lori_as_q[i, j*4:(j+1)*4])
                step_R = maths.convet_q2R(step_q)

                det_R = np.linalg.det(step_R.reshape((3, 3)))
                if not np.isclose(det_R, 1.0, atol=0.01):
                    raise ValueError(f"Invalid rotation matrix determinant: {det_R}")
                lori_as_R[i, j*9:(j+1)*9] = step_R.reshape((1, 9))

        self.link_data_reduced["lori"] = lori_as_R
        print(f"Updated link orientation array shape: {self.link_data_reduced["lori"].shape}")
        return self.link_data_reduced

    def filter_link_data(self):
        r"""Filter the reduced link data if required"""
        # check if reduced link data (cut head frames + min links) available
        if not hasattr(self, "link_data_reduced"):
            raise AttributeError("Not found proper reduced link data!")
        self.link_data_filtered = self.link_data_reduced.copy()

        window_size = con.data_preprocessing["savitzky_window"]
        polyorder = con.data_preprocessing["savitzky_order"]

        for key, data in self.link_data_reduced.items():
            self.link_data_filtered[key] = np.apply_along_axis(
                lambda col: signal.savgol_filter(
                    col,
                    window_length=window_size,
                    polyorder=polyorder
                ),
                axis=0, arr=data
            )
        print(f"Applied Savitzky-Golay filter with window {window_size}" 
              f"and polyorder {polyorder} to reduced link data.")
        
        self.save_link_data(self.link_data_filtered, "filtered")
        print(f"Saved all filtered link data to: {self.save_link_path}/filtered")
        return self.link_data_filtered