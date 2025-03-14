import numpy as np
import torch
from torch.utils.data import Dataset

class DataHandler(Dataset):
    def __init__(self, x, y, window_size, stride, output_steps):
        r"""
        Generate sliding windows through the whole sequence.
        Input x (dict) structure:
            - lori: link orientation (N, 9*5)
            - lacc: link acceleration (N, 3*5)
        Output y (dict) structure:
            - pb: base position (N, 3)
            - rb: base orientation (N, 9)
            - vb: base velocity (N, 6)
            - s: joint position (N, 31)
            - sdot: joint velocity (N, 31)
        """
        self.window_size = window_size
        self.stride = stride
        self.output_steps = output_steps
        # extract I/O features from the raw dicts
        self.x_lacc, self.x_lori = x['lacc'], x['lori']
        self.pb, self.rb, self.vb = y['pb'], y['rb'], y['vb']
        self.s, self.sdot = y['s'], y['sdot']

        # initialization
        lacc_list, lori_list = [], []
        pb_list, rb_list, vb_list = [], [], []
        s_list, sdot_list = [], []
        s_buffer_list, sdot_buffer_list = [], []

        # start generating overlapped windows
        n_frames = min(self.x_lacc.shape[0], self.s.shape[0])
        end_point = n_frames - self.window_size - self.output_steps
        for i in range(0, end_point, self.stride):
            lacc_list.append(self.x_lacc[i:i+self.window_size, :])
            lori_list.append(self.x_lori[i:i+self.window_size, :])

            s_buffer_list.append(self.s[i:i+self.window_size, :])
            sdot_buffer_list.append(self.sdot[i:i+self.window_size, :])

            s_list.append(self.s[i+self.window_size:i+self.window_size+self.output_steps, :])
            sdot_list.append(self.sdot[i+self.window_size:i+self.window_size+self.output_steps, :])

            pb_list.append(self.pb[i+self.window_size:i+self.window_size+self.output_steps, :])
            rb_list.append(self.rb[i+self.window_size:i+self.window_size+self.output_steps, :])
            vb_list.append(self.vb[i+self.window_size:i+self.window_size+self.output_steps, :])
        # make torch tensors
        self.lacc_tensor = torch.from_numpy(np.array(lacc_list))
        self.lori_tensor = torch.from_numpy(np.array(lori_list))
        self.s_buffer_tensor = torch.from_numpy(np.array(s_buffer_list))
        self.sdot_buffer_tensor = torch.from_numpy(np.array(sdot_buffer_list))
        self.s_tensor = torch.from_numpy(np.array(s_list))
        self.sdot_tensor = torch.from_numpy(np.array(sdot_list))
        self.pb_tensor = torch.from_numpy(np.array(pb_list))
        self.rb_tensor = torch.from_numpy(np.array(rb_list))
        self.vb_tensor = torch.from_numpy(np.array(vb_list))
        # debugging
        print(f"[Dataset] lacc tensor shape: {self.lacc_tensor.shape}")
        print(f"[Dataset] lori tensor shape: {self.lori_tensor.shape}")
        print(f"[Dataset] s_buffer tensor shape: {self.s_buffer_tensor.shape}")
        print(f"[Dataset] sdot_buffer tensor shape: {self.sdot_buffer_tensor.shape}")
        print(f"[Dataset] s tensor shape: {self.s_tensor.shape}")
        print(f"[Dataset] sdot tensor shape: {self.sdot_tensor.shape}")
        print(f"[Dataset] pb tensor shape: {self.pb_tensor.shape}")
        print(f"[Dataset] rb tensor shape: {self.rb_tensor.shape}")
        print(f"[Dataset] vb tensor shape: {self.vb_tensor.shape}")
    
    def __len__(self):
        r"""Return the total number of sliding windows."""
        return self.s_tensor.shape[0]

    def __getitem__(self, index):
        return {
            'acc': self.lacc_tensor[index],
            'ori': self.lori_tensor[index],
            's_buffer': self.s_buffer_tensor[index],
            'sdot_buffer': self.sdot_buffer_tensor[index],
            'pb': self.pb_tensor[index],
            'rb': self.rb_tensor[index],
            'vb': self.vb_tensor[index],
            's': self.s_tensor[index],
            'sdot': self.sdot_tensor[index]
        }



