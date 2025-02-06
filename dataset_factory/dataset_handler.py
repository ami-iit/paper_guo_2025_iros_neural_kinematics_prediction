import numpy as np
import torch
from torch.utils.data import Dataset

class DatasetHandler(Dataset):
    def __init__(self, x, y, window_size, stride, output_steps):
        r"""
        Args:
            :param x --> dict of arrays
            :param y --> dict of arrays
        """
        self.window_size = window_size
        self.stride = stride
        self.output_steps = output_steps

        # extract input features
        acc_data, ori_data = x['lacc'], x['lori']

        # extract output features
        output_keys = [
            'pb', 'rb', 'vb', 's', 'sdot',
            'vleft_foot', 'vright_foot', 'vleft_arm', 'vright_arm'
        ]
        outputs = {
            key: y[key] for key in output_keys
        }

        num_samples = acc_data.shape[0] - window_size - output_steps

        # allocate space for collected windows
        link_acc, link_ori = [], []
        s_buffer, sdot_buffer = [], []
        pbase, vbase, rbase = [], [], []
        vlink_lfoot, vlink_rfoot, vlink_larm, vlink_rarm = [], [], [], []
        j_s, j_sdot = [], []

        # single loop to extract all features at once
        for i in range(0, num_samples, stride):
            # inertial inputs
            link_acc.append(acc_data[i:i+window_size, :3*5]) # (window_size, 3*5)
            link_ori.append(ori_data[i:i+window_size, :9*5]) # (window_size, 9*5)

            # buffer for joint states
            s_buffer.append(outputs['s'][i:i+window_size, :])
            sdot_buffer.append(outputs['sdot'][i:i+window_size, :])

            # base targets
            pbase.append(outputs['pb'][i+window_size:i+window_size+output_steps, :])
            rbase.append(outputs['rb'][i+window_size:i+window_size+output_steps, :])
            vbase.append(outputs['vb'][i+window_size:i+window_size+output_steps, :])

            # desired links twist
            vlink_lfoot.append(outputs['vleft_foot'][i+window_size:i+window_size+output_steps, :])
            vlink_rfoot.append(outputs['vright_foot'][i+window_size:i+window_size+output_steps, :])
            vlink_larm.append(outputs['vleft_arm'][i+window_size:i+window_size+output_steps, :])
            vlink_rarm.append(outputs['vright_arm'][i+window_size:i+window_size+output_steps, :])

            # joint state targets
            j_s.append(outputs['s'][i+window_size:i+window_size+output_steps, :])
            j_sdot.append(outputs['sdot'][i+window_size:i+window_size+output_steps, :])

        # convert all arrays to torch tensors
        self.link_acc, self.link_ori = map(torch.from_numpy, map(np.array, [link_acc, link_ori]))
        self.s_buffer, self.sdot_buffer = map(torch.from_numpy, map(np.array, [s_buffer, sdot_buffer]))
        self.pbase, self.rbase, self.vbase = map(
            torch.from_numpy,
            map(np.array, [pbase, rbase, vbase])
        )
        self.vlink_lfoot, self.vlink_rfoot = map(
            torch.from_numpy,
            map(np.array, [vlink_lfoot, vlink_rfoot])
        )
        self.vlink_larm, self.vlink_rarm = map(
            torch.from_numpy,
            map(np.array, [vlink_larm, vlink_rarm])
        )
        self.j_s, self.j_sdot = map(torch.from_numpy, map(np.array, [j_s, j_sdot]))

    def __len__(self):
        return self.j_s.shape[0]
    
    def __getitem__(self, index):
        return {
            'acc': self.link_acc[index],
            'ori': self.link_ori[index],
            's_buffer': self.s_buffer[index],
            'sdot_buffer': self.sdot_buffer[index],
            'pb': self.pbase[index],
            'rb': self.rbase[index],
            'vb': self.vbase[index],
            'vleft_foot': self.vlink_lfoot[index],
            'vright_foot': self.vlink_rfoot[index],
            'vleft_arm': self.vlink_larm[index],
            'vright_arm': self.vlink_rarm[index],
            's': self.j_s[index],
            'sdot': self.j_sdot[index]
        }


