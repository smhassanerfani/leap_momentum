import os
import torch
import numpy as np
import torchvision.transforms as T
from typing import Any, Callable, Optional, Tuple

class Agulhas(torch.utils.data.Dataset):

    def __init__(self, split, joint_transform=None):
        super(Agulhas, self).__init__()
        
        self.split = split
        self.inps_mean_std = (-0.4315792, 0.5710749)
        self.tars_mean_std = (2.9930247e-06, 0.009364196)

        self.inputs, self.targets = self._get_data_array()
        self.joint_transform = joint_transform

        self.inps_transform = T.Compose([T.ToTensor(), T.Normalize(*self.inps_mean_std)])
        self.tars_transform = T.Compose([T.ToTensor(), T.Normalize(*self.tars_mean_std)])

    def _get_data_array(self):
        
        inps_path = os.path.join('dataset', self.split, 'inputs.npy')
        tars_path = os.path.join('dataset', self.split, 'targets.npy')
                                 
        with open(inps_path, 'rb') as f:
            inputs = np.load(f)
        
        with open(tars_path, 'rb') as f:
            targets = np.load(f)
            
        return inputs, targets

    def __getitem__(self, index):
        
        x = self.inputs[..., index]
        y = self.targets[..., index]
    
        x = self.inps_transform(x)
        y = self.tars_transform(y)
        
        if self.joint_transform:
            x, y = self.joint_transform(x, y)

        return x, y

    def __len__(self):
        return self.inputs.shape[2]

    
    from typing import Any, Callable, Optional, Tuple

class Agulhas2(torch.utils.data.Dataset):

    def __init__(self, split, joint_transform=None):
        super(Agulhas2, self).__init__()
        
        self.split = split
        self.inps_mean_std = (-0.4315792, 0.5710749)
        self.tars_mean_std = (2.9930247e-06, 0.009364196)
        self.tars_bm_mean_std = (-0.43158054, 0.57093924)

        self.inputs, self.targets, self.targets_bm = self._get_data_array()
        self.joint_transform = joint_transform

        self.inps_transform = T.Compose([T.ToTensor(), T.Normalize(*self.inps_mean_std)])
        self.tars_transform = T.Compose([T.ToTensor(), T.Normalize(*self.tars_mean_std)])
        self.tars_bm_transform = T.Compose([T.ToTensor(), T.Normalize(*self.tars_bm_mean_std)])

    def _get_data_array(self):
        
        inps_path = os.path.join('dataset', self.split, 'inputs.npy')
        tars_path = os.path.join('dataset', self.split, 'targets.npy')
        tars_bm_path = os.path.join('dataset', self.split, 'targets_bm.npy')
                                 
        with open(inps_path, 'rb') as f:
            inputs = np.load(f)
        
        with open(tars_path, 'rb') as f:
            targets = np.load(f)
        
        with open(tars_bm_path, 'rb') as f:
            targets_bm = np.load(f)
            
        return inputs, targets, targets_bm

    def __getitem__(self, index):
        
        x = self.inputs[..., index]
        y = self.targets[..., index]
        y_bm = self.targets_bm[..., index]
    
        x = self.inps_transform(x)
        y = self.tars_transform(y)
        y_bm = self.tars_bm_transform(y_bm)
        
        if self.joint_transform:
            x, y, y_bm = self.joint_transform(x, y, y_bm)

        return x, y, y_bm

    def __len__(self):
        return self.inputs.shape[2]