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

    
class Agulhas2(torch.utils.data.Dataset):

    def __init__(self, split, joint_transform=None):
        super(Agulhas2, self).__init__()
        
        self.split = split
        self.inps_mean_std = (-0.4315792, 0.5710749)
        self.tars_mean_std = (2.9930247e-06, 0.009364196)
        self.tars_bm_mean_std = (-0.43158054, 0.57093924)
        
        self.transform_dict = {
            'inputs' : (-0.4315792, 0.5710749),
            'targets_it': (2.9930247e-06, 0.009364196),
            'targets_bm': (-0.43158054, 0.57093924)
                              }

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
    
class Agulhas3(torch.utils.data.Dataset):

    def __init__(self, split, joint_transform=None):
        super(Agulhas3, self).__init__()
        
        self.split = split
        self.inps_min_max = (-2.0309153, 1.098078)
        self.tars_min_max = (-0.121434465, 0.12447834)
        self.tars_bm_min_max = (-2.032202, 1.0936853)
        
        self.transform_dict = {'inputs' : (-2.0309153, 1.098078),
                                'targets_it': (-0.121434465, 0.12447834),
                                'targets_bm': (-2.032202, 1.0936853)
                              }
        
        self.joint_transform = joint_transform
        self.transform = T.ToTensor()
        
        self.inputs, self.targets, self.targets_bm = self._get_data_array()
    
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
    
        x = (x - self.inps_min_max[0]) / (self.inps_min_max[1] - self.inps_min_max[0])
        y = (y - self.tars_min_max[0]) / (self.tars_min_max[1] - self.tars_min_max[0])
        y_bm = (y_bm - self.tars_bm_min_max[0]) / (self.tars_bm_min_max[1] - self.tars_bm_min_max[0])
        
        x = 2 * x - 1
        y = 2 * y - 1
        y_bm = 2 * y_bm - 1
        
        x = self.transform(x)
        y = self.transform(y)
        y_bm = self.transform(y_bm)
        
        if self.joint_transform:
            x, y, y_bm = self.joint_transform(x, y, y_bm)

        return x, y, y_bm
    
    
    def __len__(self):
        return self.inputs.shape[2]

    
class Agulhas4(torch.utils.data.Dataset):

    def __init__(self, split, joint_transform=None):
        super(Agulhas4, self).__init__()
        
        self.split = split
        self.ssh_min_max = (-2.0346121788024902, 1.1185286045074463)
        self.bms_min_max = (-2.0395283699035645, 1.1078916788101196)
        self.its_min_max = (-0.1476919949054718, 0.12447834014892578)
        
        self.transform_dict = {'ssh': self.ssh_min_max,
                               'bms': self.bms_min_max,
                               'its': self.its_min_max
                              }
        
        self.joint_transform = joint_transform
        self.transform = T.ToTensor()
        
        self.ssh, self.bms, self.its = self._get_data_array()
    
    
    def _get_data_array(self):
        
        import os
        import xarray as xr

        PERSISTENT_BUCKET = os.environ['PERSISTENT_BUCKET']
        data_path = f'{PERSISTENT_BUCKET}/LLC4320/dataset/{self.split}.zarr'

        ds = xr.open_zarr(data_path)
                
        return ds['SSH'], ds['BMs'], ds['ITs']
        
        
    def __getitem__(self, index):
        
        x  = self.ssh[..., index].values
        bm = self.bms[..., index].values
        it = self.its[..., index].values
    
        x  = (x - self.ssh_min_max[0]) / (self.ssh_min_max[1] - self.ssh_min_max[0])
        bm = (bm - self.bms_min_max[0]) / (self.bms_min_max[1] - self.bms_min_max[0])
        it = (it - self.its_min_max[0]) / (self.its_min_max[1] - self.its_min_max[0])
        
        x = 2 * x - 1
        bm = 2 * bm - 1
        it = 2 * it - 1
        
        x  = self.transform(x)
        bm = self.transform(bm)
        it = self.transform(it)           
            
        if self.joint_transform:
            x, bm, it = self.joint_transform(x, bm, it)

        return x, bm, it
        
        
    def __len__(self):
        return self.ssh.shape[0]
    

def main():
    dataset = Agulhas4('val', joint_transform=None)
    print(len(dataset))
    dataiter = iter(dataset)
    x, y1, y2 = next(dataiter)
    print(x.shape, y1.shape, y2.shape)

    
if __name__ == '__main__':
    main()