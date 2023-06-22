import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


class Agulhas(Dataset):

    def __init__(self, split, joint_transform=None):
        super(Agulhas, self).__init__()
        self.split = split

        self.inputs, self.targets = self._get_data_array()
        self.joint_transform = joint_transform

        self.input_transforms = T.Compose([T.ToTensor(), T.Normalize(*self.inputs_mean_std)])
        self.target_transforms = T.Compose([T.ToTensor(), T.Normalize(*self.targets_mean_std)])

    def _get_data_array(self):
        
        import xarray as xr
        
        link1 = 'gs://leap-persistent/dhruvbalwada/ssh_reconstruction_project/unfiltered_data.zarr'
        link2 = 'gs://leap-persistent/dhruvbalwada/ssh_reconstruction_project/filtered_data.zarr'
        ds_unfiltered = xr.open_zarr(link1)
        ds_filtered = xr.open_zarr(link2)
        ds_it = ds_unfiltered['ssh_unfiltered'] - ds_filtered['ssh_filtered']

        rng = np.random.default_rng(1948)
        arr = np.arange(70)
        rng.shuffle(arr)
        
        # if self.split == 'train':
        inputs = ds_unfiltered['ssh_unfiltered'][arr[:49], 56:-56, 56:-56].fillna(0) 
        targets = ds_it[arr[:49], 56:-56, 56:-56].fillna(0) 

        self.inputs_mean_std = (inputs.mean().compute().item(), inputs.std().compute().item())
        self.targets_mean_std = (targets.mean().compute().item(), targets.std().compute().item())
        
        if self.split == 'val':
            inputs = ds_unfiltered['ssh_unfiltered'][arr[49:56], 56:-56, 56:-56].fillna(0) 
            targets = ds_it[arr[49:56], 56:-56, 56:-56].fillna(0)
        
        if self.split == 'test': # test set:
            inputs = ds_unfiltered['ssh_unfiltered'][arr[56:], 56:-56, 56:-56].fillna(0) 
            targets = ds_it[arr[56:], 56:-56, 56:-56].fillna(0)
        
        inputs = inputs.coarsen(i=256, j=256, boundary="pad").construct(i=("x_coarse", "x_fine"), j=("y_coarse", "y_fine"))
        inputs = inputs.stack(z=("x_coarse", "y_coarse"))[..., [i for i in range(64) if i not in [47, 54, 55, 62, 63]]]
        inputs = inputs.reset_index("x_coarse").stack(z_time=("time", "z"))

        targets = targets.coarsen(i=256, j=256, boundary="pad").construct(i=("x_coarse", "x_fine"), j=("y_coarse", "y_fine"))
        targets = targets.stack(z=("x_coarse", "y_coarse"))[..., [i for i in range(64) if i not in [47, 54, 55, 62, 63]]]
        targets = targets.reset_index("x_coarse").stack(z_time=("time", "z"))
            
        return inputs.values, targets.values

    def __getitem__(self, index):
        
        x = self.inputs[..., index]
        y = self.targets[..., index]
    
        x = self.input_transforms(x)
        y = self.target_transforms(y)
        
        if self.joint_transform:
            x, y = self.joint_transform(x, y)

        return x, y

    def __len__(self):
        return self.inputs.shape[2]
