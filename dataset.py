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

    
class MapDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        X_generator,
        y_generator,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        joint_transform: Optional[Callable] = None,
    ) -> None:
        """
        PyTorch Dataset adapter for Xbatcher

        Parameters
        ----------
        X_generator : xbatcher.BatchGenerator
        y_generator : xbatcher.BatchGenerator
        transform : callable, optional
            A function/transform that takes in an array and returns a transformed version.
        target_transform : callable, optional
            A function/transform that takes in the target and transforms it.
        """
        self.X_generator = X_generator
        self.y_generator = y_generator
        self.transform = transform
        self.target_transform = target_transform
        self.joint_transform = joint_transform

    def __len__(self) -> int:
        return len(self.X_generator)

    def __getitem__(self, idx) -> Tuple[Any, Any]:
        if torch.is_tensor(idx):
            idx = idx.tolist()
            if len(idx) == 1:
                idx = idx[0]
            else:
                raise NotImplementedError(
                    f"{type(self).__name__}.__getitem__ currently requires a single integer key"
                )

        X_batch = self.X_generator[idx].values
        y_batch = self.y_generator[idx].values

        if self.transform:
            X_batch = self.transform(X_batch)

        if self.target_transform:
            y_batch = self.target_transform(y_batch)
        
        if self.joint_transform:
            X_batch, y_batch = self.joint_transform([X_batch, y_batch])
        
        return X_batch, y_batch