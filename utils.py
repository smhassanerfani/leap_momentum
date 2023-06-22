import torch
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import xarray as xr
import csv

def csv_writer(logger, save_path):

    with open(f"{save_path}/loss_log.csv", 'a+', newline='') as filehandler:

        w = csv.DictWriter(filehandler, logger.keys())
        w.writerow(logger)
        
#     fieldnames = logger[0].keys()

#     with open(f"{save_path}/loss_log.csv", 'a+', newline='') as filehandler:
#         fh_writer = csv.DictWriter(filehandler, fieldnames=fieldnames)

#         fh_writer.writeheader()
#         for item in logger:
#             fh_writer.writerow(item)


class AdjustLearningRate:
    num_of_iterations = 0

    def __init__(self, optimizer, base_lr, max_iter, power):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.max_iter = max_iter
        self.power = power

    def __call__(self, current_iter):
        lr = self.base_lr * ((1 - float(current_iter) / self.max_iter) ** self.power)
        self.optimizer.param_groups[0]['lr'] = lr
        if len(self.optimizer.param_groups) > 1:
            self.optimizer.param_groups[1]['lr'] = lr * 10

        return lr


def save_checkpoint(model, optimizer, filename):

    print("Saving Checkpoint...")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)

def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("Loading Checkpoint...")
    checkpoint = torch.load(checkpoint_file)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def save_examples(x, y, y_fake, transform_params, counter, saving_path):
    
    fig, axes = plt.subplots(nrows= x.shape[0], ncols=3, figsize=(12, 13), constrained_layout=True)
    
    for idx in range(x.shape[0]):
    
        image = x[idx].detach().cpu().numpy() * transform_params['inputs_std'] + transform_params['inputs_mean']
        target = y[idx].detach().cpu().numpy() * transform_params['targets_std'] + transform_params['targets_mean']
        gen_target = y_fake[idx].detach().cpu().numpy() * transform_params['targets_std'] + transform_params['targets_mean']

        xr.DataArray(image.squeeze(), dims=['x', 'y']).plot(x="x", y="y", robust=True, yincrease=False, ax=axes[idx, 0])
        xr.DataArray(target.squeeze(), dims=['x', 'y']).plot(x="x", y="y", robust=True, yincrease=False, ax=axes[idx, 1])
        xr.DataArray(gen_target.squeeze(), dims=['x', 'y']).plot(x="x", y="y", robust=True, yincrease=False, ax=axes[idx, 2])

    plt.savefig(f'{saving_path}/{counter}.png', format='png', bbox_inches='tight', pad_inches=0.1)
    plt.close()
    
    del x, y, y_fake

