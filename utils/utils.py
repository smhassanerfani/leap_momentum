import torch
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from matplotlib import figure
import xarray as xr
import csv
import numpy as np

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
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if lr is not None:
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr


def save_examples(x, y, y_fake, transform_params, counter, saving_path):
    import matplotlib
    matplotlib.use('Agg')
    
    # fig, axes = plt.subplots(nrows= x.shape[0], ncols=3, figsize=(12, 13), constrained_layout=True)
    
    fig = figure.Figure(figsize=(12, 13), constrained_layout=True)
    axes = fig.subplots(nrows= x.shape[0], ncols=3)
    
    for idx in range(x.shape[0]):
    
        image = x[idx].detach().cpu().numpy() * transform_params['inputs_std'] + transform_params['inputs_mean']
        target = y[idx].detach().cpu().numpy() * transform_params['targets_std'] + transform_params['targets_mean']
        gen_target = y_fake[idx].detach().cpu().numpy() * transform_params['targets_std'] + transform_params['targets_mean']

        xr.DataArray(image.squeeze(), dims=['x', 'y']).plot(x="x", y="y", robust=True, yincrease=False, ax=axes[idx, 0])
        xr.DataArray(target.squeeze(), dims=['x', 'y']).plot(x="x", y="y", robust=True, yincrease=False, ax=axes[idx, 1])
        xr.DataArray(gen_target.squeeze(), dims=['x', 'y']).plot(x="x", y="y", robust=True, yincrease=False, ax=axes[idx, 2])

    fig.savefig(f'{saving_path}/{counter}.png', format='png', bbox_inches='tight', pad_inches=0.1)
    # plt.close('all')
    
    del image, target, gen_target

def save_examples2(x, y, y_fake, transform_params, counter, saving_path):
    import matplotlib
    matplotlib.use('Agg')
    
    # fig, axes = plt.subplots(nrows= x.shape[0], ncols=3, figsize=(12, 13), constrained_layout=True)
    
    fig = figure.Figure(figsize=(20, 7), constrained_layout=True)
    axes = fig.subplots(nrows= x.shape[0], ncols=5)
    
    for idx in range(x.shape[0]):
    
        image = x[idx].detach().cpu().numpy() * transform_params['inputs_std'] + transform_params['inputs_mean']
        target_it = y[idx, 0, ...].detach().cpu().numpy() * transform_params['targets_std'] + transform_params['targets_mean']
        target_bm = y[idx, 1, ...].detach().cpu().numpy() * transform_params['targets_bm_std'] + transform_params['targets_bm_mean']
        gen_target_it = y_fake[idx, 0, ...].detach().cpu().numpy() * transform_params['targets_std'] + transform_params['targets_mean']
        gen_target_bm = y_fake[idx, 1, ...].detach().cpu().numpy() * transform_params['targets_bm_std'] + transform_params['targets_bm_mean']

        xr.DataArray(image.squeeze(), dims=['x', 'y']).plot(x="x", y="y", robust=True, yincrease=False, ax=axes[idx, 0])
        xr.DataArray(target_it.squeeze(), dims=['x', 'y']).plot(x="x", y="y", robust=True, yincrease=False, ax=axes[idx, 1])
        xr.DataArray(target_bm.squeeze(), dims=['x', 'y']).plot(x="x", y="y", robust=True, yincrease=False, ax=axes[idx, 3])
        xr.DataArray(gen_target_it.squeeze(), dims=['x', 'y']).plot(x="x", y="y", robust=True, yincrease=False, ax=axes[idx, 2])
        xr.DataArray(gen_target_bm.squeeze(), dims=['x', 'y']).plot(x="x", y="y", robust=True, yincrease=False, ax=axes[idx, 4])

    fig.savefig(f'{saving_path}/{counter}.png', format='png', bbox_inches='tight', pad_inches=0.1)
    # plt.close('all')
    
    del image, target_it, target_bm, gen_target_it, gen_target_bm


def save_examples3(x, y, y_fake, transform_params, counter, saving_path):
    import matplotlib
    matplotlib.use('Agg')
    
    fig = figure.Figure(figsize=(20, 7), constrained_layout=True)
    axes = fig.subplots(nrows= x.shape[0], ncols=5)
    
    for idx in range(x.shape[0]):
        
        image = (x[idx].detach().cpu().numpy() + 1) / 2
        
        target_it = (y[idx, 0, ...].detach().cpu().numpy() + 1) / 2
        target_bm = (y[idx, 1, ...].detach().cpu().numpy() + 1) / 2
        
        gen_target_it = (y_fake[idx, 0, ...].detach().cpu().numpy() + 1) / 2
        gen_target_bm = (y_fake[idx, 1, ...].detach().cpu().numpy() + 1) / 2
    
        image = image * (transform_params['inputs'][1] - transform_params['inputs'][0]) + transform_params['inputs'][0]
        
        target_it = target_it * (transform_params['targets_it'][1] - transform_params['targets_it'][0]) + transform_params['targets_it'][0]
        target_bm = target_bm * (transform_params['targets_bm'][1] - transform_params['targets_bm'][0]) + transform_params['targets_bm'][0]
        
        gen_target_it = gen_target_it * (transform_params['targets_it'][1] - transform_params['targets_it'][0]) + transform_params['targets_it'][0]
        gen_target_bm = gen_target_bm * (transform_params['targets_bm'][1] - transform_params['targets_bm'][0]) + transform_params['targets_bm'][0]

        xr.DataArray(image.squeeze(), dims=['x', 'y']).plot(x="x", y="y", robust=True, yincrease=False, ax=axes[idx, 0])
        xr.DataArray(target_it.squeeze(), dims=['x', 'y']).plot(x="x", y="y", robust=True, yincrease=False, ax=axes[idx, 1])
        xr.DataArray(target_bm.squeeze(), dims=['x', 'y']).plot(x="x", y="y", robust=True, yincrease=False, ax=axes[idx, 3])
        xr.DataArray(gen_target_it.squeeze(), dims=['x', 'y']).plot(x="x", y="y", robust=True, yincrease=False, ax=axes[idx, 2])
        xr.DataArray(gen_target_bm.squeeze(), dims=['x', 'y']).plot(x="x", y="y", robust=True, yincrease=False, ax=axes[idx, 4])

    fig.savefig(f'{saving_path}/{counter}.png', format='png', bbox_inches='tight', pad_inches=0.1)
    del image, target_it, target_bm, gen_target_it, gen_target_bm
    
    
def plot_examples(inp, tar, gen, saving_path=None):
    
    fig, axes = plt.subplots(nrows= 1, ncols=3, figsize=(12, 3), constrained_layout=True)

    xr.DataArray(inp.squeeze(), dims=['x', 'y']).plot(x="x", y="y", vmax=inp.max(), vmin=inp.min(), robust=True, yincrease=False, cmap='RdBu_r', ax=axes[0])
    axes[0].set_title('RAW SSH')
    
    xr.DataArray(tar.squeeze(), dims=['x', 'y']).plot(x="x", y="y", vmax=tar.max(), vmin=tar.min(), robust=True, yincrease=False, cmap='RdBu_r', ax=axes[1])
    axes[1].set_title('GT')
    
    xr.DataArray(gen.squeeze(), dims=['x', 'y']).plot(x="x", y="y", vmax=tar.max(), vmin=tar.min(), robust=True, yincrease=False, cmap='RdBu_r', ax=axes[2])
    axes[2].set_title('MODEL')
    
    if saving_path:
        plt.savefig(f'{saving_path}/{counter}.png', format='png', bbox_inches='tight', pad_inches=0.1)
        plt.close()

    

def qqplot(y_test, y_pred, yax1='Depth (m)' ,axis_names=None, site_name=None, quantiles=None):

    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(12, 4), constrained_layout=True)

    if axis_names is None:
        y_test_name='GT'
        y_pred_name='MODEL'
    else:
        y_test_name=axis_names[0]
        y_pred_name=axis_names[1]

    ax1.boxplot([y_test, y_pred])

    ax1.set_xticklabels([y_test_name, y_pred_name])
    ax1.tick_params(axis='x', labelrotation=0, labelsize=12)
    ax1.set_ylabel(yax1)
    ax1.grid(True)
    # ax1.set_title(f'BOX PLOT')

    x1 = np.sort(y_test)
    y1 = np.arange(1, len(y_test) + 1) / len(y_test)
    ax2.plot(x1, y1, linestyle='none', marker='o', alpha=0.2, label=y_test_name)
    # ax2.plot(x1, y1, linestyle='-', alpha=0.8, label='GT')

    x2 = np.sort(y_pred)
    y2 = np.arange(1, len(y_pred) + 1) / len(y_pred)
    # ax1.plot(x2, y2, linestyle='none', marker='.', alpha=0.5, label='GT')
    ax2.plot(x2, y2, linestyle='-.', alpha=1, label=y_pred_name)

    # ax2.set_title(f'ECDF')
    ax2.legend()

    if quantiles is None:
        quantiles = min(len(y_test), len(y_pred))
    quantiles = np.linspace(start=0, stop=1, num=int(quantiles))

    x_quantiles = np.quantile(y_test, quantiles, method='nearest')
    y_quantiles = np.quantile(y_pred, quantiles, method='nearest')

    ax3.scatter(x_quantiles, y_quantiles)
    # ax3.plot([0, 100], [0, 100], '--', color = 'black', linewidth=1.5)

    max_value = np.array((x_quantiles, y_quantiles)).max()
    ax3.plot([0, max_value], [0, max_value], '--', color='black', linewidth=1.5)

    ax3.set_xlabel(y_test_name)
    ax3.set_ylabel(y_pred_name)
    # ax3.set_title(f'Q-Q PLOT')

    if site_name is not None:
        plt.savefig(f'./{site_name}.pdf', format='pdf', bbox_inches='tight', pad_inches=0.05)

    plt.show()
