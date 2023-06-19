import torch
from torchvision.utils import save_image
import matplotlib.pyplot as plt

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


def save_examples(x, y, y_fake, dataset, counter, saving_path):

    # save_image(x * dataset.inputs_mean_std[1] + dataset.inputs_mean_std[0], saving_path + f"/input_{counter}.png")
    # save_image(y_fake * dataset.targets_mean_std[1] + dataset.targets_mean_std[1], saving_path + f"/y_gen_{counter}.png")
    # save_image(y * dataset.targets_mean_std[1] + dataset.targets_mean_std[1], saving_path + f"/label_{counter}.png")
    fig, axes = plt.subplots(nrows= x.shape[0], ncols=3, figsize=(14, 15), constrained_layout=True)
    for idx in range(x.shape[0]):
        image = x[idx].detach().cpu().numpy() * dataset.inputs_mean_std[1] + dataset.inputs_mean_std[0]
        target = y[idx].detach().cpu().numpy() * dataset.targets_mean_std[1] + dataset.targets_mean_std[0]
        gen_target = y_fake[idx].detach().cpu().numpy() * dataset.targets_mean_std[1] + dataset.targets_mean_std[0]
        axes[idx, 0].imshow(image.transpose(1, 2, 0))
        axes[idx, 1].imshow(target.transpose(1, 2, 0))
        axes[idx, 2].imshow(gen_target.transpose(1, 2, 0))
    plt.savefig(f'{saving_path}/{counter}.png', format='png', bbox_inches='tight', pad_inches=0.1)

