import torch
from torchvision.utils import save_image


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


def save_examples(x, y, y_fake, counter, saving_path):

    save_image(y_fake * 0.5 + 0.5, saving_path + f"/y_gen_{counter}.png")
    save_image(x * 0.5 + 0.5, saving_path + f"/input_{counter}.png")
    save_image(y * 0.5 + 0.5, saving_path + f"/label_{counter}.png")

