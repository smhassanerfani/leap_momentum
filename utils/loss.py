import torch
import torch.nn as nn
import torch.nn.functional as F


class LaplacianLoss(nn.Module):
    def __init__(self):
        super(LaplacianLoss, self).__init__()

    def forward(self, predicted, target):
        laplacian_kernel = torch.tensor([[1, 1, 1], [1, -8, 1], [1, 1, 1]], dtype=torch.float32)
        laplacian_kernel = laplacian_kernel.view(1, 1, 3, 3).to(target.device)

        predicted_laplacian = F.conv2d(predicted, laplacian_kernel, padding=1)
        target_laplacian = F.conv2d(target, laplacian_kernel, padding=1)

        # loss = F.l1_loss(input_laplacian, target_laplacian)
        loss = F.mse_loss(predicted_laplacian, target_laplacian)
        
        return loss


def prewitt_filter(input):
    # prewitt operator kernels
    kernel_x = torch.tensor([[+1, 0, -1], [+1, 0, -1], [+1, 0, -1]], dtype=torch.float32).view(1, 1, 3, 3)
    kernel_y = torch.tensor([[+1, +1, +1], [0, 0, 0], [-1, -1, -1]], dtype=torch.float32).view(1, 1, 3, 3)

    if input.is_cuda:
        kernel_x = kernel_x.cuda()
        kernel_y = kernel_y.cuda()

    # Compute gradients using Sobel filters
    gradient_x = F.conv2d(input, kernel_x, padding=1)
    gradient_y = F.conv2d(input, kernel_y, padding=1)

    return gradient_x, gradient_y


class PrewittLoss(nn.Module):
    def __init__(self):
        super(PrewittLoss, self).__init__()

    def forward(self, predicted, target):
        pred_gradient_x, pred_gradient_y = prewitt_filter(predicted)
        target_gradient_x, target_gradient_y = prewitt_filter(target)

        # Calculate the squared differences between predicted and target gradients
        diff_x = (pred_gradient_x - target_gradient_x) ** 2
        diff_y = (pred_gradient_y - target_gradient_y) ** 2

        # Combine x and y gradients and compute the mean
        sobel_loss = torch.mean(diff_x + diff_y)

        return sobel_loss


def sobel_filter(input):
    # Sobel operator kernels
    kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
    kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)

    if input.is_cuda:
        kernel_x = kernel_x.cuda()
        kernel_y = kernel_y.cuda()

    # Compute gradients using Sobel filters
    gradient_x = F.conv2d(input, kernel_x, padding=1)
    gradient_y = F.conv2d(input, kernel_y, padding=1)

    return gradient_x, gradient_y


class SobelLoss(nn.Module):
    def __init__(self):
        super(SobelLoss, self).__init__()

    def forward(self, predicted, target):
        pred_gradient_x, pred_gradient_y = sobel_filter(predicted)
        target_gradient_x, target_gradient_y = sobel_filter(target)

        # Calculate the squared differences between predicted and target gradients
        diff_x = (pred_gradient_x - target_gradient_x) ** 2
        diff_y = (pred_gradient_y - target_gradient_y) ** 2

        # Combine x and y gradients and compute the mean
        sobel_loss = torch.mean(diff_x + diff_y)

        return sobel_loss