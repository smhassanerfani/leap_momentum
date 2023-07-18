import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.CNN = nn.Sequential(
            nn.Conv2d(1, 8, 7, padding=(3, 3), padding_mode='reflect'), nn.PReLU(),
            nn.Conv2d(8, 8, 7, padding=(3, 3), padding_mode='reflect'), nn.PReLU(),
            nn.Conv2d(8, 8, 7, padding=(3, 3), padding_mode='reflect'), nn.PReLU(),
            nn.Conv2d(8, 8, 7, padding=(3, 3), padding_mode='reflect'), nn.PReLU(),
            nn.Conv2d(8, 2, 7, padding=(3, 3), padding_mode='reflect'), nn.Tanh()
        )
    def forward(self, x):
        x = self.CNN(x)
        return x
        