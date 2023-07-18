import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, second_component=False, head_activation=None): # nn.Tanh()
        super(CNN, self).__init__()
        self.CNN = nn.Sequential(
            nn.Conv2d(1, 8, 7, padding=(3, 3), padding_mode='reflect'), nn.PReLU(),
            nn.Conv2d(8, 8, 7, padding=(3, 3), padding_mode='reflect'), nn.PReLU(),
            nn.Conv2d(8, 8, 7, padding=(3, 3), padding_mode='reflect'), nn.PReLU(),
            nn.Conv2d(8, 8, 7, padding=(3, 3), padding_mode='reflect'), nn.PReLU(),
            nn.Conv2d(8, 2, 7, padding=(3, 3), padding_mode='reflect') if second_component else nn.Conv2d(8, 1, 7, padding=(3, 3), padding_mode='reflect'),
            head_activation if head_activation else nn.Identity()
        )
    def forward(self, x):
        x = self.CNN(x)
        return x
        