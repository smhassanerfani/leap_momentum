import torch
import torch.nn as nn

class DnCNN_v0(nn.Module):
    
    def __init__(self, in_channels=1, num_features=16):
        super().__init__()
        self.initialization = nn.Sequential(
            nn.Conv2d(in_channels, num_features, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.PReLU()
        )
    
        self.cnn = nn.Sequential(
            self._block(num_features, num_features, 3, 1, 1),
            self._block(num_features, num_features, 3, 1, 1),
            self._block(num_features, num_features, 3, 1, 1),
            self._block(num_features, num_features, 3, 1, 1),
            self._block(num_features, num_features, 3, 1, 1),
            self._block(num_features, num_features, 3, 1, 1)
        )
        
        self.head = nn.Conv2d(num_features, in_channels, kernel_size=3, padding=1, padding_mode='reflect')
    
    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, padding_mode='reflect', bias=False),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.PReLU()
        )
    
    def forward(self, x):
        
        out = self.initialization(x)
        out = self.cnn(out)
        out = self.head(out) + x
        
        return torch.tanh(out)


class DnCNN_v1(nn.Module):
    def __init__(self, in_channels=1, num_features=64):
        super().__init__()
        self.initialization = nn.Sequential(
            nn.Conv2d(in_channels, num_features, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.PReLU()
        )
    
        self.cnn = nn.Sequential(
            self._block(num_features, num_features*2, 3, 1, 1),
            self._block(num_features*2, num_features*4, 3, 1, 1),
            self._block(num_features*4, num_features*4, 3, 1, 1),
            self._block(num_features*4, num_features*4, 3, 1, 1),
            self._block(num_features*4, num_features*2, 3, 1, 1),
            self._block(num_features*2, num_features, 3, 1, 1)
        )
        
        self.head = nn.Conv2d(num_features, in_channels, kernel_size=3, padding=1, padding_mode='reflect')
    
    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, padding_mode='reflect', bias=False),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.PReLU()
        )
    
    def forward(self, x):
        
        out = self.initialization(x)
        out = self.cnn(out)
        out = self.head(out) + x
        return torch.tanh(out)

class DnCNN(nn.Module):
    def __init__(self, in_channels=1, num_features=64):
        super().__init__()
        self.initialization = nn.Sequential(
            nn.Conv2d(in_channels, num_features, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.PReLU()
        )
    
        self.cnn = nn.Sequential(
            self._block(num_features, num_features, 3, 1, 1),
            self._block(num_features, num_features*2, 3, 1, 1),
            self._block(num_features*2, num_features*4, 3, 1, 1),
            self._block(num_features*4, num_features*4, 3, 1, 1),
            self._block(num_features*4, num_features*2, 3, 1, 1),
            self._block(num_features*2, num_features, 3, 1, 1)
        )
        
        self.head = nn.Conv2d(num_features, in_channels, kernel_size=3, padding=1, padding_mode='reflect')
    
    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, padding_mode='reflect', bias=False),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.PReLU()
        )
    
    def forward(self, x):
        
        out = self.initialization(x)
        out = self.cnn(out)
        out = self.head(out)
        
        BM = out + x
        IT = -1 * out
        
        return torch.tanh(BM), torch.tanh(IT) 


def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
            nn.init.constant_(m.weight.data, 1.25)
    
def main():
    x = torch.randn((4, 1, 256, 256))
    y = torch.randn((4, 1, 256, 256))
    print(x.shape)
    model = DnCNN()
    print(model(x).shape)
    
if __name__ == '__main__':
    main()