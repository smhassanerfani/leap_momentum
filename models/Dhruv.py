import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, img_channels=1, num_features=4):
        super().__init__()
        self.lyr1 = nn.Conv2d(img_channels, num_features, kernel_size=3, stride=1, bias=False, padding=1, padding_mode='reflect')
        self.bn1  = nn.InstanceNorm2d(num_features, affine=True)
        self.act1 = nn.ReLU()
        
        self.lyr2 = nn.Conv2d(num_features, num_features*2, kernel_size=3, stride=1, bias=False, padding=1, padding_mode='reflect')
        self.bn2  = nn.InstanceNorm2d(num_features*2, affine=True)
        self.act2 = nn.ReLU()
        
        self.lyr3 = nn.Conv2d(num_features*2, num_features*4, kernel_size=3, stride=1, bias=False, padding=1, padding_mode='reflect')
        self.bn3  = nn.InstanceNorm2d(num_features*4, affine=True)
        self.act3 = nn.ReLU()
        
        
        self.lyr4 = nn.Conv2d(num_features*4, num_features*8, kernel_size=3, stride=1, bias=False, padding=1, padding_mode='reflect')
        self.bn4  = nn.InstanceNorm2d(num_features*8, affine=True)
        self.act4 = nn.ReLU()
        
        
        self.lyr5 = nn.Conv2d(num_features*8, num_features, kernel_size=3, stride=1, bias=False, padding=1, padding_mode='reflect')
        self.bn4  = nn.InstanceNorm2d(num_features, affine=True)
        self.act5 = nn.ReLU()
        
        self.lyr6 = nn.Conv2d(num_features, 1, kernel_size=3, stride=1, padding=1, padding_mode='reflect')
          
    
    def forward(self, x):
        
        output = self.lyr1(x)
        output = self.act1(output)

        output = self.lyr2(output)
        output = self.act2(output)

        output = self.lyr3(output)
        output = self.act3(output)

        output = self.lyr4(output)
        output = self.act4(output)

        output = self.lyr5(output)
        output = self.act5(output)

        output = self.lyr6(output)
            
        return torch.tanh(output)

    
class DnCNN(nn.Module):
    def __init__(self, in_channels=1, num_features=64):
        super().__init__()
        self.initialization = nn.Sequential(
            nn.Conv2d(in_channels, num_features, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.PReLU()
        )
    
        self.cnn = nn.Sequential(
            self._block(num_features, num_features*2, 3, 1, 1),
            self._block(num_features*2, num_features*2, 3, 1, 1),
            self._block(num_features*2, num_features*2, 3, 1, 1),
            self._block(num_features*2, num_features*2, 3, 1, 1),
            self._block(num_features*2, num_features*2, 3, 1, 1),
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
        return out


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=7, padding=(3,3)), nn.PReLU(),
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=7, padding=(3,3)), nn.PReLU(),
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=7, padding=(3,3)), nn.PReLU(),
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=7, padding=(3,3)), nn.PReLU(),
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=7, padding=(3,3)), nn.PReLU()
                     )
    def forward(self, x):
        return self.cnn(x)

def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
        elif isinstance(m, (nn.BatchNorm2d)):
            pass
    
def main():
    x = torch.randn((4, 1, 256, 256))
    y = torch.randn((4, 1, 256, 256))
    print(x.shape)
    model = DnCNN()
    print(model(x).shape)
    
if __name__ == '__main__':
    main()