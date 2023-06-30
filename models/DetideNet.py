import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.rep = nn.ReplicationPad2d(1)
        self.conv1 = nn.Conv2d(in_channels, 2*int(out_channels), 3, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.LeakyReLU(0.2,inplace=True) 
        self.conv2 = nn.Conv2d(2*int(out_channels), int(out_channels), 3, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(2*int(out_channels))

        nn.init.kaiming_normal_(self.conv1.weight.data)
        nn.init.kaiming_normal_(self.conv2.weight.data)
        nn.init.ones_(self.bn1.weight.data)
        nn.init.zeros_(self.bn2.weight.data) ### Trick
        nn.init.zeros_(self.bn1.bias.data)
        nn.init.zeros_(self.bn2.bias.data) 
    
        
    def forward(self, x):
        residual = x        
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(self.rep(out))
        
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(self.rep(out))
        out = out + residual
        return out


nbf = 64

class IGWResNet(nn.Module):
    def __init__(self):
        super(IGWResNet, self).__init__() 
        self.rep = nn.ReplicationPad2d(1)
        self.FirstLayer = nn.Conv2d(1, nbf, 3, padding=0, bias=True)
        self.list_1 = nn.ModuleList([])
        for i in range(5):
            self.list_1.append(ResidualBlock(nbf, nbf))
            
        nn.init.kaiming_normal_(self.FirstLayer.weight.data)    
        self.LastLayer = nn.Conv2d(nbf, 1, 1, padding=0, bias=False)
        nn.init.zeros_(self.FirstLayer.bias.data)
        
    def forward(self, x):
        out = self.FirstLayer(self.rep(x))
        for i in range(5):
            out = self.list_1[i](out)
        out = self.LastLayer(out)
        return out - torch.mean(out, dim=(1,2,3), keepdim=True)