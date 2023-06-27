import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, img_channels=1, num_features=4):
        super().__init__()
        self.lyr1 = nn.Conv2d(img_channels, num_features, kernel_size=3, stride=1, padding=1, padding_mode='reflect')
        self.act1 = nn.ReLU()
        
        self.lyr2 = nn.Conv2d(num_features, num_features*2, kernel_size=3, stride=1, padding=1, padding_mode='reflect')
        self.act2 = nn.ReLU()
        
        self.lyr3 = nn.Conv2d(num_features*2, num_features*4, kernel_size=3, stride=1, padding=1, padding_mode='reflect')
        self.act3 = nn.ReLU()
        
        
        self.lyr4 = nn.Conv2d(num_features*4, num_features*8, kernel_size=3, stride=1, padding=1, padding_mode='reflect')
        self.act4 = nn.ReLU()
        
        
        self.lyr5 = nn.Conv2d(num_features*8, num_features, kernel_size=3, stride=1, padding=1, padding_mode='reflect')
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

def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.kaiming_normal_(m.weight.data)
    
def main():
    x = torch.randn((4, 1, 256, 256))
    y = torch.randn((4, 1, 256, 256))
    print(x.shape)
    model = Generator()
    print(model(x).shape)
    
if __name__ == '__main__':
    main()