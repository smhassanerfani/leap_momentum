import torch
import torch.nn as nn


class DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=stride, padding=1, padding_mode='reflect', bias=True),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        return self.conv(x)


class Discriminator(nn.Module):
    def __init__(self, in_channels=3, features=[64, 128, 256, 512]):
        super(Discriminator, self).__init__()

        self.initial_layer = nn.Sequential(
            nn.Conv2d(in_channels, features[0], kernel_size=4, stride=2, padding=1, padding_mode='reflect'),
            nn.LeakyReLU(0.2)
        )
        layers = []
        in_channels = features[0]
        for feature in features[1:]:
            layers.append(DBlock(in_channels, feature, stride=1 if feature == features[-1] else 2))
            in_channels = feature
        
        layers.append(nn.Conv2d(in_channels, 1, kernel_size=4, stride=1, padding=1, padding_mode='reflect'))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        out = self.initial_layer(x)
        return torch.sigmoid(self.layers(out))
   

class GBlock(nn.Module):
    def __init__(self, in_channels, out_channels, encoding=True, activation=True, **kwargs):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, **kwargs)
            if encoding
            else nn.ConvTranspose2d(in_channels, out_channels, **kwargs),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU() if activation else nn.Identity()
        )
    
    def forward(self, x):
        return self.conv(x)


class RBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self._block = nn.Sequential(
            GBlock(channels, channels, kernel_size=3, padding=1),
            GBlock(channels, channels, activation=False, kernel_size=3, padding=1)
        )
    
    def forward(self, x):
        return x + self._block(x)


class Generator(nn.Module):
    def __init__(self, img_channels=3, num_features=64, num_residuals=9):
        super().__init__()
        self.initial_layer = nn.Sequential(
            nn.Conv2d(img_channels, num_features, kernel_size=7, stride=1, padding=3, padding_mode='reflect'),
            nn.ReLU(inplace=True)
        )
        self.encoding_blocks = nn.ModuleList(
            [
                GBlock(num_features, num_features*2, kernel_size=3, stride=2, padding=1),
                GBlock(num_features*2, num_features*4, kernel_size=3, stride=2, padding=1)
            ]
        )
        self.residual_blocks = nn.Sequential(
            *[RBlock(num_features*4) for _ in range(num_residuals)]
        )
        self.decoding_blocks = nn.ModuleList(
            [
                GBlock(num_features*4, num_features*2, encoding=False, activation=True, kernel_size=3, stride=2, padding=1, output_padding=1),
                GBlock(num_features*2, num_features, encoding=False, activation=True, kernel_size=3, stride=2, padding=1, output_padding=1)
            ]
        )
        self.head_layer = nn.Conv2d(num_features, img_channels, kernel_size=7, stride=1, padding=3, padding_mode='reflect')
    
    def forward(self, x):
        out = self.initial_layer(x)
        for layer in self.encoding_blocks:
            out = layer(out)
        out = self.residual_blocks(out)
        for layer in self.decoding_blocks:
            out = layer(out)
        out = self.head_layer(out)
        return torch.tanh(out)


def main():
    x = torch.randn((2, 3, 256, 256))
    disc = Discriminator()
    gen = Generator()
    print(gen(x).shape)

if __name__ == '__main__':
    main()