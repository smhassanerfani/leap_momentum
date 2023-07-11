import torch
import torch.nn as nn


class DBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride=2):
        super(DBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=stride, bias=False, padding=1, padding_mode='reflect'),
            # nn.BatchNorm2d(out_channels),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class Discriminator(nn.Module):

    def __init__(self, in_channels=3, features=[64, 128, 256, 512]):
        super(Discriminator, self).__init__()
        self.initial_layer = nn.Sequential(
            nn.Conv2d(in_channels*2, features[0], kernel_size=4, stride=2, padding=1, padding_mode='reflect'),
            nn.LeakyReLU(0.2, inplace=True)
        )

        layers = []
        in_channels = features[0]
        for feature in features[1:]:
            layers.append(
                DBlock(in_channels, feature, stride=1 if feature == features[-1] else 2)
            )
            in_channels = feature

        layers.append(
            nn.Conv2d(in_channels, 1, kernel_size=4, stride=1, padding=1, padding_mode='reflect')
        )

        self.model = nn.Sequential(*layers)

    def forward(self, x, y):
        out = self.initial_layer(torch.concat([x, y], dim=1))
        return self.model(out)


class GBlock(nn.Module):

    def __init__(self, in_channels, out_channels, encoder=True, act='ReLU', use_dropout=False):
        super(GBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False, padding_mode='reflect')
            if encoder
            else nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            # nn.BatchNorm2d(out_channels),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.ReLU() if act == 'ReLU' else nn.LeakyReLU(0.2)
        )
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout2d(0.5)

    def forward(self, x):
        x = self.conv(x)
        return self.dropout(x) if self.use_dropout else x


class Generator(nn.Module):

    def __init__(self, in_channels, features=64):
        super(Generator, self).__init__()
        self.initial_encoder = nn.Sequential(
            nn.Conv2d(in_channels, features, kernel_size=4, stride=2, padding=1, padding_mode='reflect'),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.encode1 = GBlock(features, features*2, encoder=True, act='LeakyReLU', use_dropout=False)
        self.encode2 = GBlock(features*2, features*4, encoder=True, act='LeakyReLU', use_dropout=False)
        self.encode3 = GBlock(features*4, features*8, encoder=True, act='LeakyReLU', use_dropout=False)
        self.encode4 = GBlock(features*8, features*8, encoder=True, act='LeakyReLU', use_dropout=False)
        self.encode5 = GBlock(features*8, features*8, encoder=True, act='LeakyReLU', use_dropout=False)
        self.encode6 = GBlock(features*8, features*8, encoder=True, act='LeakyReLU', use_dropout=False)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(features*8, features*8, kernel_size=4, stride=2, padding=1, padding_mode='reflect'),
            # nn.ReLU()
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.decode1 = GBlock(features*8, features*8, encoder=False, act='ReLU', use_dropout=True)
        self.decode2 = GBlock(features*8*2, features*8, encoder=False, act='ReLU', use_dropout=True)
        self.decode3 = GBlock(features*8*2, features*8, encoder=False, act='ReLU', use_dropout=True)
        self.decode4 = GBlock(features*8*2, features*8, encoder=False, act='ReLU', use_dropout=False)
        self.decode5 = GBlock(features*8*2, features*4, encoder=False, act='ReLU', use_dropout=False)
        self.decode6 = GBlock(features*4*2, features*2, encoder=False, act='ReLU', use_dropout=False)
        self.decode7 = GBlock(features*2*2, features, encoder=False, act='ReLU', use_dropout=False)
        self.final_decode = nn.Sequential(
            nn.ConvTranspose2d(features*2, in_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        en0 = self.initial_encoder(x)
        en1 = self.encode1(en0)
        en2 = self.encode2(en1)
        en3 = self.encode3(en2)
        en4 = self.encode4(en3)
        en5 = self.encode5(en4)
        en6 = self.encode6(en5)
        bn = self.bottleneck(en6)
        de1 = self.decode1(bn)
        de2 = self.decode2(torch.cat([de1, en6], dim=1))
        de3 = self.decode3(torch.cat([de2, en5], dim=1))
        de4 = self.decode4(torch.cat([de3, en4], dim=1))
        de5 = self.decode5(torch.cat([de4, en3], dim=1))
        de6 = self.decode6(torch.cat([de5, en2], dim=1))
        de7 = self.decode7(torch.cat([de6, en1], dim=1))
        return self.final_decode(torch.cat([de7, en0], dim=1))
  
    
def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
def main():
    x = torch.randn((1, 1, 256, 256))
    y = torch.randn((1, 1, 256, 256))
    disc = Discriminator(in_channels=1)
    print(disc(x, y).shape)
if __name__ == '__main__':
    main()