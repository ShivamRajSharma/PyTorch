import torch 
import torch.nn as nn

class DownSamplerBlock_(nn.Module):
    def __init__(self, input_channels, out_channels):
        super(DownSamplerBlock_, self).__init__()
        self.residual = nn.Conv2d(input_channels, out_channels, kernel_size=1)
        self.downsample = nn.Sequential(
            nn.Conv2d(input_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        a = x
        x = self.downsample(x)
        a = self.residual(a)
        x = nn.LeakyReLU()(x + a)
        return x

class DownSampler(nn.Module):
    def __init__(self, input_channels):
        super(DownSampler, self).__init__()
        self.channels = [32, 64, 128, 256, 512]
        self.channels.insert(0, input_channels)
        self.layers = nn.Sequential(
            *[
                DownSamplerBlock_(self.channels[num], self.channels[num+1])
                for num in range(len(self.channels)-1)
            ]
        )
        self.max_pool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        past = []
        for num, layer in enumerate(self.layers):
            x = layer(x)
            if num+1 != len(self.channels) -1:
                past.append(x)
                x = self.max_pool(x)
        return x, past

    
class UpSamplerBlock(nn.Module):
    def __init__(self, input_channels, out_channels):
        super(UpSamplerBlock, self).__init__()
        self.upsample = nn.ConvTranspose2d(
            input_channels, 
            out_channels, 
            kernel_size=2, 
            stride=2
        )
        self.residual = nn.Conv2d(input_channels, out_channels, kernel_size=1)
        self.out = nn.Sequential(
            nn.Conv2d(input_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU()
        )
    
    def forward(self, x, past):
        x = self.upsample(x)
        x = torch.cat((x, past), dim=1)
        a = x
        x = self.out(x)
        a = self.residual(a)
        x = nn.LeakyReLU()(x+a)
        return x

class UpSampler(nn.Module):
    def __init__(self):
        super(UpSampler, self).__init__()
        self.channels = [512, 256, 128, 64, 32]
        self.layers = nn.Sequential(
            *[
                UpSamplerBlock(self.channels[num], self.channels[num+1])
                for num in range(len(self.channels) -1)
            ]
        )

    def forward(self, x, past):
        past = past[::-1]
        for num, layer in enumerate(self.layers):
            x = layer(x, past[num])
        return x

class UNet(nn.Module):
    def __init__(self, input_channels):
        super(UNet, self).__init__()
        self.down = DownSampler(input_channels)
        self.up = UpSampler()
        self.conv1 = nn.Conv2d(32, 1, kernel_size=1)
    
    def forward(self, x):
        x, past = self.down(x)
        x = self.up(x, past)
        x = self.conv1(x)
        return x

if __name__ == "__main__":
    a = torch.randn(1, 3, 512, 512)
    model = UNet(3)
    print(a.shape)
    y = model(a)
    print(y.shape)
