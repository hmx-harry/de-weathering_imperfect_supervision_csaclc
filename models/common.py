import torchvision
from utils.dysample import *

class vgg_19(nn.Module):
    def __init__(self, layer):
        super(vgg_19, self).__init__()
        vgg_model = torchvision.models.vgg19(pretrained=True)
        self.fea_extractor = nn.Sequential(*list(vgg_model.features.children())[:layer])

    def forward(self, x):
        return self.fea_extractor(x)

class UpSample(nn.Module):
    def __init__(self, scale, in_ch, out_ch, k_size=3, stride=1, bias=False, alpha=1e-2):
        super(UpSample, self).__init__()
        self.scale = scale
        self.dysample = DySample(in_channels=in_ch, scale=scale)
        self.up_seq = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=k_size,
                      stride=stride, padding=k_size // 2, bias=bias),
            nn.BatchNorm2d(out_ch),
            #nn.LeakyReLU(negative_slope=alpha, inplace=True),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale, mode='bilinear', align_corners=True)
        #x = self.dysample(x)
        return self.up_seq(x)

class UpDysample(nn.Module):
    def __init__(self, scale, in_ch, out_ch, k_size=3, stride=1, bias=False, alpha=1e-2):
        super(UpDysample, self).__init__()
        self.scale = scale
        self.dysample = DySample(in_channels=in_ch, scale=scale)
        self.up_seq = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=k_size,
                      stride=stride, padding=k_size // 2, bias=bias),
            nn.BatchNorm2d(out_ch),
            #nn.LeakyReLU(negative_slope=alpha, inplace=True),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        x = self.dysample(x)
        return self.up_seq(x)

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k_size=3, stride=1, bias=False, alpha=1e-2):
        super(ResBlock, self).__init__()
        #self.leakyrelu = nn.LeakyReLU(negative_slope=alpha, inplace=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv_seq = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=k_size,
                      stride=stride, padding=k_size//2, bias=bias),
            nn.BatchNorm2d(out_ch),
            #nn.LeakyReLU(negative_slope=alpha, inplace=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=k_size,
                      stride=stride, padding=k_size // 2, bias=bias),
            nn.BatchNorm2d(out_ch)
        )

    def forward(self, x):
        out = self.conv_seq(x)
        return self.relu(out + x)


class Conv(nn.Module):
    def __init__(self, in_ch, out_ch, k_size, stride, bias=False, alpha=1e-2):
        super(Conv, self).__init__()
        self.conv_seq = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=k_size,
                      stride=stride, padding=k_size // 2, bias=bias),
            nn.BatchNorm2d(out_ch),
            #nn.LeakyReLU(negative_slope=alpha, inplace=True)
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv_seq(x)

class DepScp_Conv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DepScp_Conv, self).__init__()
        self.depth_conv = nn.Conv2d(
            in_channels=in_channel,
            out_channels=in_channel,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=in_channel
        )
        self.point_conv = nn.Conv2d(
            in_channels=in_channel,
            out_channels=out_channel,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1
        )

    def forward(self, x):
        x1 = self.depth_conv(x)
        out = self.point_conv(x1)
        return out
