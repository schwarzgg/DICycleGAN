import torch
import torch.nn as nn

from models.CoordAttention import CoordAtt
from models.SkFusion import SKFusion


class Generator(nn.Module):
    def __init__(self, in_channel, out_channel=3, dim=64, res_block=9):
        super(Generator, self).__init__()

        # Encoder
        down1 = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channel, dim, kernel_size=7, stride=1, padding=0),
            nn.InstanceNorm2d(dim),
            nn.ReLU(inplace=True)
        ]
        down2 = [
            nn.Conv2d(dim, dim * 2, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(dim * 2),
            nn.ReLU(inplace=True)
        ]
        down3 = [
            nn.Conv2d(dim * 2, dim * 4, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(dim * 4),
            nn.ReLU(inplace=True)
        ]

        # Converter
        res = [ResidualBlock(in_channel=dim * 4) for _ in range(res_block)]

        # Decoder
        up1 = [
            nn.ConvTranspose2d(dim * 4, dim * 2, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(dim * 2),
            nn.ReLU(inplace=True)
        ]
        self.sk1 = SKFusion(dim * 2)
        up2 = [
            nn.ConvTranspose2d(dim * 2, dim, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(dim),
            nn.ReLU(inplace=True)
        ]
        self.sk2 = SKFusion(dim)
        up3 = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(dim, out_channel, kernel_size=7, stride=1, padding=0),
            nn.Tanh()
        ]
        self.down1 = nn.Sequential(*down1)
        self.down2 = nn.Sequential(*down2)
        self.down3 = nn.Sequential(*down3)

        self.res = nn.Sequential(*res)
        self.up1 = nn.Sequential(*up1)
        self.up2 = nn.Sequential(*up2)
        self.up3 = nn.Sequential(*up3)

    def forward(self, input):
        down1 = self.down1(input)
        down2 = self.down2(down1)
        down3 = self.down3(down2)

        res = self.res(down3)

        up1 = self.up1(res)

        # up2 = self.up2(up1)
        # output = self.up3(up2)
        sk1 = self.sk1([up1, down2])
        up2 = self.up2(sk1)
        sk2 = self.sk2([up2, down1])
        output = self.up3(sk2)

        return output


class ResidualBlock(nn.Module):
    def __init__(self, in_channel):
        super(ResidualBlock, self).__init__()

        res_block = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=0, bias=True),
            nn.InstanceNorm2d(in_channel),
            nn.ReLU(inplace=True),
        ]

        att_block = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=0, bias=True),
            nn.InstanceNorm2d(in_channel),

            CoordAtt(in_channel, in_channel)
        ]
        #
        self.res_block = nn.Sequential(*res_block)
        self.att_block = nn.Sequential(*att_block)

    def forward(self, input):
        res = input + self.res_block(input)
        output = res + self.att_block(res)
        return output


if __name__ == '__main__':
    gen = Generator(in_channel=3, out_channel=3, dim=64)
    res = ResidualBlock(in_channel=3)

    # writer = SummaryWriter("../logs/model")

    # B C H W
    input_tensor = torch.randn(1, 3, 256, 256)
    output_tensor = gen(input_tensor)
    output_tensor2 = res(input_tensor)

    # writer.add_graph(gen,input_tensor)
    # writer.add_graph(res,input_tensor)

    print(res)
    print(gen)
    print(input_tensor.shape)
    print(output_tensor.shape)
    print(output_tensor2.shape)
