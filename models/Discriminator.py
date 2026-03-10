import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, in_channel, out_channel=1, dim=64):
        super(Discriminator, self).__init__()

        model = [
            nn.Conv2d(in_channel, dim, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(dim, dim * 2, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(dim * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(dim * 2, dim * 4, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(dim * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(dim * 4, dim * 8, kernel_size=4, stride=1, padding=1),
            nn.InstanceNorm2d(dim * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(dim * 8, out_channel, kernel_size=4, stride=1, padding=1)
        ]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        input = self.model(input)
        output = nn.functional.avg_pool2d(input, input.size()[2:])
        output = output.view(input.size()[0], -1)
        return output


if __name__ == '__main__':
    input = torch.randn(1, 3, 256, 256)

    disc = Discriminator(in_channel=3)

    output = disc(input)

    print(input.shape)
    print(output.shape)
    print(disc)
