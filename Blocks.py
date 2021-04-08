import torch.nn as nn

class ResBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super(ResBlock, self).__init__()

        self.kernel_size = 3
        self.stride = 1
        self.padding = 1
        self.slope = 0.02

        self.block = nn.Sequential(
            nn.Conv2d(in_features, out_features, self.kernel_size,
                      self.stride, self.padding, bias=False),
            nn.InstanceNorm2d(out_features),
            nn.LeakyReLU(self.slope, True),

            nn.Conv2d(out_features, out_features, self.kernel_size,
                      self.stride, self.padding, bias=False),
            nn.InstanceNorm2d(out_features)
        )

        self.out = nn.LeakyReLU(self.slope)

    def forward(self, x):
        x = x + self.block(x)
        x = self.out(x)
        return x