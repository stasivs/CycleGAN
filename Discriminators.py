import torch.nn as nn

import Layers

class PatchGAN(nn.Module):
    def __init__(self):
        super(PatchGAN, self).__init__()

        self.img_channels = 3
        self.image_size = 256
        self.n = 64
        self.slope = 0.02

        self.main = nn.Sequential(
            Layers.down_grade_layer(3, self.n, False),  # (bs, 64, 128, 128)
            Layers.down_grade_layer(self.n, self.n * 2, self.slope),  # (bs, 128, 64, 64)
            Layers.down_grade_layer(self.n * 2, self.n * 4, self.slope),  # (bs, 256, 32, 32)

            nn.Conv2d(self.n * 4, self.n * 8, 4, 1, 1),  # 512 x 31x31
            nn.BatchNorm2d(self.n * 8),
            nn.LeakyReLU(self.slope, inplace=True),

        )
        self.out = nn.Sequential(
            nn.Conv2d(self.n * 8, 1, 4, 1, 0),  # 1 x 28x28
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.main(x)
        x = self.out(x)
        return x