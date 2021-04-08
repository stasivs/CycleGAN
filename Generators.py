import torch
import torch.nn as nn

from Blocks import ResBlock
import Layers


class ResnetGenerator(nn.Module):
    def __init__(self, resnet_blocks=4):
        super(ResnetGenerator, self).__init__()

        self.img_channels = 3
        self.image_size = 256
        self.n = 64
        self.resnet_blocks = resnet_blocks

        self.encoder = self.return_generated_decode_block()
        self.transformer = self.return_generated_transform_block()
        self.decoder = self.return_generated_decode_block()

    def return_generated_encode_block(self):
        encode_block = nn.Sequential(
            nn.Conv2d(self.img_channels, self.n, 7, 1, 3),  # (bs, 64, 256, 256)
            nn.LeakyReLU(0.2, True),

            Layers.resnet_encoder_layer(self.n, self.n * 2, self.slope),  # (bs, 128, 128, 128)
            Layers.resnet_encoder_layer(self.n * 2, self.n * 4, self.slope),  # (bs, 256, 64, 64)
            Layers.resnet_encoder_layer(self.n * 4, self.n * 8, self.slope),  # (bs, 512, 32, 32)
        )

        return encode_block

    def return_generated_transform_block(self):
        transform_block = []
        for _ in range(self.resnet_blocks):
            rb = ResBlock(self.n*8, self.n*8)
            transform_block.append(rb)

        return nn.Sequential(*transform_block)

    def return_generated_decode_block(self):
        decode_block = nn.Sequential(
            Layers.resnet_decoder_layer(self.n * 8, self.n * 4, self.slope),  # (bs, 256, 64, 64)
            Layers.resnet_decoder_layer(self.n * 4, self.n * 2, self.slope),  # (bs, 128, 128, 128)
            Layers.resnet_decoder_layer(self.n * 2, self.n * 1, self.slope),  # (bs, 64, 256, 256)

            nn.Conv2d(self.n, self.img_channels, 7, 1, 3),  # (bs, 3, 256, 256)
            nn.Tanh()
        )

        return decode_block

    def forward(self, x):
        x = self.encoder(x)
        x = self.transformer(x)
        x = self.decoder(x)
        return x


class UNetGen(nn.Module):
    def __init__(self):
        super(UNetGen, self).__init__()



    def forward(self):
        pass



class ImprovedUNetGen(nn.Module):
    def __init__(self):
        super(ImprovedUNetGen, self).__init__()

        self.img_channels = 3
        self.image_size = 256
        self.n = 64
        self.slope = 0.02

        self.down_stack = nn.ModuleList([
            Layers.down_grade_layer(self.img_channels, self.n, self.slope, norm=False),  # (bs, 128, 128, 64)
            Layers.down_grade_layer(self.n, self.n * 2, self.slope),  # (bs, 64, 64, 128)
            Layers.down_grade_layer(self.n * 2, self.n * 4, self.slope),  # (bs, 32, 32, 256)
            Layers.down_grade_layer(self.n * 4, self.n * 8, self.slope),  # (bs, 16, 16, 512)
            Layers.down_grade_layer(self.n * 8, self.n * 8, self.slope),  # (bs, 8, 8, 512)
            Layers.down_grade_layer(self.n * 8, self.n * 8, self.slope),  # (bs, 4, 4, 512)
            Layers.down_grade_layer(self.n * 8, self.n * 8, self.slope),  # (bs, 2, 2, 512)
            Layers.down_grade_layer(self.n * 8, self.n * 8, self.slope),  # (bs, 1, 1, 512)
        ])

        self.up_stack = nn.ModuleList([
            Layers.up_grade_layer(self.n * 8, self.n * 8, self.slope),  # (bs, 2, 2, 1024)
            Layers.up_grade_layer(self.n * 16, self.n * 8, self.slope),  # (bs, 4, 4, 1024)
            Layers.up_grade_layer(self.n * 16, self.n * 8, self.slope),  # (bs, 8, 8, 1024)
            Layers.up_grade_layer(self.n * 16, self.n * 8, self.slope),  # (bs, 16, 16, 1024)
            Layers.up_grade_layer(self.n * 16, self.n * 4, self.slope),  # (bs, 32, 32, 512)
            Layers.up_grade_layer(self.n * 8, self.n * 2, self.slope),  # (bs, 64, 128, 256)
            Layers.up_grade_layer(self.n * 4, self.n, self.slope),  # (bs, 128, 128, 128)
        ])

        self.out = nn.Sequential(
            nn.ConvTranspose2d(self.n * 2, self.img_channels, 4, 2, 1, 0),  # (bs, 256, 256, 3)
            nn.Tanh()
        )

    def forward(self, x):
        skips = []
        for down in self.down_stack:
            x = down(x)
            skips.append(x)

        skips = skips[::-1][1:]
        for up, skip in zip(self.up_stack, skips):
            x = up(x)
            x = torch.cat([x, skip], dim=1)

        x = self.out(x)

        return x


