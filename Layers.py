import torch.nn as nn

def resnet_encoder_layer(in_channels, out_channels, slope):
    kernel_size = 4
    stride = 2
    padding = 1

    block = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
        nn.InstanceNorm2d(out_channels),
        nn.LeakyReLU(slope, True)
    )
    return block


def resnet_decoder_layer(in_channels, out_channels, slope):
    kernel_size = 3
    stride = 2
    padding = 1
    out_padding = 1

    block = nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, out_padding),
        nn.InstanceNorm2d(out_channels),
        nn.LeakyReLU(slope, True)
    )
    return block


def down_grade_layer(inp, out, slope, norm=True):
    kernel_size = 4
    stride = 2
    padding = 1

    block = nn.ModuleList([
        nn.Conv2d(inp, out, kernel_size, stride, padding, bias=False),
    ])
    if norm:
        block.append(nn.BatchNorm2d(out))
    block.append(nn.LeakyReLU(slope, True))

    return nn.Sequential(*block)



def up_grade_layer(inp, out, slope, dropout=False):
    kernel_size = 4
    stride = 2
    padding = 1
    out_padding = 0

    block = nn.ModuleList([
        nn.ConvTranspose2d(inp, out, kernel_size, stride, padding, out_padding, bias=False),
        nn.BatchNorm2d(out),
    ])
    if dropout:
        block.append(nn.Dropout2d)

    block.append(nn.LeakyReLU(slope))

    return nn.Sequential(*block)


if __name__ == "__main__":
    up_grade_layer(3, 3, 0.02)
    down_grade_layer(3, 3, 0.02)
    resnet_encoder_layer(3, 3, 0.02)
    resnet_decoder_layer(3, 3, 0.02)
