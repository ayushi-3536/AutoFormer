import math

import torch.nn as nn

from AutoFormer.model.swinIR.module import Conv2DSuper


class Upsample(nn.Sequential):
    """Upsample module.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. ' 'Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)


class UpsampleOneStep(nn.Module):
    """UpsampleOneStep module (the difference with Upsample is that it always only has 1conv + 1pixelshuffle)
       Used in lightweight SR to save parameters.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.

    """

    def __init__(self, scale, num_feat, num_out_ch, input_resolution=None):
        super().__init__()
        self.num_feat = num_feat
        self.input_resolution = input_resolution
        self.scale = scale
        self.out_ch = num_out_ch
        self.layers = nn.ModuleList([
            Conv2DSuper(num_feat, (scale ** 2) * num_out_ch, 3, 1, 1),
            nn.PixelShuffle(scale)
        ])

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.num_feat * 3 * 9
        return flops

    def set_sample_config(self,
                          sample_embed_dim=None
                          ):
        self.num_feat = sample_embed_dim
        # Access Conv2DSuper module and set_sample_config
        self.layers[0].set_sample_config(in_channels=self.num_feat,
                                         out_channels=(self.scale ** 2) * self.out_ch)

    def forward(self, x):
        for m in self.layers:
            x = m(x)
        return x

    def calc_sampled_param_num(self):
        # all submodules are calculated in their respective classes
        return 0
