# -----------------------------------------------------------------------------------
# SwinIR: Image Restoration Using Swin Transformer, https://arxiv.org/abs/2108.10257
# Originally Written by Ze Liu, Modified by Jingyun Liang.
# -----------------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

import logging
from AutoFormer.lib.utils import calc_dropout
from AutoFormer.model.swinIR.module import WindowAttention, Mlp, LayerNormSuper, \
    PatchUnEmbed, PatchEmbed, UpsampleOneStep, Upsample

logger = logging.getLogger('swinIR_model')


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: LayerNormSuper
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=LayerNormSuper):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = LayerNormSuper(self.dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = LayerNormSuper(self.dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            attn_mask = self.calculate_mask(self.input_resolution)
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def calculate_mask(self, x_size):
        # calculate attention mask for SW-MSA
        H, W = x_size
        img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        return attn_mask

    def forward(self, x, x_size):
        if self.is_identity_layer:
            return x

        H, W = x_size
        B, L, C = x.shape
        # assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA (to be compatible for testing on images whose shapes are the multiple of window size
        if self.input_resolution == x_size:
            attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C
        else:
            attn_windows = self.attn(x_windows, mask=self.calculate_mask(x_size).to(x.device))

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops

    def set_sample_config(self,
                          is_identity_layer=False,
                          sample_embed_dim=None,
                          sample_mlp_ratio=None,
                          sample_num_heads=None,
                          sample_dropout=None,
                          sample_out_dim=None,
                          sample_attn_dropout=None
                          ):

        if is_identity_layer:
            self.is_identity_layer = True
            return

        self.is_identity_layer = False

        self.attn.set_sample_config(sample_embed_dim, sample_num_heads)
        self.norm1.set_sample_config(sample_embed_dim)
        self.norm2.set_sample_config(sample_embed_dim)
        self.mlp.set_sample_config(sample_embed_dim, sample_mlp_ratio, sample_dropout)

    def calc_sampled_param_num(self):
        # all submodules are calculated in their respective classes
        return 0


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: LayerNormSuper
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=LayerNormSuper, downsample=None, use_checkpoint=False):

        super().__init__()

        self.dim = dim
        self.input_resolution = input_resolution

        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, x_size):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, x_size)
            else:
                x = blk(x, x_size)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops

    def set_sample_config(self,
                          sample_stl_num,
                          sample_embed_dim=None,
                          sample_mlp_ratio=None,
                          sample_num_heads=None,
                          sample_dropout=None,
                          sample_out_dim=None,
                          sample_attn_dropout=None
                          ):
        self.sample_stl_num = sample_stl_num
        self.sample_embed_dim = sample_embed_dim
        self.sample_out_dim = sample_out_dim
        self.sample_mlp_ratio = sample_mlp_ratio
        self.sample_num_heads = sample_num_heads
        self.sample_dropout = sample_dropout
        self.sample_attn_dropout = sample_attn_dropout

        for i in range(self.depth):
            if i < self.sample_stl_num:
                self.blocks[i].set_sample_config(is_identity_layer=False,
                                                 sample_embed_dim=sample_embed_dim,
                                                 sample_mlp_ratio=sample_mlp_ratio,
                                                 sample_num_heads=sample_num_heads,
                                                 sample_dropout=sample_dropout,
                                                 sample_out_dim=sample_out_dim,
                                                 sample_attn_dropout=sample_attn_dropout)
            else:
                self.blocks[i].set_sample_config(is_identity_layer=True)

        # todo: see path merging layer, downsample is None though

    def calc_sampled_param_num(self):
        # all submodules are calculated in their respective classes
        return 0


class RSTB(nn.Module):
    """Residual Swin Transformer Block (RSTB).

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: LayerNormSuper
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        img_size: Input image size.
        patch_size: Patch size.
        resi_connection: The convolutional block before residual connection.
    """

    def __init__(self, dim,
                 input_resolution, depth, num_heads, window_size,
                 output_dim=None,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=LayerNormSuper, downsample=None, use_checkpoint=False,
                 img_size=224, patch_size=4, resi_connection='1conv'):
        super(RSTB, self).__init__()

        self.dim = dim
        self.output_dim = self.dim if output_dim == None else output_dim
        self.input_resolution = input_resolution

        # the configs of current sampled arch
        self.sample_embed_dim = None
        self.sample_mlp_ratio = None
        self.sample_ffn_embed_dim_this_layer = None
        self.sample_num_stl_this_layer = None
        self.sample_scale = None
        self.sample_dropout = None
        self.sample_attn_dropout = None
        self.conv_sample_weight = None
        self.conv_sample_bias = None

        self.residual_group = BasicLayer(dim=dim,
                                         input_resolution=input_resolution,
                                         depth=depth,
                                         num_heads=num_heads,
                                         window_size=window_size,
                                         mlp_ratio=mlp_ratio,
                                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                                         drop=drop, attn_drop=attn_drop,
                                         drop_path=drop_path,
                                         norm_layer=norm_layer,
                                         downsample=downsample,
                                         use_checkpoint=use_checkpoint)

        if resi_connection == '1conv':
            self.conv = nn.Conv2d(dim, self.output_dim, 3, 1, 1)
        elif resi_connection == '3conv':
            # to save parameters and memory
            self.conv = nn.Sequential(nn.Conv2d(dim, dim // 4, 3, 1, 1),
                                      nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                      nn.Conv2d(dim // 4, dim // 4, 1, 1, 0),
                                      nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                      nn.Conv2d(dim // 4, dim, 3, 1, 1))

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim,
            norm_layer=None)

        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim,
            norm_layer=None)

    def set_sample_config(self,
                          is_identity_layer,
                          sample_stl_num=None,
                          sample_embed_dim=None,
                          sample_mlp_ratio=None,
                          sample_num_heads=None,
                          sample_dropout=None,
                          sample_attn_dropout=None,
                          sample_out_dim=None):

        if is_identity_layer:
            self.is_identity_layer = True
            return

        self.is_identity_layer = False

        self.sample_embed_dim = sample_embed_dim
        self.sample_out_dim = sample_out_dim
        self.sample_mlp_ratio = sample_mlp_ratio
        self.sample_num_heads_this_layer = sample_num_heads

        self.sample_dropout = sample_dropout
        self.sample_attn_dropout = sample_attn_dropout

        self.residual_group.set_sample_config(sample_stl_num=sample_stl_num,
                                              sample_embed_dim=sample_embed_dim,
                                              sample_mlp_ratio=sample_mlp_ratio,
                                              sample_num_heads=sample_num_heads,
                                              sample_dropout=sample_dropout,
                                              sample_out_dim=sample_out_dim,
                                              sample_attn_dropout=sample_attn_dropout)
        self.patch_embed.set_sample_config(embed_dim=sample_out_dim)
        # Todo: see conv weights using sampled weights for the last layer
        #
        logger.debug(f'embed_dim:{self.sample_embed_dim}')
        self.conv_sample_weight = self.conv.weight[:self.sample_out_dim, :self.sample_embed_dim, ...]
        self.conv_sample_bias = self.conv.bias[:self.sample_out_dim, ...]

        # Todo:see path unembedding set sample config
        self.patch_unembed.set_sample_config(embed_dim=sample_embed_dim)

    def forward(self, x, x_size):
        if self.is_identity_layer:
            return x
        basic_layer_x = self.residual_group(x, x_size)
        path_unembed_x = self.patch_unembed(basic_layer_x, x_size)
        conv_x = F.conv2d(path_unembed_x,
                          self.conv_sample_weight, self.conv_sample_bias,
                          stride=1,
                          padding=self.conv.padding,
                          dilation=self.conv.dilation)
        path_embed_x = self.patch_embed(conv_x)
        path_embed_x = path_embed_x + x
        return path_embed_x

    def flops(self):
        flops = 0
        flops += self.residual_group.flops()
        H, W = self.input_resolution
        flops += H * W * self.dim * self.dim * 9
        flops += self.patch_embed.flops()
        flops += self.patch_unembed.flops()

        return flops

    def calc_sampled_param_num(self):
        # all submodules are calculated in their respective
        logger.debug("RSTB block")
        if self.conv_sample_weight is not None:
            weight = self.conv_sample_weight.numel()
        else:
            weight = 0

        if self.conv_sample_bias is not None:
            bias = self.conv_sample_bias.numel()
        else:
            bias = 0

        return weight + bias


class SwinIR(nn.Module):
    r""" SwinIR
        A PyTorch impl of : `SwinIR: Image Restoration Using Swin Transformer`, based on Swin Transformer.

    Args:
        img_size (int | tuple(int)): Input image size. Default 64
        patch_size (int | tuple(int)): Patch size. Default: 1
        in_chans (int): Number of input image channels. Default: 3
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: LayerNormSuper.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
        upscale: Upscale factor. 2/3/4/8 for image SR, 1 for denoising and compress artifact reduction
        img_range: Image range. 1. or 255.
        upsampler: The reconstruction reconstruction module. 'pixelshuffle'/'pixelshuffledirect'/'nearest+conv'/None
        resi_connection: The convolutional block before residual connection. '1conv'/'3conv'
    """

    def __init__(self, img_size=64, patch_size=1, in_chans=3,
                 embed_dim=96, depths=[6, 6, 6, 6], num_heads=[6, 6, 6, 6],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=LayerNormSuper, ape=False, patch_norm=True,
                 use_checkpoint=False, upscale=2, img_range=1., upsampler='', resi_connection='1conv',
                 **kwargs):
        super(SwinIR, self).__init__()
        num_in_ch = in_chans
        num_out_ch = in_chans
        num_feat = 64
        self.img_range = img_range
        if in_chans == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)
        self.upscale = upscale
        self.upsampler = upsampler
        self.window_size = window_size
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate

        #####################################################################################################
        ################################### 1, shallow feature extraction ###################################
        self.conv_first = nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1)

        #####################################################################################################
        ################################### 2, deep feature extraction ######################################
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio

        # configs for the sampled subTransformer
        self.sample_rstb_num = None
        self.sample_embed_dim = None
        self.sample_mlp_ratio = None
        self.sample_stl_num = None
        self.sample_num_heads = None
        self.sample_dropout = None
        self.conv_sample_weight = None
        self.conv_sample_bias = None
        # self.sample_output_dim = None

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # merge non-overlapping patches into image
        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build Residual Swin Transformer blocks (RSTB)
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = RSTB(dim=embed_dim,
                         input_resolution=(patches_resolution[0],
                                           patches_resolution[1]),
                         depth=depths[i_layer],
                         num_heads=num_heads[i_layer],
                         window_size=window_size,
                         mlp_ratio=self.mlp_ratio,
                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                         drop=drop_rate, attn_drop=attn_drop_rate,
                         drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],  # no impact on SR results
                         norm_layer=norm_layer,
                         downsample=None,
                         use_checkpoint=use_checkpoint,
                         img_size=img_size,
                         patch_size=patch_size,
                         resi_connection=resi_connection

                         )
            self.layers.append(layer)
        self.norm = norm_layer(self.num_features)

        # build the last conv layer in deep feature extraction
        if resi_connection == '1conv':
            self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        elif resi_connection == '3conv':
            # to save parameters and memory
            self.conv_after_body = nn.Sequential(nn.Conv2d(embed_dim, embed_dim // 4, 3, 1, 1),
                                                 nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                                 nn.Conv2d(embed_dim // 4, embed_dim // 4, 1, 1, 0),
                                                 nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                                 nn.Conv2d(embed_dim // 4, embed_dim, 3, 1, 1))

        ################################ 3, high quality image reconstruction ################################
        if self.upsampler == 'pixelshuffle':
            # for classical SR
            self.conv_before_upsample = nn.Sequential(nn.Conv2d(embed_dim, num_feat, 3, 1, 1),
                                                      nn.LeakyReLU(inplace=True))
            self.upsample = Upsample(upscale, num_feat)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        elif self.upsampler == 'pixelshuffledirect':
            # for lightweight SR (to save parameters)
            self.upsample = UpsampleOneStep(upscale, embed_dim, num_out_ch,
                                            (patches_resolution[0], patches_resolution[1]))
        elif self.upsampler == 'nearest+conv':
            # for real-world SR (less artifacts)
            self.conv_before_upsample = nn.Sequential(nn.Conv2d(embed_dim, num_feat, 3, 1, 1),
                                                      nn.LeakyReLU(inplace=True))
            self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            if self.upscale == 4:
                self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
            self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        else:
            # for image denoising and JPEG compression artifact reduction
            self.conv_last = nn.Conv2d(embed_dim, num_out_ch, 3, 1, 1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, LayerNormSuper):
            nn.init.constant_(m.norm.bias, 0)
            nn.init.constant_(m.norm.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
        mod_pad_w = (self.window_size - w % self.window_size) % self.window_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def set_sample_config(self, config: dict):
        """Sets the sample config for SwinIR

        An example config:
        {
            'rstb_num': 4,                  # Num of RSTB layers/blocks in the network
            'stl_num': 6,                   # Num STL blocks per RSTB layer
            'embed_dim': [36, 36, 60, 60]   # Per RSTB layer
            'mlp_ratio': [2., 2., 1.5, 1.5] # Per RSTB layer
            'num_heads': [6, 6, 6, 6]       # Per RSTB layer
        }
        So, all STL blocks are same within one RSTB layer.
        The variation is between RSTB blocks.
        """
        self.sample_rstb_num = config['rstb_num']
        self.sample_embed_dim = config['embed_dim']
        self.sample_mlp_ratio = config['mlp_ratio']
        self.sample_stl_num = config['stl_num']
        self.sample_num_heads = config['num_heads']

        print("rstb_num", self.sample_rstb_num)
        print("embed dim", self.sample_embed_dim)
        print("mlp ration", self.sample_mlp_ratio)
        print("stl num", self.sample_stl_num)
        print("num heads", self.sample_num_heads)
        self.sample_dropout = calc_dropout(self.drop_rate, self.sample_embed_dim[0], self.embed_dim)

        self.patch_embed.set_sample_config(self.sample_embed_dim[0])

        self.conv_sample_weight = self.conv_first.weight[:self.sample_embed_dim[0], ...]
        self.conv_sample_bias = self.conv_first.bias[:self.sample_embed_dim[0], ...]

        self.sample_output_dim = [out_dim for out_dim in self.sample_embed_dim[1:]] + [self.sample_embed_dim[-1]]
        for i, layer in enumerate(self.layers):
            # not exceed sample layer number
            if i < self.sample_rstb_num:
                sample_dropout = calc_dropout(self.drop_rate, self.sample_embed_dim[i], self.embed_dim)
                sample_attn_dropout = calc_dropout(self.attn_drop_rate, self.sample_embed_dim[i], self.embed_dim)

                layer.set_sample_config(is_identity_layer=False,
                                        sample_stl_num=self.sample_stl_num,
                                        sample_embed_dim=self.sample_embed_dim[i],
                                        sample_mlp_ratio=self.sample_mlp_ratio[i],
                                        sample_num_heads=self.sample_num_heads[i],
                                        sample_dropout=sample_dropout,
                                        sample_out_dim=self.sample_output_dim[i],
                                        sample_attn_dropout=sample_attn_dropout)
            # exceeds sample layer number
            else:
                layer.set_sample_config(is_identity_layer=True)

        self.patch_unembed.set_sample_config(embed_dim=self.sample_embed_dim[-1])
        self.norm.set_sample_config(self.sample_embed_dim[-1])

        self.conv_after_body_sample_weight = self.conv_after_body.weight[:self.sample_embed_dim[-1],
                                             :self.sample_embed_dim[0], ...]
        self.conv_after_body_sample_bias = self.conv_after_body.bias[:self.sample_embed_dim[-1], ...]
        self.upsample.set_sample_config(sample_embed_dim=self.sample_embed_dim[-1])

    def forward_features(self, x):
        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = F.dropout(x, p=self.sample_dropout, training=self.training)

        for layer in self.layers:
            x = layer(x, x_size)

        x = self.norm(x)  # B L C
        x = self.patch_unembed(x, x_size)

        return x

    def forward(self, x):
        H, W = x.shape[2:]
        x = self.check_image_size(x)
        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range
        if self.upsampler == 'pixelshuffle':
            # for classical SR
            x = F.conv2d(x,
                         self.conv_sample_weight, self.conv_sample_bias,
                         stride=1,
                         padding=self.conv_first.padding,
                         dilation=self.conv_first.dilation)

            x = self.conv_after_body(self.forward_features(x)) + x
            x = self.conv_before_upsample(x)
            x = self.conv_last(self.upsample(x))
        elif self.upsampler == 'pixelshuffledirect':
            # for lightweight SR
            x = F.conv2d(x,
                         self.conv_sample_weight, self.conv_sample_bias,
                         stride=1,
                         padding=self.conv_first.padding,
                         dilation=self.conv_first.dilation)

            forward_x = self.forward_features(x)
            dfe_x = F.conv2d(forward_x,
                             self.conv_after_body_sample_weight,
                             self.conv_after_body_sample_bias,
                             stride=1,
                             padding=self.conv_after_body.padding,
                             dilation=self.conv_after_body.dilation
                             )
            dfe_x = dfe_x + x
            x = self.upsample(dfe_x)
        elif self.upsampler == 'nearest+conv':
            # for real-world SR
            x = F.conv2d(x,
                         self.conv_sample_weight, self.conv_sample_bias,
                         stride=1,
                         padding=self.conv_first.padding,
                         dilation=self.conv_first.dilation)
            x = self.conv_after_body(self.forward_features(x)) + x
            x = self.conv_before_upsample(x)
            x = self.lrelu(self.conv_up1(torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')))
            if self.upscale == 4:
                x = self.lrelu(self.conv_up2(torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')))
            x = self.conv_last(self.lrelu(self.conv_hr(x)))
        else:
            # for image denoising and JPEG compression artifact reduction
            x_first = F.conv2d(x,
                               self.conv_sample_weight, self.conv_sample_bias,
                               stride=1,
                               padding=self.conv_first.padding,
                               dilation=self.conv_first.dilation)
            res = self.conv_after_body(self.forward_features(x_first)) + x_first
            x = x + self.conv_last(res)

        x = x / self.img_range + self.mean
        return x[:, :, :H * self.upscale, :W * self.upscale]

    def flops(self):
        flops = 0
        H, W = self.patches_resolution
        flops += H * W * 3 * self.embed_dim * 9
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += H * W * 3 * self.embed_dim * self.embed_dim
        flops += self.upsample.flops()
        return flops

    def calc_sampled_param_num(self):
        # all submodules are calculated in their respective classes
        # These conv layers are not wrapped with Conv2DSuper
        return self.conv_sample_weight.numel() + self.conv_sample_bias.numel() \
               + self.conv_after_body_sample_weight.numel() \
               + self.conv_after_body_sample_bias.numel()

    def get_sampled_params_numel(self, config):
        self.set_sample_config(config)
        numels = []
        for name, module in self.named_modules():
            # print("name", name)
            # print("has calc",hasattr(module, 'calc_sampled_param_num'))
            if hasattr(module, 'calc_sampled_param_num'):
                name_splits = name.split('.')
                if name_splits[0] == 'layers':
                    # E.g. of a relevant module: `layers.3.residual_group.blocks.4.attn`

                    # If layer is outside range, skip
                    if int(name_splits[1]) >= config['rstb_num']:
                        continue

                    # If block is outside range, skip
                    if len(name_splits) > 3 and name_splits[3] == 'blocks' and int(name_splits[4]) >= config['stl_num']:
                        continue

                params = module.calc_sampled_param_num()
                # print("name", name, "params",params)
                numels.append(params)

        return sum(numels)


if __name__ == '__main__':
    upscale = 4
    window_size = 8
    height = (1024 // upscale // window_size + 1) * window_size
    width = (720 // upscale // window_size + 1) * window_size
    model = SwinIR(upscale=2, img_size=(height, width),
                   window_size=window_size, img_range=1., depths=[6, 6, 6, 6],
                   embed_dim=60, num_heads=[6, 6, 6, 6], mlp_ratio=2, upsampler='pixelshuffledirect')
    logger.debug(f'obtained model:{model}')
    logger.debug(f'height:{height}, width:{width}, params:{model.flops() / 1e9}')

    x = torch.randn((1, 3, height, width))
    x = model(x)
    logger.debug(f'output shape:{x.shape}')
