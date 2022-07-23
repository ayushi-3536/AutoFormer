import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerNormSuper(nn.Module):
    """Applies Layer Normalization with ability to sample weights

    For now only supports one dimensional `normalization_shape`
    unlike `nn.LayerNorm`
    """

    def __init__(self, normalized_dim: int):
        super(LayerNormSuper, self).__init__()

        self.normalized_dim = normalized_dim
        self.norm = nn.LayerNorm(normalized_shape=normalized_dim)

        self.sample_normalized_dim = normalized_dim
        self.sample_weight = None
        self.sample_bias = None

    def set_sample_config(self, normalized_dim):
        self.sample_normalized_dim = normalized_dim
        self.sample_weight = self.norm.weight[:normalized_dim]
        self.sample_bias = self.norm.bias[:normalized_dim]

    def forward(self, x: torch.Tensor):
        return F.layer_norm(x,
                            tuple((self.sample_normalized_dim,)),
                            self.sample_weight, self.sample_bias, self.norm.eps)

    def calc_sampled_param_num(self):
        if self.sample_weight is not None:
            weight = self.sample_weight.numel()
        else:
            weight = 0
        if self.sample_bias is not None:
            bias = self.sample_bias.numel()
        else:
            bias = 0

        return weight + bias
