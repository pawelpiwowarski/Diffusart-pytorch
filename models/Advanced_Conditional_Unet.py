import math
from inspect import isfunction
from functools import partial
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from einops import rearrange
import torch
from torch import nn, einsum
import torch.nn.functional as F
from .Advanced_Network_Helpers import *
from transformers import PreTrainedModel


class Unet(nn.Module):
    def __init__(
        self,
        dim,
        init_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        channels=3,
        with_time_emb=True,
        resnet_block_groups=8,
        use_convnext=True,
        convnext_mult=2,
    ):
        super().__init__()

        # determine dimensions
        self.channels = channels  # since we are concatenating the images and the conditionings along the channel dimension

        init_dim = default(init_dim, dim // 3 * 2)
        self.init_conv = nn.Conv2d(self.channels * 2, init_dim, 7, padding=3)
        self.conditioning_init = nn.Conv2d(self.channels, init_dim, 7, padding=3)
        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        self.in_out = in_out

        if use_convnext:
            block_klass = partial(ConvNextBlock, mult=convnext_mult)
        else:
            block_klass = partial(ResnetBlock, groups=resnet_block_groups)

        # time embeddings
        if with_time_emb:
            time_dim = dim * 4
            self.time_mlp = nn.Sequential(
                SinusoidalPositionEmbeddings(dim),
                nn.Linear(dim, time_dim),
                nn.GELU(),
                nn.Linear(time_dim, time_dim),
            )
        else:
            time_dim = None
            self.time_mlp = None

        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        self.conditioning_encoder = nn.ModuleList([])
        num_resolutions = len(in_out)
        self.num_resolutions = num_resolutions

        # conditioning encoder
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.conditioning_encoder.append(
                nn.ModuleList(
                    [
                        block_klass(dim_in, dim_out),
                        Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                        Downsample(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                nn.ModuleList(
                    [
                        block_klass(dim_in, dim_out, time_emb_dim=time_dim),
                        block_klass(dim_out, dim_out, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                        Downsample(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )

        mid_dim = dims[-1]

        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.cross_attention_1 = Residual(
            PreNorm(mid_dim, LinearCrossAttention(mid_dim))
        )
        self.cross_attention_2 = Residual(
            PreNorm(mid_dim, LinearCrossAttention(mid_dim))
        )
        self.cross_attention_3 = Residual(
            PreNorm(mid_dim, LinearCrossAttention(mid_dim))
        )
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)
            self.ups.append(
                nn.ModuleList(
                    [
                        block_klass(dim_out * 2, dim_in, time_emb_dim=time_dim),
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                        Upsample(dim_in) if not is_last else nn.Identity(),
                    ]
                )
            )

        out_dim = default(out_dim, channels)
        self.final_conv = nn.Sequential(
            block_klass(dim, dim), nn.Conv2d(dim, out_dim, 1)
        )

    def forward(self, x, time, implicit_conditioning, explicit_conditioning):
        x = torch.cat((x, explicit_conditioning), dim=1)

        x = self.init_conv(x)

        conditioning = self.conditioning_init(implicit_conditioning)

        t = self.time_mlp(time) if exists(self.time_mlp) else None

        h = []

        # conditioning encoder

        for block1, attn, downsample in self.conditioning_encoder:
            conditioning = block1(conditioning)
            conditioning = attn(conditioning)
            conditioning = downsample(conditioning)

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            x = block2(x, t)
            x = attn(x)
            h.append(x)
            x = downsample(x)

        # reverse the c list

        # bottleneck

        x = self.cross_attention_1(x, conditioning)
        x = self.mid_block1(x, t)
        x = self.cross_attention_2(x, conditioning)
        x = self.mid_block2(x, t)
        x = self.cross_attention_3(x, conditioning)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)
            x = block2(x, t)
            x = attn(x)
            x = upsample(x)

        return self.final_conv(x)
