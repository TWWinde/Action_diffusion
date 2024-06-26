import torch
import torch.nn as nn
import einops
from einops.layers.torch import Rearrange
import math

from .helpers import (
    SinusoidalPosEmb,
    Downsample1d,
    Upsample1d,
    Conv1dBlock,
)

class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)
        
def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

def normalization(channels):
    """
    Make a standard normalization layer.

    :param channels: number of input channels.
    :return: an nn.Module for normalization.
    """
    return GroupNorm32(32, channels)

class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.

    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(self, channels, num_heads=1, use_checkpoint=False):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.use_checkpoint = use_checkpoint

        self.norm = normalization(channels)
        #self.qkv = conv_nd(1, channels, channels * 3, 1)
        self.qkv = Conv1dBlock( channels, channels * 3, 2)
        
        self.attention = QKVAttention()
        #self.proj_out = zero_module(conv_nd(1, channels, channels, 1))
        self.proj_out = zero_module(Conv1dBlock(channels, channels, 4))

    def forward(self, x):
        #print(x.shape)
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        qkv = qkv.reshape(b * self.num_heads, -1, qkv.shape[2])
        h = self.attention(qkv)
        #print(h.shape, qkv.shape)
        h = h.reshape(b, -1, h.shape[-1])
        h = self.proj_out(h)
        #print(x.shape, h.shape)
        return (x + h).reshape(b, c, *spatial)


class QKVAttention(nn.Module):
    """
    A module which performs QKV attention.
    """

    def forward(self, qkv):
        """
        Apply QKV attention.

        :param qkv: an [N x (C * 3) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x C x T] tensor after attention.
        """
        ch = qkv.shape[1] // 3
        q, k, v = torch.split(qkv, ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        return torch.einsum("bts,bcs->bct", weight, v)


class ResidualTemporalBlock(nn.Module):

    def __init__(self, inp_channels, out_channels, embed_dim, kernel_size=3):
        super().__init__()

        self.blocks = nn.ModuleList([
            Conv1dBlock(inp_channels, out_channels, kernel_size),
            Conv1dBlock(out_channels, out_channels, kernel_size, if_zero=True)
        ])
        self.time_mlp = nn.Sequential(    # should be removed for Noise and Deterministic Baselines
            nn.Mish(),
            nn.Linear(embed_dim, out_channels),
            Rearrange('batch t -> batch t 1'),
        )
        self.residual_conv = nn.Conv1d(inp_channels, out_channels, 1) \
            if inp_channels != out_channels else nn.Identity()
            
        self.dropout = nn.Dropout(0.5)

    def forward(self, x, t):
        out = self.blocks[0](x) + self.time_mlp(t)   # for diffusion
        # out = self.blocks[0](x)    # for Noise and Deterministic Baselines
        out = self.blocks[1](out)
        return out + self.residual_conv(self.dropout(x))


class TemporalUnet(nn.Module):
    def __init__(
        self,
        transition_dim,
        num_class,
        dim=32,
        dim_mults=(1, 2, 4, 8),
    ):
        super().__init__()

        dims = [transition_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        time_dim = dim
        self.time_mlp = nn.Sequential(    # should be removed for Noise and Deterministic Baselines
            SinusoidalPosEmb(dim),
            nn.Linear(dim, dim * 4),
            nn.Mish(),
            nn.Linear(dim * 4, dim),
        )

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)
        
        self.label_embed = nn.Embedding(num_class, time_dim)

        # print(in_out)
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                ResidualTemporalBlock(dim_in, dim_out, embed_dim=time_dim),
                AttentionBlock(dim_out, use_checkpoint=False, num_heads=4),
                ResidualTemporalBlock(dim_out, dim_out, embed_dim=time_dim),
                Downsample1d(dim_out) if not is_last else nn.Identity()
            ]))
            
            '''self.downs.append(nn.ModuleList([
                ResidualTemporalBlock(dim_in, dim_out, embed_dim=time_dim*2),
                ResidualTemporalBlock(dim_out, dim_out, embed_dim=time_dim*2),
                Downsample1d(dim_out) if not is_last else nn.Identity()
            ]))'''

        mid_dim = dims[-1]
        self.mid_block1 = ResidualTemporalBlock(mid_dim, mid_dim, embed_dim=time_dim)
        self.attention = AttentionBlock(mid_dim, use_checkpoint=False, num_heads=16)
        self.mid_block2 = ResidualTemporalBlock(mid_dim, mid_dim, embed_dim=time_dim)
        '''self.mid_block1 = ResidualTemporalBlock(mid_dim, mid_dim, embed_dim=time_dim*2)
        self.mid_block2 = ResidualTemporalBlock(mid_dim, mid_dim, embed_dim=time_dim*2)'''

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(nn.ModuleList([
                ResidualTemporalBlock(dim_out * 2, dim_in, embed_dim=time_dim),
                AttentionBlock(dim_in, use_checkpoint=False, num_heads=4),
                ResidualTemporalBlock(dim_in, dim_in, embed_dim=time_dim),
                Upsample1d(dim_in) if not is_last else nn.Identity()
            ]))
            '''self.ups.append(nn.ModuleList([
                ResidualTemporalBlock(dim_out * 2, dim_in, embed_dim=time_dim*2),
                ResidualTemporalBlock(dim_in, dim_in, embed_dim=time_dim*2),
                Upsample1d(dim_in) if not is_last else nn.Identity()
            ]))'''

        self.final_conv = nn.Sequential(
            Conv1dBlock(dim, dim, kernel_size=3, if_zero=True),
            nn.Conv1d(dim, transition_dim, 1),
        )

    def forward(self, x, time, class_label):
        x = einops.rearrange(x, 'b h t -> b t h')

        # t = None    # for Noise and Deterministic Baselines
        t = self.time_mlp(time)   # for diffusion
        #print(x.shape, time.shape, t.shape, class_label.shape)
        #y_emb = self.label_embed(class_label)
        #print(t.shape, y_emb.shape)
        #t = t + y_emb
        #t = torch.cat((t, y_emb), 1)
        h = []

        for resnet, attn, resnet2, downsample in self.downs:
            x = resnet(x, t)
            x = attn(x)
            x = resnet2(x, t)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.attention(x)
        x = self.mid_block2(x, t)

        for resnet,attn, resnet2, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, t)
            x = attn(x)
            x = resnet2(x, t)
            x = upsample(x)

        x = self.final_conv(x)
        x = einops.rearrange(x, 'b t h -> b h t')
        return x
        
class TemporalUnetNoAttn(nn.Module):
    def __init__(
        self,
        transition_dim,
        num_class,
        dim=32,
        dim_mults=(1, 2, 4, 8),
    ):
        super().__init__()

        dims = [transition_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        time_dim = dim
        self.time_mlp = nn.Sequential(    # should be removed for Noise and Deterministic Baselines
            SinusoidalPosEmb(dim),
            nn.Linear(dim, dim * 4),
            nn.Mish(),
            nn.Linear(dim * 4, dim),
        )

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)
        
        self.label_embed = nn.Embedding(num_class, time_dim)

        # print(in_out)
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                ResidualTemporalBlock(dim_in, dim_out, embed_dim=time_dim),
                #AttentionBlock(dim_out, use_checkpoint=False, num_heads=4),
                ResidualTemporalBlock(dim_out, dim_out, embed_dim=time_dim),
                Downsample1d(dim_out) if not is_last else nn.Identity()
            ]))
            
            '''self.downs.append(nn.ModuleList([
                ResidualTemporalBlock(dim_in, dim_out, embed_dim=time_dim*2),
                ResidualTemporalBlock(dim_out, dim_out, embed_dim=time_dim*2),
                Downsample1d(dim_out) if not is_last else nn.Identity()
            ]))'''

        mid_dim = dims[-1]
        self.mid_block1 = ResidualTemporalBlock(mid_dim, mid_dim, embed_dim=time_dim)
        #self.attention = AttentionBlock(mid_dim, use_checkpoint=False, num_heads=16)
        self.mid_block2 = ResidualTemporalBlock(mid_dim, mid_dim, embed_dim=time_dim)
        '''self.mid_block1 = ResidualTemporalBlock(mid_dim, mid_dim, embed_dim=time_dim*2)
        self.mid_block2 = ResidualTemporalBlock(mid_dim, mid_dim, embed_dim=time_dim*2)'''

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(nn.ModuleList([
                ResidualTemporalBlock(dim_out * 2, dim_in, embed_dim=time_dim),
                #AttentionBlock(dim_in, use_checkpoint=False, num_heads=4),
                ResidualTemporalBlock(dim_in, dim_in, embed_dim=time_dim),
                Upsample1d(dim_in) if not is_last else nn.Identity()
            ]))
            '''self.ups.append(nn.ModuleList([
                ResidualTemporalBlock(dim_out * 2, dim_in, embed_dim=time_dim*2),
                ResidualTemporalBlock(dim_in, dim_in, embed_dim=time_dim*2),
                Upsample1d(dim_in) if not is_last else nn.Identity()
            ]))'''

        self.final_conv = nn.Sequential(
            Conv1dBlock(dim, dim, kernel_size=3, if_zero=True),
            nn.Conv1d(dim, transition_dim, 1),
        )

    def forward(self, x, time, class_label):
        x = einops.rearrange(x, 'b h t -> b t h')

        # t = None    # for Noise and Deterministic Baselines
        t = self.time_mlp(time)   # for diffusion
        #print(x.shape, time.shape, t.shape, class_label.shape)
        #y_emb = self.label_embed(class_label)
        #print(t.shape, y_emb.shape)
        #t = t + y_emb
        #t = torch.cat((t, y_emb), 1)
        h = []

        for resnet, resnet2, downsample in self.downs:
            x = resnet(x, t)
            x = resnet2(x, t)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_block2(x, t)

        for resnet, resnet2, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, t)
            x = resnet2(x, t)
            x = upsample(x)

        x = self.final_conv(x)
        x = einops.rearrange(x, 'b t h -> b h t')
        return x
