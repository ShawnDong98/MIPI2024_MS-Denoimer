from collections import defaultdict
import numbers

import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import rearrange
import math
import warnings
from torch import einsum

from . import BaseModel

from denoiser.config import instantiate

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight
    

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias
    

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        # x: (b, c, h, w)
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class PreNorm(nn.Module):
    def __init__(self, dim, fn, layernorm_type='WithBias'):
        super().__init__()
        self.fn = fn
        self.layernorm_type = layernorm_type
        if layernorm_type == 'BiasFree' or layernorm_type == 'WithBias':
            self.norm = LayerNorm(dim, layernorm_type)
        else:
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        if self.layernorm_type == 'BiasFree' or self.layernorm_type == 'WithBias':
            x = self.norm(x)
        else:
            h, w = x.shape[-2:]
            x = to_4d(self.norm(to_3d(x)), h, w)
        return self.fn(x, *args, **kwargs)

class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)
    


class LocalNonLocalMultiheadSelfAttention(nn.Module):
    def __init__(
            self,
            dim,
            num_heads,  # 1, 2, 4=
            window_size=(8, 8),
            only_local_branch=False
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        self.window_size = window_size
        self.only_local_branch = only_local_branch

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=False)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=False)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=False)

    def forward(self, x):
        """
        x: [b,h,w,c]
        return out: [b,h,w,c]
        """
        b, c, h_inp, w_inp = x.shape
        hb, wb = 16, 16
        pad_h = (hb - h_inp % hb) % hb
        pad_w = (wb - w_inp % wb) % wb
        x = F.pad(x, [0, pad_w, 0, pad_h], mode='reflect')

        b, c, h, w = x.shape
        w_size = self.window_size
        assert h % w_size[0] == 0 and w % w_size[1] == 0, 'fmap dimensions must be divisible by the window size'
        if self.only_local_branch:
            q, k, v = self.qkv_dwconv(self.qkv(x)).chunk(3, dim=1)

            q, k, v = map(lambda t: rearrange(t, 'b c (h b0) (w b1) -> (b h w) (b0 b1) c',
                                              b0=self.window_size[0], b1=self.window_size[1]), (q, k, v))
            
            q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads), (q, k, v))
            q *= self.scale
            sim = einsum('b h i d, b h j d -> b h i j', q, k)
            attn = sim.softmax(dim=-1)
            out = einsum('b h i j, b h j d -> b h i d', attn, v)
            out = rearrange(out, 'b h n d -> b n (h d)')
            out = rearrange(out, '(b h w) (b0 b1) c -> b c (h b0) (w b1)', h=h // w_size[0], w=w // w_size[1], b0=w_size[0])
            out = self.project_out(out)
        else:
            q, k, v = self.qkv_dwconv(self.qkv(x)).chunk(3, dim=1)
            q1, q2 = q[:,:c//2,:,:], q[:,c//2:,:,:]
            k1, k2 = k[:,:c//2,:,:], k[:,c//2:,:,:]
            v1, v2 = v[:,:c//2,:,:], v[:,c//2:,:,:]

            # local branch
            q1, k1, v1 = map(lambda t: rearrange(t, 'b c (h b0) (w b1) -> b (h w) (b0 b1) c', b0=self.window_size[0], b1=self.window_size[1]), (q1, k1, v1))
            q1, k1, v1 = map(lambda t: rearrange(t, 'b n mm (h d) -> b n h mm d', h=self.num_heads//2), (q1, k1, v1))
            q1 *= self.scale
            sim1 = einsum('b n h i d, b n h j d -> b n h i j', q1, k1)
            attn1 = sim1.softmax(dim=-1)
            out1 = einsum('b n h i j, b n h j d -> b n h i d', attn1, v1)
            out1 = rearrange(out1, 'b n h mm d -> b n mm (h d)')

            # non-local branch
            q2, k2, v2 = map(lambda t: rearrange(t, 'b c (h b0) (w b1) -> b (h w) (b0 b1) c',
                                                 b0=w_size[0], b1=w_size[1]), (q2, k2, v2))
            q2, k2, v2 = map(lambda t: t.permute(0, 2, 1, 3), (q2.clone(), k2.clone(), v2.clone()))
            q2, k2, v2 = map(lambda t: rearrange(t, 'b n mm (h d) -> b n h mm d', h=self.num_heads//2), (q2, k2, v2))
            q2 *= self.scale
            sim2 = einsum('b n h i d, b n h j d -> b n h i j', q2, k2)
            attn2 = sim2.softmax(dim=-1)
            out2 = einsum('b n h i j, b n h j d -> b n h i d', attn2, v2)
            out2 = rearrange(out2, 'b n h mm d -> b n mm (h d)')
            out2 = out2.permute(0, 2, 1, 3)

            out = torch.cat([out1,out2],dim=-1).contiguous()
            out = rearrange(out, 'b (h w) (b0 b1) c -> b c (h b0) (w b1)', h=h // w_size[0], w=w // w_size[1], b0=w_size[0])
            
            out = self.project_out(out)

        return out[:, :, :h_inp, :w_inp]
    

class ChannelMultiheadSelfAttention(nn.Module):
    def __init__(self, 
                 dim, 
                 num_heads, 
                 bias=False,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)


    def forward(self, x):
        b,c,h,w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads) # (b, c, h, w) -> (b, head, c_, h * w)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w) #(b, head, c_, h*w) -> (b, c, h, w)

        out = self.project_out(out) # (b, c, h, w)
        return out
    


## Gated-Dconv Feed-Forward Network (GDFN)
class Gated_Dconv_FeedForward(nn.Module):
    def __init__(self, 
                 dim, 
                 ffn_expansion_factor = 2.66
    ):
        super(Gated_Dconv_FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=False)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=True)

        self.act_fn = nn.GELU()

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=False)

    def forward(self, x):
        """
        x: [b, c, h, w]
        return out: [b, c, h, w]
        """
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = self.act_fn(x1) * x2
        x = self.project_out(x)
        return x
    

class SpatialAttentionBlock(nn.Module):
    def __init__(
            self,
            dim,
            window_size=(8, 8),
            num_heads=8,
            num_blocks=2,
            ffn_expansion_factor = 2.66,
            layernorm_type="BiasFree",
            bias=False,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleList([
                PreNorm(dim, 
                    LocalNonLocalMultiheadSelfAttention(
                        dim=dim, 
                        num_heads=num_heads, 
                        window_size=window_size, 
                        only_local_branch=(num_heads==1)
                    ),
                    layernorm_type = layernorm_type
                ),
                PreNorm(
                    dim, 
                    Gated_Dconv_FeedForward(
                        dim=dim,
                        ffn_expansion_factor=ffn_expansion_factor
                    ),
                    layernorm_type = layernorm_type
                )
            ]))

    def forward(self, x):
        for (spatial_attn, ff) in self.blocks:
            x = spatial_attn(x) + x
            x = ff(x) + x
        return x


class DownSample(nn.Module):
    def __init__(self, in_channels, bias=False):
        super(DownSample, self).__init__()
        self.down = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * 2, 4, 2, 1, bias=False)
        )

    def forward(self, x):
        x = self.down(x)
        return x

class UpSample(nn.Module):
    def __init__(self, in_channels, bias=False):
        super(UpSample, self).__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_channels, in_channels // 2, stride=2, kernel_size=2, padding=0, output_padding=0)
        )

    def forward(self, x):
        x = self.up(x)
        return x    

class LNLT(nn.Module):
    def __init__(self,
                 in_dim,
                 dim,
                 out_dim,
                 window_sizes,
                 layernorm_type,
                 ffn_expansion_factor,
                 num_level,
                 num_blocks,
                 ):
        super().__init__()
        self.num_level = num_level
        
        self.embedding = nn.Conv2d(in_dim, dim, kernel_size=3, stride=1, padding=1, bias=False)

        self.Encoder = nn.ModuleList([])
        for i in range(num_level-1):
            self.Encoder.append(
                SpatialAttentionBlock(
                    dim = dim * 2 ** i, 
                    window_size = window_sizes[i],
                    num_heads = 2 ** i, 
                    layernorm_type = layernorm_type,
                    ffn_expansion_factor = ffn_expansion_factor,
                    num_blocks = num_blocks[i],
                )
            )

        self.BottleNeck = SpatialAttentionBlock(
                dim = dim * 2 ** (num_level-1), 
                window_size = window_sizes[(num_level-1)], 
                num_heads = 2 ** (num_level-1), 
                layernorm_type = layernorm_type,
                ffn_expansion_factor = ffn_expansion_factor,
                num_blocks = num_blocks[(num_level-1)],
            )
        
        self.Decoder = nn.ModuleList([])
        for i in range(num_level-2, -1, -1): # 2, 1, 0
            self.Decoder.append(
                SpatialAttentionBlock(
                    dim = dim * 2 ** i, 
                    window_size = window_sizes[(num_level*2-2)-i],
                    num_heads = 2 ** i, 
                    layernorm_type = layernorm_type,
                    ffn_expansion_factor = ffn_expansion_factor,
                    num_blocks = num_blocks[(num_level*2-2)-i], 
                )
            )
    
        self.Downs = nn.ModuleList([])
        for i in range(num_level-1): # 0, 1, 2
            self.Downs.append(DownSample(dim * 2 ** i))


        self.Ups = nn.ModuleList([])
        for i in range(num_level-1, 0, -1): # 3, 2, 1
            self.Ups.append(UpSample(dim * 2 ** i))

        self.fusions = nn.ModuleList([])
        for i in range(num_level-1, 0, -1): # 3, 2, 1
            self.fusions.append(
                nn.Conv2d(
                in_channels = dim * 2 ** i,
                out_channels = dim * 2 ** (i-1),
                kernel_size = 1,
                stride = 1,
                padding = 0,
                bias = False
            ),
        )

        self.mapping = nn.Conv2d(dim, out_dim, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        out = defaultdict()
        res = self.embedding(x)

        enc_features = []
        for i in range(self.num_level-1):
            res = self.Encoder[i](res)
            enc_features.append(res)
            out[f'block{i+1}'] = res
            res = self.Downs[i](res)

        res = self.BottleNeck(res)
        out[f'block{self.num_level}'] = res

        for i in range(self.num_level-1):
            res = self.Ups[i](res) # dim * 2 ** 2 -> dim * 2 ** 1
            res = torch.cat([res, enc_features[self.num_level-2-i]], dim=1) # dim * 2 ** 2
            res = self.fusions[i](res) # dim * 2 ** 2 -> dim * 2 ** 1
            res = self.Decoder[i](res)
            out[f'block{self.num_level+1+i}'] = res

        res = self.mapping(res) + x

        out['pred'] = res

        return out



class MultiStageLNLT(BaseModel):
    def __init__(self,
                 stages,
                 num_level,
                 share_params,
                 in_dim,
                 dim,
                 out_dim,
                 window_sizes,
                 layernorm_type,
                 ffn_expansion_factor,
                 num_blocks,
                 pretrain=False,
                 losses=defaultdict(),
                 noise_maker = None
                 ):
        super().__init__(losses=losses)
        self.pretrain = pretrain
        self.stages = stages
        self.num_blocks = num_blocks
        self.share_params = share_params


        self.stage = nn.ModuleList([
            LNLT(
                in_dim = in_dim,
                dim = dim, 
                out_dim = out_dim,
                window_sizes = window_sizes,
                layernorm_type = layernorm_type,
                ffn_expansion_factor = ffn_expansion_factor,
                num_level = num_level,
                num_blocks = num_blocks,

            ) for _ in range(stages)
        ]) if not share_params else LNLT(
                in_dim = in_dim,
                dim = dim, 
                out_dim = out_dim,
                window_sizes = window_sizes,
                layernorm_type = layernorm_type,
                ffn_expansion_factor = ffn_expansion_factor,
                num_level = num_level,
                num_blocks = num_blocks,
        )

        if noise_maker: 
            self.noise_maker = instantiate(noise_maker)
        else:
            self.noise_maker = noise_maker

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Conv2d) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    def prepare_input(self, data):
        return data
    

    def forward_train(self, data):
        losses = defaultdict()
        out = defaultdict()

        if self.pretrain: 
            camera_id = torch.randint(0, len(self.noise_maker), (1,)).item()
            gt, pred, cur_metadata = self.noise_maker(data['gt'], data['scale'], data['ratio'], camera_id)
        else: 
            pred = data['lq']
        for i in range(self.stages):
            pred = self.stage[i](pred) if not self.share_params else self.stage(pred)
            out[f'stage{i}'] = pred.copy()
            pred = pred['pred']

        out['pred'] = pred
        for loss_name, loss_fn in self.losses.items():
            losses[loss_name] = loss_fn(out, data)
        
        return losses
        
    def forward_test(self, data):
        out = defaultdict()
        pred = data['lq']
        for i in range(self.stages):
            pred = self.stage[i](pred) if not self.share_params else self.stage(pred)
            out[f'stage{i}'] = pred.copy()
            pred = pred['pred']

        out['pred'] = pred

        return out
    
    def forward_test_tta(self, data):
        lq = data['lq']
        tta_out = torch.zeros_like(lq).to(lq.device)
        tta_list = ['vflip', 'hflip', 'rot90']
        for tta in tta_list:
            if tta == 'vflip':
                data['lq'] = torch.flip(lq, (2,))
                pred = self.forward_test(data)['pred']
                pred = torch.flip(pred, (2,))
                data['lq'] = torch.flip(data['lq'], (2,))
            if tta == 'hflip': 
                data['lq'] = torch.flip(lq, (3,))
                pred = self.forward_test(data)['pred']
                pred = torch.flip(pred, (3,))
                data['lq'] = torch.flip(data['lq'], (3,))
            if tta == 'rot90': 
                data['lq'] =  torch.permute(lq, (0, 1, 3, 2))
                pred = self.forward_test(data)['pred']
                pred = torch.permute(pred, (0, 1, 3, 2))
                data['lq'] =  torch.permute(data['lq'], (0, 1, 3, 2))
            tta_out += pred
        tta_out = tta_out / len(tta_list)
        return tta_out
        
        

class MultiStageLNLTProfiling(BaseModel):
    def __init__(self,
                 stages,
                 share_params,
                 in_dim,
                 dim,
                 out_dim,
                 window_size,
                 layernorm_type,
                 ffn_expansion_factor,
                 num_blocks,
                 pretrain=False
                 ):
        super().__init__()
        self.pretrain = pretrain
        self.stages = stages
        self.num_blocks = num_blocks
        self.share_params = share_params


        self.stage = nn.ModuleList([
            LNLT(
                in_dim = in_dim,
                dim = dim, 
                out_dim = out_dim,
                window_size = window_size,
                layernorm_type = layernorm_type,
                ffn_expansion_factor = ffn_expansion_factor,
                num_blocks = num_blocks,

            ) for _ in range(stages)
        ]) if not share_params else LNLT(
                in_dim = in_dim,
                dim = dim, 
                out_dim = out_dim,
                window_size = window_size,
                layernorm_type = layernorm_type,
                ffn_expansion_factor = ffn_expansion_factor,
                num_blocks = num_blocks,
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Conv2d) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    def prepare_input(self, data):
        return data
    
    def forward(self, x):
        if self.pretrain: out = x
        else: out = x
        for i in range(self.stages):
            out = self.stage[i](out) if not self.share_params else self.stage(out)
        return out
