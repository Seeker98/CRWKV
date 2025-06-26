import math
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.cpp_extension import load
import torch.utils.checkpoint as cp
from einops import rearrange

# Use this line for experiments
T_MAX = 512 * 512

# Only use this for performance testing
# T_MAX = 1024 * 1024


wkv_cuda = load(
    name="wkv",
    sources=[
        "basicsr/models/archs/cuda/wkv_op.cpp",
        "basicsr/models/archs/cuda/wkv_cuda.cu",
    ],
    verbose=True,
    extra_cuda_cflags=[
        "-res-usage",
        "--maxrregcount 60",
        "--use_fast_math",
        "-O3",
        "-Xptxas -O3",
        f"-DTmax={T_MAX}",
    ],
)


class WKV(torch.autograd.Function):
    @staticmethod
    def forward(ctx, B, T, C, w, u, k, v):
        ctx.B = B
        ctx.T = T
        ctx.C = C
        assert T <= T_MAX
        assert B * C % min(C, 1024) == 0

        half_mode = (w.dtype == torch.half)
        bf_mode = (w.dtype == torch.bfloat16)
        ctx.save_for_backward(w, u, k, v)
        w = w.float().contiguous()
        u = u.float().contiguous()
        k = k.float().contiguous()
        v = v.float().contiguous()
        y = torch.empty((B, T, C), device='cuda', memory_format=torch.contiguous_format)
        wkv_cuda.forward(B, T, C, w, u, k, v, y)
        if half_mode:
            y = y.half()
        elif bf_mode:
            y = y.bfloat16()
        return y

    @staticmethod
    def backward(ctx, gy):
        B = ctx.B
        T = ctx.T
        C = ctx.C
        assert T <= T_MAX
        assert B * C % min(C, 1024) == 0
        w, u, k, v = ctx.saved_tensors
        gw = torch.zeros((B, C), device='cuda').contiguous()
        gu = torch.zeros((B, C), device='cuda').contiguous()
        gk = torch.zeros((B, T, C), device='cuda').contiguous()
        gv = torch.zeros((B, T, C), device='cuda').contiguous()
        half_mode = (w.dtype == torch.half)
        bf_mode = (w.dtype == torch.bfloat16)
        wkv_cuda.backward(B, T, C,
                          w.float().contiguous(),
                          u.float().contiguous(),
                          k.float().contiguous(),
                          v.float().contiguous(),
                          gy.float().contiguous(),
                          gw, gu, gk, gv)
        if half_mode:
            gw = torch.sum(gw.half(), dim=0)
            gu = torch.sum(gu.half(), dim=0)
            return (None, None, None, gw.half(), gu.half(), gk.half(), gv.half())
        elif bf_mode:
            gw = torch.sum(gw.bfloat16(), dim=0)
            gu = torch.sum(gu.bfloat16(), dim=0)
            return (None, None, None, gw.bfloat16(), gu.bfloat16(), gk.bfloat16(), gv.bfloat16())
        else:
            gw = torch.sum(gw, dim=0)
            gu = torch.sum(gu, dim=0)
            return (None, None, None, gw, gu, gk, gv)


def RUN_CUDA(B, T, C, w, u, k, v):
    return WKV.apply(B, T, C, w.cuda(), u.cuda(), k.cuda(), v.cuda())


def context_shift(input, shift_pixel=1, patch_resolution=None):   
    B, N, C = input.shape
    input = input.transpose(1, 2).reshape(B, C, patch_resolution[0], patch_resolution[1])
    B, _, H, W = input.shape
    output = torch.zeros_like(input)

    # horizontal/vertical +1/-1
    output[:, 0:int(C//8), :, shift_pixel:W] = input[:, 0:int(C//8), :, 0:W-shift_pixel]
    output[:, int(C//8):int(C//4), :, 0:W-shift_pixel] = input[:, int(C//8):int(C//4), :, shift_pixel:W]
    output[:, int(C//4):int(C//8*3), :, shift_pixel:H] = input[:, int(C//4):int(C//8*3), :, 0:H-shift_pixel]
    output[:, int(C//8*3):int(C//2), :, 0:H-shift_pixel] = input[:, int(C//8*3):int(C//2), :, shift_pixel:H]

    # horizontal/vertical +2/-2
    output[:, int(C//2):int(C//2+C//16), :, shift_pixel*2:W] = input[:, int(C//2):int(C//2+C//16), :, 0:W-shift_pixel*2]
    output[:, int(C//2+C//16):int(C//2+2*C//16), :, 0:W-shift_pixel*2] = input[:, int(C//2+C//16):int(C//2+2*C//16), :, shift_pixel*2:W]
    output[:, int(C//2+2*C//16):int(C//2+3*C//16), :, shift_pixel*2:H] = input[:, int(C//2+2*C//16):int(C//2+3*C//16), :, 0:H-shift_pixel*2]
    output[:, int(C//2+3*C//16):int(C//2+C//4), :, 0:H-shift_pixel*2] = input[:, int(C//2+3*C//16):int(C//2+C//4), :, shift_pixel*2:H]

    # diag/anti-diag +1/-1
    output[:, int(C//2+C//4):int(C//2+C//4+C//16), shift_pixel:W, shift_pixel:W] = input[:, int(C//2+C//4):int(C//2+C//4+C//16), 0:W-shift_pixel, 0:W-shift_pixel]
    output[:, int(C//2+C//4+C//16):int(C//2+C//4+2*C//16), 0:W-shift_pixel, 0:W-shift_pixel] = input[:, int(C//2+C//4+C//16):int(C//2+C//4+2*C//16), shift_pixel:W, shift_pixel:W]
    output[:, int(C//2+C//4+2*C//16):int(C//2+C//4+3*C//16), shift_pixel:H, shift_pixel:W] = input[:, int(C//2+C//4+2*C//16):int(C//2+C//4+3*C//16), 0:H-shift_pixel, 0:W-shift_pixel]
    output[:, int(C//2+C//4+3*C//16):int(C//2+C//2), 0:H-shift_pixel, 0:W-shift_pixel] = input[:, int(C//2+C//4+3*C//16):int(C//2+C//2), shift_pixel:H, shift_pixel:W]

    return output.reshape(B, C, N).transpose(1, 2)


class FrequencyMix(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.ln = nn.LayerNorm(n_embd)
        self.weighted = nn.Conv2d(n_embd*2, n_embd*2, kernel_size=1, bias=False)
        self.lrelu = nn.LeakyReLU(0.1)

    def forward(self, x, resolution):
        B, T, C = x.size()

        x = rearrange(x, "b (h w) c -> b c h w", h=resolution[0], w=resolution[1])
        x_freq = torch.fft.rfft2(
            x, dim=[2, 3], s=(resolution[0], resolution[1]), norm="ortho"
        )
        # x_freq: B, C, H, W, 2
        x_freq_ri = torch.stack([x_freq.real, x_freq.imag], dim=-1) 
        x_freq_ri=rearrange(x_freq_ri, "b c h w r -> b (r c) h w")
        x_freq_learn = self.weighted(x_freq_ri)
        x_freq_learn = self.lrelu(x_freq_learn)
        x_freq_learn = rearrange(x_freq_learn, "b (r c) h w -> b c h w r", r=2)

        x_freq=torch.view_as_complex(x_freq_learn.contiguous())
        x_spatial = torch.fft.irfft2(x_freq, dim=[2, 3], s=(resolution[0], resolution[1]), norm="ortho")
        # gate
        x = F.gelu(x) * x_spatial
        x = rearrange(x, "b c h w -> b (h w) c")
        x = self.ln(x)
        return x


class VRWKV_SpatialMix(nn.Module):
    def __init__(self, n_embd, n_layer, layer_id, shift_mode='context_shift',
                 shift_pixel=1, init_mode='fancy', 
                 key_norm=False, with_cp=False):
        super().__init__()
        self.layer_id = layer_id
        self.n_layer = n_layer
        self.n_embd = n_embd
        self.device = None
        attn_sz = n_embd
        self._init_weights(init_mode)
        self.shift_pixel = shift_pixel
        self.shift_mode = shift_mode
        if shift_pixel > 0:
            self.shift_func = eval(shift_mode)
        else:
            self.spatial_mix_k = None
            self.spatial_mix_v = None
            self.spatial_mix_r = None

        self.key = nn.Linear(n_embd, attn_sz, bias=False)
        self.value = nn.Linear(n_embd, attn_sz, bias=False)
        self.receptance = nn.Linear(n_embd, attn_sz, bias=False)
        if key_norm:
            self.key_norm = nn.LayerNorm(n_embd)
        else:
            self.key_norm = None
        self.output = nn.Linear(attn_sz, n_embd, bias=False)

        self.key.scale_init = 0
        self.receptance.scale_init = 0
        self.output.scale_init = 0

        self.with_cp = with_cp

    def _init_weights(self, init_mode):
        if init_mode=='fancy':
            with torch.no_grad(): # fancy init
                ratio_0_to_1 = (self.layer_id / (self.n_layer - 1)) # 0 to 1
                ratio_1_to_almost0 = (1.0 - (self.layer_id / self.n_layer)) # 1 to ~0
                
                # fancy time_decay
                decay_speed = torch.ones(self.n_embd)
                for h in range(self.n_embd):
                    decay_speed[h] = -5 + 8 * (h / (self.n_embd-1)) ** (0.7 + 1.3 * ratio_0_to_1)
                self.spatial_decay = nn.Parameter(decay_speed)

                # fancy time_first
                zigzag = (torch.tensor([(i+1)%3 - 1 for i in range(self.n_embd)]) * 0.5)
                self.spatial_first = nn.Parameter(torch.ones(self.n_embd) * math.log(0.3) + zigzag)
                
                # fancy time_mix
                x = torch.ones(1, 1, self.n_embd)
                for i in range(self.n_embd):
                    x[0, 0, i] = i / self.n_embd
                self.spatial_mix_k = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
                self.spatial_mix_v = nn.Parameter(torch.pow(x, ratio_1_to_almost0) + 0.3 * ratio_0_to_1)
                self.spatial_mix_r = nn.Parameter(torch.pow(x, 0.5 * ratio_1_to_almost0))
        elif init_mode=='local':
            self.spatial_decay = nn.Parameter(torch.ones(self.n_embd))
            self.spatial_first = nn.Parameter(torch.ones(self.n_embd))
            self.spatial_mix_k = nn.Parameter(torch.ones([1, 1, self.n_embd]))
            self.spatial_mix_v = nn.Parameter(torch.ones([1, 1, self.n_embd]))
            self.spatial_mix_r = nn.Parameter(torch.ones([1, 1, self.n_embd]))
        elif init_mode=='global':
            self.spatial_decay = nn.Parameter(torch.zeros(self.n_embd))
            self.spatial_first = nn.Parameter(torch.zeros(self.n_embd))
            self.spatial_mix_k = nn.Parameter(torch.ones([1, 1, self.n_embd]) * 0.5)
            self.spatial_mix_v = nn.Parameter(torch.ones([1, 1, self.n_embd]) * 0.5)
            self.spatial_mix_r = nn.Parameter(torch.ones([1, 1, self.n_embd]) * 0.5)
        else:
            raise NotImplementedError

    def jit_func(self, x, patch_resolution):
        # Mix x with the previous timestep to produce xk, xv, xr
        B, T, C = x.size()
        if self.shift_pixel > 0:
            xx = self.shift_func(x, self.shift_pixel, patch_resolution)
            xk = x * self.spatial_mix_k + xx * (1 - self.spatial_mix_k)
            xv = x * self.spatial_mix_v + xx * (1 - self.spatial_mix_v)
            xr = x * self.spatial_mix_r + xx * (1 - self.spatial_mix_r)
        else:
            xk = x
            xv = x
            xr = x

        # Use xk, xv, xr to produce k, v, r
        k = self.key(xk)
        v = self.value(xv)
        r = self.receptance(xr)
        sr = torch.sigmoid(r)

        return sr, k, v

    def forward(self, x, patch_resolution=None):
        def _inner_forward(x):
            B, T, C = x.size()
            self.device = x.device

            sr, k, v = self.jit_func(x, patch_resolution)
            x = RUN_CUDA(B, T, C, self.spatial_decay / T, self.spatial_first / T, k, v)
            if self.key_norm is not None:
                x = self.key_norm(x)
            x = sr * x
            x = self.output(x)
            return x
        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)
        return x


class VRWKV_ChannelMix(nn.Module):
    def __init__(self, n_embd, n_layer, layer_id, shift_mode='context_shift',
                 shift_pixel=1, hidden_rate=4, init_mode='fancy',
                 key_norm=False, with_cp=False):
        super().__init__()
        self.layer_id = layer_id
        self.n_layer = n_layer
        self.n_embd = n_embd
        self.with_cp = with_cp
        self._init_weights(init_mode)
        self.shift_pixel = shift_pixel
        self.shift_mode = shift_mode
        if shift_pixel > 0:
            self.shift_func = eval(shift_mode)
        else:
            self.spatial_mix_k = None
            self.spatial_mix_r = None

        hidden_sz = hidden_rate * n_embd
        self.key = nn.Linear(n_embd, hidden_sz, bias=False)
        if key_norm:
            self.key_norm = nn.LayerNorm(hidden_sz)
        else:
            self.key_norm = None
        self.receptance = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(hidden_sz, n_embd, bias=False)

        self.value.scale_init = 0
        self.receptance.scale_init = 0

    def _init_weights(self, init_mode):
        if init_mode == 'fancy':
            with torch.no_grad(): # fancy init of time_mix
                ratio_1_to_almost0 = (1.0 - (self.layer_id / self.n_layer)) # 1 to ~0
                x = torch.ones(1, 1, self.n_embd)
                for i in range(self.n_embd):
                    x[0, 0, i] = i / self.n_embd
                self.spatial_mix_k = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
                self.spatial_mix_r = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
        elif init_mode == 'local':
            self.spatial_mix_k = nn.Parameter(torch.ones([1, 1, self.n_embd]))
            self.spatial_mix_r = nn.Parameter(torch.ones([1, 1, self.n_embd]))
        elif init_mode == 'global':
            self.spatial_mix_k = nn.Parameter(torch.ones([1, 1, self.n_embd]) * 0.5)
            self.spatial_mix_r = nn.Parameter(torch.ones([1, 1, self.n_embd]) * 0.5)
        else:
            raise NotImplementedError

    def forward(self, x, patch_resolution=None):
        def _inner_forward(x):
            if self.shift_pixel > 0:
                xx = self.shift_func(x, self.shift_pixel, patch_resolution)
                xk = x * self.spatial_mix_k + xx * (1 - self.spatial_mix_k)
                xr = x * self.spatial_mix_r + xx * (1 - self.spatial_mix_r)
            else:
                xk = x
                xr = x

            k = self.key(xk)
            k = torch.square(torch.relu(k))
            if self.key_norm is not None:
                k = self.key_norm(k)
            kv = self.value(k)
            x = torch.sigmoid(self.receptance(xr)) * kv
            return x
        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)
        return x


class CRBv1(nn.Module):
    def __init__(
        self,
        n_embd,
        n_layer,
        layer_id,
        shift_mode="context_shift",
        shift_pixel=1,
        hidden_rate=4,
        init_mode="fancy",
        key_norm=False,
    ):
        super().__init__()
        self.layer_id = layer_id

        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

        self.att = VRWKV_SpatialMix(
            n_embd, n_layer, layer_id, 
            shift_mode=shift_mode, shift_pixel=shift_pixel, 
            init_mode=init_mode, key_norm=key_norm
        )

        self.ffn = VRWKV_ChannelMix(
            n_embd, n_layer, layer_id, 
            shift_mode=shift_mode, shift_pixel=shift_pixel, hidden_rate=hidden_rate,
            init_mode=init_mode, key_norm=key_norm
        )

        self.gamma1 = nn.Parameter(torch.ones((n_embd)), requires_grad=True)
        self.gamma2 = nn.Parameter(torch.ones((n_embd)), requires_grad=True)

    def forward(self, x):
        b, c, h, w = x.shape

        resolution = (h, w)

        x = rearrange(x, "b c h w -> b (h w) c")
        x = x + self.gamma1 * self.att(self.ln1(x), resolution)

        x = x + self.gamma2 * self.ffn(self.ln2(x), resolution)
        x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w)

        return x


class CRBv2(nn.Module):
    def __init__(
        self,
        n_embd,
        n_layer,
        layer_id,
        shift_mode="context_shift",
        shift_pixel=1,
        hidden_rate=4,
        init_mode="fancy",
        key_norm=False,
    ):
        super().__init__()
        self.layer_id = layer_id

        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

        self.att = FrequencyMix(n_embd)

        self.ffn = VRWKV_ChannelMix(
            n_embd, n_layer, layer_id, 
            shift_mode=shift_mode, shift_pixel=shift_pixel, hidden_rate=hidden_rate,
            init_mode=init_mode, key_norm=key_norm
        )

        self.gamma1 = nn.Parameter(torch.ones((n_embd)), requires_grad=True)
        self.gamma2 = nn.Parameter(torch.ones((n_embd)), requires_grad=True)

    def forward(self, x):
        _, _, h, w = x.shape

        resolution = (h, w)

        x = rearrange(x, "b c h w -> b (h w) c")
        x = x + self.gamma1 * self.att(self.ln1(x), resolution)

        x = x + self.gamma2 * self.ffn(self.ln2(x), resolution)
        x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w)

        return x


##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(
                n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False
            ),
            nn.PixelUnshuffle(2),
        )

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(
                n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False
            ),
            nn.PixelShuffle(2),
        )

    def forward(self, x):
        return self.body(x)


class CRWKV(nn.Module):
    def __init__(
        self,
        inp_channels=3,
        out_channels=3,
        dim=48,
        num_blocks=[3,4,4,6],
        num_refinement_blocks=4,
    ):

        super(CRWKV, self).__init__()

        self.patch_embed = nn.Conv2d(
            inp_channels, dim, kernel_size=3, stride=1, padding=1, bias=True
        )

        self.encoder_level1 = nn.Sequential(
            *[
                CRBv1(n_embd=dim, n_layer=num_blocks[0], layer_id=i)
                for i in range(num_blocks[0])
            ]
        )

        self.down1_2 = Downsample(dim)  ## From Level 1 to Level 2
        self.encoder_level2 = nn.Sequential(
            *[
                CRBv1(n_embd=int(dim * 2**1), n_layer=num_blocks[1], layer_id=i)
                for i in range(num_blocks[1])
            ]
        )

        self.down2_3 = Downsample(int(dim * 2**1))  ## From Level 2 to Level 3
        self.encoder_level3 = nn.Sequential(
            *[
                CRBv1(n_embd=int(dim * 2**2), n_layer=num_blocks[2], layer_id=i)
                for i in range(num_blocks[2])
            ]
        )

        self.down3_4 = Downsample(int(dim * 2**2))  ## From Level 3 to Level 4
        self.latent = nn.Sequential(
            *[
                CRBv2(n_embd=int(dim * 2**3), n_layer=num_blocks[3], layer_id=i)
                for i in range(num_blocks[3])
            ]
        )

        self.up4_3 = Upsample(int(dim * 2**3))  ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(
            int(dim * 2**3), int(dim * 2**2), kernel_size=1, bias=True
        )
        self.decoder_level3 = nn.Sequential(
            *[
                CRBv1(n_embd=int(dim * 2**2), n_layer=num_blocks[2], layer_id=i)
                for i in range(num_blocks[2])
            ]
        )

        self.up3_2 = Upsample(int(dim * 2**2))  ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(
            int(dim * 2**2), int(dim * 2**1), kernel_size=1, bias=True
        )
        self.decoder_level2 = nn.Sequential(
            *[
                CRBv1(n_embd=int(dim * 2**1), n_layer=num_blocks[1], layer_id=i)
                for i in range(num_blocks[1])
            ]
        )

        self.up2_1 = Upsample(
            int(dim * 2**1)
        )  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)

        self.decoder_level1 = nn.Sequential(
            *[
                CRBv1(n_embd=int(dim * 2**1), n_layer=num_blocks[0], layer_id=i)
                for i in range(num_blocks[0])
            ]
        )

        self.refinement = nn.Sequential(
            *[
                CRBv1(n_embd=int(dim * 2**1), n_layer=num_refinement_blocks, layer_id=i)
                for i in range(num_refinement_blocks)
            ]
        )

        ###########################

        self.output = nn.Conv2d(
            int(dim * 2**1), out_channels, kernel_size=3, stride=1, padding=1, bias=True
        )

    def forward(self, inp_img):

        inp_enc_level1 = self.patch_embed(inp_img) 
        out_enc_level1 = self.encoder_level1(inp_enc_level1)

        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3)

        inp_enc_level4 = self.down3_4(out_enc_level3)
        latent = self.latent(inp_enc_level4)

        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)

        out_dec_level1 = self.refinement(out_dec_level1)

        out_dec_level1 = self.output(out_dec_level1) + inp_img

        return out_dec_level1


if __name__ == "__main__":
    model = CRWKV(
        inp_channels=3,
        out_channels=3,
        dim=48,
        num_blocks=[3, 4, 4, 6],
        num_refinement_blocks=4,
    )
    model.cuda()
    model.eval()

    size_list = [128, 256, 384, 512, 768, 1024]
    peak_mem_list = []
    time_list = []
    for sz in size_list:
        x = torch.zeros((1, 3, sz, sz)).float().cuda()
        # warm up
        with torch.no_grad():
            for _ in range(5):
                model(x)
        torch.cuda.reset_peak_memory_stats()
        # test memory
        torch.cuda.synchronize()
        with torch.no_grad():
            x = model(x)
        torch.cuda.synchronize()
        peak_mem = torch.cuda.max_memory_allocated()
        peak_mem_list.append(peak_mem)
        print(f"Peak memory used: {peak_mem / 1024 ** 2:.2f} MB")

        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
            enable_timing=True
        )
        # test time
        rep = 10
        timings = np.zeros((rep, 1))
        with torch.no_grad():
            for i in range(rep):
                starter.record()
                model(x)
                ender.record()
                torch.cuda.synchronize()
                curr_time = starter.elapsed_time(ender)
                timings[i] = curr_time
        avg_time = np.sum(timings) / rep
        time_list.append(avg_time)
    print(peak_mem_list)
    print(time_list)
