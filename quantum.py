# from .transformer import PreNorm, PEG, ChanLayerNorm, FeedForward, exists
import torch.nn.functional as F
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf
import torch
from functools import partial
from torch import nn
from einops import rearrange, reduce
from einops.layers.torch import Reduce


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, 
                                   stride=stride, padding=padding, dilation=dilation, 
                                   groups=in_channels)  
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1) 
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class MultiScaleQuantumBlock(nn.Module):
    def __init__(self, in_channels=512, out_channels=512, n_scales=5, qm_channels=16, n_wires=4):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.qm_channels = qm_channels
        self.n_scales = n_scales
        self.n_wires = n_wires

        dilation_rates = [1, 3, 5, 7, 9]  
        
        self.conv_scales = nn.ModuleList([
            DepthwiseSeparableConv(in_channels, in_channels // 2, kernel_size=3, padding=rate, dilation=rate)
            for rate in dilation_rates
        ])

        self.quantum_layers = nn.ModuleList([
            QFCModel(in_channel=in_channels // 2, qm_channel=qm_channels)
            for _ in range(n_scales)
        ])
        
        self.increase = nn.Conv2d(in_channels // 2 * n_scales, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        bs, c, h, w = x.shape
        
        multi_scale_outputs = []
        
        for i, conv in enumerate(self.conv_scales):

            scale_features = conv(x)  
            
            quantum_output = self.quantum_layers[i](scale_features) 
            
            multi_scale_outputs.append(quantum_output)
        
        x = torch.cat(multi_scale_outputs, dim=1)  
        
        x = self.increase(x)  
        x = self.sigmoid(x)*x
        
        return x

class QFCModel(tq.QuantumModule):
    class QLayer(tq.QuantumModule):
        def __init__(self):
            super().__init__()
            self.n_wires = 4
            self.random_layer = tq.RandomLayer(n_ops=50, wires=list(range(self.n_wires)))

            # Trainable quantum gates
            self.rx0 = tq.RX(has_params=True, trainable=True)
            self.ry0 = tq.RY(has_params=True, trainable=True)
            self.rz0 = tq.RZ(has_params=True, trainable=True)
            self.crx0 = tq.CRX(has_params=True, trainable=True)

        @tq.static_support
        def forward(self, q_device: tq.QuantumDevice):
            self.q_device = q_device

            # Random initialized quantum gates
            self.random_layer(self.q_device)

            # Trainable quantum gates
            self.rx0(self.q_device, wires=0)
            self.ry0(self.q_device, wires=1)
            self.rz0(self.q_device, wires=3)
            self.crx0(self.q_device, wires=[0, 2])

            # Add non-parameterized gates
            tqf.hadamard(self.q_device, wires=3, static=self.static_mode, parent_graph=self.graph)
            tqf.sx(self.q_device, wires=2, static=self.static_mode, parent_graph=self.graph)
            tqf.cnot(self.q_device, wires=[3, 0], static=self.static_mode, parent_graph=self.graph)

    def __init__(self, in_channel=512, qm_channel=16):
        super().__init__()
        self.n_wires = 4
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires, device="cuda")
        self.encoder = tq.AmplitudeEncoder()

        self.q_layer = self.QLayer()
        self.measure = tq.MeasureAll(tq.PauliZ)
        
        self.decrease = nn.Conv2d(in_channel, qm_channel, kernel_size=1)
        self.increase = nn.Conv2d(qm_channel, in_channel, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Dimensionality reduction for quantum circuit processing
        x = self.decrease(x)
        b, c, h, w = x.shape
        x = x.permute(0, 2, 3, 1).reshape(b * h * w, c)
        
        # Encode into quantum device
        self.encoder(self.q_device, x)
        self.q_layer(self.q_device)
        
        # Measurement and reshape back to original form
        x = self.measure(self.q_device)
        x = x.repeat(1, 4).reshape(b, h, w, c).permute(0, 3, 1, 2)
        x = self.increase(x)
        x = self.sigmoid(x)
        return x




def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def cast_tuple(val, length = 1):
    return val if isinstance(val, tuple) else ((val,) * length)

# helper classes

class ChanLayerNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1)) # [1, 512, 1, 1]
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1)) # [1, 512, 1, 1]

    def forward(self, x):
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True) # [16, 1, 16, 16]
        mean = torch.mean(x, dim = 1, keepdim = True) # [16, 1, 16, 16]
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = ChanLayerNorm(dim)
        self.fn = fn

    def forward(self, x):
        return self.fn(self.norm(x))

class Downsample(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.conv = nn.Conv2d(dim_in, dim_out, 3, stride = 2, padding = 1)

    def forward(self, x):
        return self.conv(x)

class PEG(nn.Module):
    def __init__(self, dim, kernel_size = 3):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim, kernel_size = kernel_size, padding = kernel_size // 2, groups = dim, stride = 1)

    def forward(self, x):
        return self.proj(x) + x # Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=512)

# feedforward

class FeedForward(nn.Module):
    def __init__(self, dim, expansion_factor = 4, dropout = 0.):
        super().__init__()
        inner_dim = dim * expansion_factor # 512 * 4 = 2048
        self.net = nn.Sequential(
            nn.Conv2d(dim, inner_dim, 1), # 512, 2048
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(inner_dim, dim, 1), # 2048 512
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

# attention

class ScalableSelfAttention(nn.Module):
    def __init__(
        self,
        dim,
        heads = 8,
        dim_key = 32,
        dim_value = 32,
        dropout = 0.,
        reduction_factor = 1
    ):
        super().__init__()
        self.heads = heads # 8
        self.scale = dim_key ** -0.5 # 0.1767766952966369     32*8 = 256
        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_q = nn.Conv2d(dim, dim_key * heads, 1, bias = False) # 512 256 1
        self.to_k = nn.Conv2d(dim, dim_key * heads, reduction_factor, stride = reduction_factor, bias = False)
        self.to_v = nn.Conv2d(dim, dim_value * heads, reduction_factor, stride = reduction_factor, bias = False)

        self.to_out = nn.Sequential(
            nn.Conv2d(dim_value * heads, dim, 1), # 256 512
            nn.Dropout(dropout)
        )

    def forward(self, x): # [16, 512, 16, 16]
        height, width, heads = *x.shape[-2:], self.heads # 16 16 8

        q, k, v = self.to_q(x), self.to_k(x), self.to_v(x) # [16, 256, 16, 16]

        # split out heads
        # [16, 8, 256, 32]
        q, k, v = map(lambda t: rearrange(t, 'b (h d) ... -> b h (...) d', h = heads), (q, k, v))

        # similarity
        # [16, 8, 256, 256]
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        # attention

        attn = self.attend(dots)
        attn = self.dropout(attn)

        # aggregate values
        # [16, 8, 256, 256] [16, 8, 256, 32]
        out = torch.matmul(attn, v) # [16, 8, 256, 32]

        # merge back heads
        # [16, 256, 16, 16]
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x = height, y = width)
        return self.to_out(out) # [16, 512, 16, 16]

class InteractiveWindowedSelfAttention(nn.Module):
    def __init__(
        self,
        dim,
        window_size,
        heads = 8,
        dim_key = 32,
        dim_value = 32,
        dropout = 0.
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_key ** -0.5
        self.window_size = window_size
        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.local_interactive_module = nn.Conv2d(dim_value * heads, dim_value * heads, 3, padding = 1)

        self.to_q = nn.Conv2d(dim, dim_key * heads, 1, bias = False)
        self.to_k = nn.Conv2d(dim, dim_key * heads, 1, bias = False)
        self.to_v = nn.Conv2d(dim, dim_value * heads, 1, bias = False)

        self.to_out = nn.Sequential(
            nn.Conv2d(dim_value * heads, dim, 1),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        height, width, heads, wsz = *x.shape[-2:], self.heads, self.window_size

        wsz_h, wsz_w = default(wsz, height), default(wsz, width)
        assert (height % wsz_h) == 0 and (width % wsz_w) == 0, f'height ({height}) or width ({width}) of feature map is not divisible by the window size ({wsz_h}, {wsz_w})'

        q, k, v = self.to_q(x), self.to_k(x), self.to_v(x)

        # get output of LIM

        local_out = self.local_interactive_module(v)

        # divide into window (and split out heads) for efficient self attention

        q, k, v = map(lambda t: rearrange(t, 'b (h d) (x w1) (y w2) -> (b x y) h (w1 w2) d', h = heads, w1 = wsz_h, w2 = wsz_w), (q, k, v))

        # similarity

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        # attention

        attn = self.attend(dots)
        attn = self.dropout(attn)

        # aggregate values

        out = torch.matmul(attn, v)

        # reshape the windows back to full feature map (and merge heads)

        out = rearrange(out, '(b x y) h (w1 w2) d -> b (h d) (x w1) (y w2)', x = height // wsz_h, y = width // wsz_w, w1 = wsz_h, w2 = wsz_w)

        # add LIM output 

        out = out + local_out

        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        heads = 8,
        ff_expansion_factor = 4,
        dropout = 0.,
        ssa_dim_key = 32,
        ssa_dim_value = 32,
        ssa_reduction_factor = 1,
        norm_output = True
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        for ind in range(depth):
            is_first = ind == 0

            self.layers.append(nn.ModuleList([
                PreNorm(dim, ScalableSelfAttention(dim, heads = heads, dim_key = ssa_dim_key, dim_value = ssa_dim_value, reduction_factor = ssa_reduction_factor, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, expansion_factor = ff_expansion_factor, dropout = dropout)),
                PEG(dim) if is_first else None,
            ]))

        self.norm = ChanLayerNorm(dim) if norm_output else nn.Identity()

    def forward(self, x): # [16, 512, 16, 16]
        for ssa, ff1, peg in self.layers: # len = 2
            x = ssa(x) + x # [16, 512, 16, 16]
            x = ff1(x) + x # ff1(x)
            if exists(peg):
                x = peg(x) # [16, 512, 16, 16]

        return self.norm(x) # [16, 512, 16, 16]

class ScalableViT(nn.Module):
    def __init__(
        self,
        *,
        num_classes,
        dim,
        depth,
        heads,
        reduction_factor,
        window_size = None,
        iwsa_dim_key = 32,
        iwsa_dim_value = 32,
        ssa_dim_key = 32,
        ssa_dim_value = 32,
        ff_expansion_factor = 4,
        channels = 3,
        dropout = 0.
    ):
        super().__init__()
        self.to_patches = nn.Conv2d(channels, dim, 7, stride = 4, padding = 3)

        assert isinstance(depth, tuple), 'depth needs to be tuple if integers indicating number of transformer blocks at that stage'

        num_stages = len(depth)
        dims = tuple(map(lambda i: (2 ** i) * dim, range(num_stages)))

        hyperparams_per_stage = [
            heads,
            ssa_dim_key,
            ssa_dim_value,
            reduction_factor,
            iwsa_dim_key,
            iwsa_dim_value,
            window_size,
        ]

        hyperparams_per_stage = list(map(partial(cast_tuple, length = num_stages), hyperparams_per_stage))
        assert all(tuple(map(lambda arr: len(arr) == num_stages, hyperparams_per_stage)))

        self.layers = nn.ModuleList([])

        for ind, (layer_dim, layer_depth, layer_heads, layer_ssa_dim_key, layer_ssa_dim_value, layer_ssa_reduction_factor, layer_iwsa_dim_key, layer_iwsa_dim_value, layer_window_size) in enumerate(zip(dims, depth, *hyperparams_per_stage)):
            is_last = ind == (num_stages - 1)

            self.layers.append(nn.ModuleList([
                Transformer(dim = layer_dim, depth = layer_depth, heads = layer_heads, ff_expansion_factor = ff_expansion_factor, dropout = dropout, ssa_dim_key = layer_ssa_dim_key, ssa_dim_value = layer_ssa_dim_value, ssa_reduction_factor = layer_ssa_reduction_factor, iwsa_dim_key = layer_iwsa_dim_key, iwsa_dim_value = layer_iwsa_dim_value, iwsa_window_size = layer_window_size, norm_output = not is_last),
                Downsample(layer_dim, layer_dim * 2) if not is_last else None
            ]))

        self.mlp_head = nn.Sequential(
            Reduce('b d h w -> b d', 'mean'),
            nn.LayerNorm(dims[-1]),
            nn.Linear(dims[-1], num_classes)
        )

    def forward(self, img):
        x = self.to_patches(img)

        for transformer, downsample in self.layers:
            x = transformer(x)

            if exists(downsample):
                x = downsample(x)

        return self.mlp_head(x)