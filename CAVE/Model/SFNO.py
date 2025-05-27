
import torch
import torch.nn as nn
from timm.models.layers import DropPath, to_2tuple
from einops import rearrange
import torch.nn.functional as F
from Utils import make_coord
from VectorAttention import VectorAttention

def partition(x, ssl):
    B, H, W, C = x.shape
    x = x.view(B, H // ssl, ssl, W // ssl, ssl, C)
    sampling_area = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, ssl, ssl, C)
    return sampling_area

def reverse(sampling_area, ssl, H, W):
    B = int(sampling_area.shape[0] / (H * W / ssl / ssl))
    x = sampling_area.view(B, H // ssl, W // ssl, ssl, ssl, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def flops(self, N):
        flops = 0
        flops += N * self.dim * 3 * self.dim
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        flops += N * self.dim * self.dim
        return flops

class ChannelAttention(nn.Module):
    def __init__(self, in_planes):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // 4, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 4, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpePyBlock(nn.Module):
    def __init__(self,C, inchannels, bias=True):
        super(SpePyBlock, self).__init__()
                
        if C % 8 != 0:
            pad = 8 - (C % 8)
            C = C + pad

        self.conv2 = nn.Sequential(
            nn.Conv2d(C, inchannels * 2, kernel_size=3, stride=1, padding=1, groups=2, bias=bias),
            nn.LeakyReLU(0.2),
            nn.Conv2d(inchannels * 2, inchannels * 2, kernel_size=3, stride=1, padding=1, groups=2, bias=bias),
            nn.LeakyReLU(0.2)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(C, inchannels * 2, kernel_size=3, stride=1, padding=1, groups=4, bias=bias),
            nn.LeakyReLU(0.2),
            nn.Conv2d(inchannels * 2, inchannels * 2, kernel_size=3, stride=1, padding=1, groups=4, bias=bias),
            nn.LeakyReLU(0.2)
        )

        self.conv8 = nn.Sequential(
            nn.Conv2d(C, inchannels * 2, kernel_size=3, stride=1, padding=1, groups=8, bias=bias),
            nn.LeakyReLU(0.2),
            nn.Conv2d(inchannels * 2, inchannels * 2, kernel_size=3, stride=1, padding=1, groups=8, bias=bias),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x2, x4, x8):
        _, c, _, _ = x2.shape
        if c % 8 != 0:
            x2 = torch.cat((x2, x2[:, c - (8 - c % 8) - 1:c - 1, :, :]), dim=1)
            x4 = torch.cat((x4, x4[:, c - (8 - c % 8) - 1:c - 1, :, :]), dim=1)
            x8 = torch.cat((x8, x8[:, c - (8 - c % 8) - 1:c - 1, :, :]), dim=1)
        x2_1 = self.conv2(x2)
        x4_1 = self.conv4(x4)
        x8_1 = self.conv8(x8)

        return x2_1, x4_1, x8_1


class SpaPyBlock(nn.Module):
    def __init__(self, inchannels, outchannels, bias=True):
        super(SpaPyBlock, self).__init__()
        self.scale1 = nn.Sequential(
            nn.Conv2d(inchannels, inchannels, kernel_size=3, stride=1, padding=1, groups=1, bias=bias)
        )
        self.scale1_2 = nn.Sequential(
            nn.Upsample(mode='bilinear', scale_factor=1 / 2),
            nn.Conv2d(inchannels, inchannels, kernel_size=3, stride=1, padding=1, groups=1, bias=bias)
        )
        self.scale1_4 = nn.Sequential(
            nn.Upsample(mode='bilinear', scale_factor=1 / 4),
            nn.Conv2d(inchannels, inchannels, kernel_size=3, stride=1, padding=1, groups=1, bias=bias)
        )
        self.channel = ChannelAttention(inchannels * 3)
        self.out = nn.Sequential(
            nn.Conv2d(inchannels * 3, outchannels, kernel_size=3, stride=1, padding=1, groups=1, bias=bias),
            nn.LeakyReLU(0.2)
        )

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)

    def forward(self, x):
        y1 = self.scale1(x)
        y2 = F.interpolate(self.scale1_2(x), size=(y1.shape[2],y1.shape[3]), mode='bilinear', align_corners=True)
        y3 = F.interpolate(self.scale1_4(x), size=(y1.shape[2],y1.shape[3]), mode='bilinear', align_corners=True)
        y = torch.cat((y1, y2, y3), dim=1)
        y = self.channel(y) * y
        y = self.out(y)
        return y
    
class Progressive_Resampling_integral(nn.Module):
    def __init__(self, dim=32, input_resolution=16, num_heads=8, ssl=4,
                 mlp_ratio=4., qkv_bias=True, qk_scale=4, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = to_2tuple(input_resolution)
        self.num_heads = num_heads
        self.ssl = ssl
        self.mlp_ratio = mlp_ratio

        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self,H, W, x):
        x = rearrange(x, 'B C H W -> B (H W) C')
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        shortcut = x
        
        x = self.norm1(x)

        perm = torch.randperm(L, device=x.device) 
        inv_perm = torch.argsort(perm) 
        x = x[:, perm, :].contiguous()

        x = x.view(B, H, W, C)
        sampling_area = partition(x, self.ssl)  
        sampling_area = sampling_area.view(-1, self.ssl * self.ssl, C) 
        attn_sampling_area = self.attn(sampling_area)
        attn_sampling_area = attn_sampling_area.view(-1, self.ssl, self.ssl, C)
        x = reverse(attn_sampling_area, self.ssl, H, W)  
        x = x.view(B, H * W, C)

        x = x[:, inv_perm, :].contiguous()
        
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = rearrange(x, 'B (H W) C -> B C H W', H=H)
        return x

class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        out = (x - mean) / (std + self.eps)
        out = self.weight * out + self.bias
        return out
    
class Galerkin_integral(nn.Module):
    def __init__(self, midc, heads):
        super().__init__()

        self.headc = midc // heads
        self.heads = heads
        self.midc = midc

        self.qkv_proj = nn.Conv2d(midc, 3*midc, 1)
        self.o_proj1 = nn.Conv2d(midc, midc, 1)
        self.o_proj2 = nn.Conv2d(midc, midc, 1)

        self.kln = LayerNorm((self.heads, 1, self.headc))
        self.vln = LayerNorm((self.heads, 1, self.headc))

        self.act = nn.GELU()
    
    def forward(self, x):
        B, C, H, W = x.shape
        bias = x

        qkv = self.qkv_proj(x).permute(0, 2, 3, 1).reshape(B, H*W, self.heads, 3*self.headc)
        qkv = qkv.permute(0, 2, 1, 3)
        q, k, v = qkv.chunk(3, dim=-1)

        k = self.kln(k)
        v = self.vln(v)
        
        v = torch.matmul(k.transpose(-2,-1), v) / (H*W)
        v = torch.matmul(q, v)
        v = v.permute(0, 2, 1, 3).reshape(B, H, W, C)

        ret = v.permute(0, 3, 1, 2) + bias
        bias = self.o_proj2(self.act(self.o_proj1(ret))) + bias
        
        return bias

class integral(nn.Module):
    def __init__(self, img_size=64, in_chans=32, head=8, ssl=4):
        super().__init__()
        self.img_size = img_size
        self.in_channels = in_chans
        self.ssl = ssl
        self.PR = Progressive_Resampling_integral(dim=self.in_channels, input_resolution=img_size, num_heads=head, ssl=self.ssl)
        self.GI = Galerkin_integral(self.in_channels, 8)

    def forward(self, H, W, x):
        shortcut = x
        
        x = self.PR(H, W, x)
        x = self.GI(x)
        
        x = x + shortcut
        return x
      
class SFNO(nn.Module):
    def __init__(self, size = 64):
        super(SFNO, self).__init__()
        # self.args = args
        self.hschannels = 31
        self.mschannels = 3
        self.embed = 64
        self.img_size = size

        # self.spe1 = SpePyBlock(self.hschannels, 32)
        # self.spa1 = SpaPyBlock(self.mschannels, 32)
        
        # self.conv_block = nn.Conv2d(in_channels = 31+32, out_channels=self.embed, kernel_size=3, stride=1, padding=1)
        # self.conv_block = nn.Conv2d(in_channels = 34, out_channels=self.embed, kernel_size=3, stride=1, padding=1)
        self.conv_block = nn.Conv2d(in_channels = 34, out_channels=self.embed, kernel_size=1)
        # self.conv_block = LiftProjectBlock(in_channels = self.hschannels + self.mschannels, out_channels=self.embed, in_size=self.img_size, out_size=self.img_size, conv_kernel = 1)
        
        self.block1 = integral(img_size=self.img_size, in_chans=self.embed, head=8, ssl=4)
        self.block2 = integral(img_size=self.img_size, in_chans=self.embed, head=8, ssl=4)
        self.block3 = integral(img_size=self.img_size, in_chans=self.embed, head=8, ssl=2)
        # self.block4 = integral(img_size=self.img_size, in_chans=self.embed, head=8, ssl=2)
        
        # self.conv = nn.Sequential(
        #     nn.Conv2d(self.embed*4, self.hschannels, 3, 1, 1), 
        #     nn.LeakyReLU(0.2, True)
        # ) # 一版
        self.conv = nn.Sequential(
            nn.Conv2d(self.embed*4, self.hschannels, 1), 
            nn.LeakyReLU(0.2, True)
        )
        # self.conv = LiftProjectBlock(in_channels = self.embed*4, out_channels=self.mschannels, in_size=self.img_size, out_size=self.img_size, conv_kernel = 1)

        # self.conv00 = nn.Conv2d(128, 64, 1)
        # self.conv01 = nn.Conv2d(64, 64, 1)

        # self.fc1 = nn.Conv2d(64, 64, 1)
        # self.fc2 = nn.Conv2d(64, 64, 1)
        # self.act = nn.ReLU()
        # self.convw = nn.Conv2d(36, 32, 1)

        self.VI1_1 = VectorAttention(34, 34)
        self.VI1_2 = VectorAttention(34, 34)
        self.VI1_3 = VectorAttention(34, 34)    
    
    def forward(self, ms, rgb, sf):
        '''
        :param rgb:
        :param ms:
        :return:
        '''


        # rgb = rgb
        ms_up = F.interpolate(ms, scale_factor=sf, mode='bicubic', align_corners=False)
        
        # xt = self.query(ms, rgb, coord)
        # rgb1 = self.spa1(rgb)
        # ms_up2, ms_up4, ms_up8 = self.spe1(ms_up, ms_up, ms_up)

        # xt = torch.cat((rgb1, ms_up2, ms_up4, ms_up8), 1) 
        # xt = torch.cat((rgb1, ms_up), 1)
        # xt = torch.cat((rgb, ms_up2, ms_up4, ms_up8), 1) 
        xt = torch.cat((rgb, ms_up), 1)
        xt = self.VI1_3(self.VI1_2(self.VI1_1(xt)))

        _, _, H, W = xt.shape
        x1 = self.conv_block(xt)
        
        x2 = self.block1(H, W, x1)
        x3 = self.block2(H, W, x2)
        x4 = self.block3(H, W, x3)
        # x4 = self.block4(H, W, x4)
        
        xout = torch.cat((x1, x2, x3, x4), 1)
        result = self.conv(xout) + ms_up
        return result
