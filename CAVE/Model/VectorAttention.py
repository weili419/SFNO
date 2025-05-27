import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True, unbiased=False)
        out = (x - mean) / (std + self.eps)
        out = self.weight * out + self.bias
        return out


class VectorAttention(nn.Module):
    def __init__(self, in_channels, mlp_hidden_dim):
        super(VectorAttention, self).__init__()
        self.qkv_conv = nn.Conv2d(in_channels, in_channels * 3, kernel_size=1, bias=False)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, mlp_hidden_dim),
            LayerNorm(mlp_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden_dim, in_channels)
        )
        self.gate = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_conv(x)  # (b, c*3, h, w)
        q, k, v = torch.chunk(qkv, chunks=3, dim=1)

        k_unfold = F.unfold(k, kernel_size=3, padding=1).view(b, c, 9, h, w)  # (b, c, 9, h, w)
        v_unfold = F.unfold(v, kernel_size=3, padding=1).view(b, c, 9, h, w)  # (b, c, 9, h, w)

        q_expanded = q.unsqueeze(2)  # (b, c, 1, h, w)
        q_minus_k = q_expanded - k_unfold  # (b, c, 9, h, w)

        q_minus_k = q_minus_k.permute(0, 3, 4, 2, 1).contiguous()  # (b, h, w, 9, c)
        mlp_output = self.mlp(q_minus_k)  # (b, h, w, 9, c)

        attention_scores = F.softmax(mlp_output, dim=-2)  # (b, h, w, 9, c)

        neighbors_v = v_unfold.permute(0, 3, 4, 2, 1).contiguous()  # (b, h, w, 9, c)
        weighted_v = torch.sum(neighbors_v * attention_scores, dim=3)  # (b, h, w, c)
        weighted_v = weighted_v.permute(0, 3, 1, 2).contiguous()  # (b, c, h, w)

        base = self.gate(x)

        return weighted_v + base + x
