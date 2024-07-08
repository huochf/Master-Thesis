import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D

from hoi_recon.models.blocks import (ActNorm, InvLinear, AffineCoupling, MLP,
    Gaussianize, CouplingAttention, )


class AttentionStep(nn.Module):

    def __init__(self, dim, pos_dim, width, head_dim=32, num_heads=8, swap=False):
        super().__init__()
        self.swap = swap
        self.attention1 = CouplingAttention(dim, pos_dim, width, head_dim, num_heads)
        self.attention2 = CouplingAttention(dim, pos_dim, width, head_dim, num_heads)


    def forward(self, x, pos):
        x1, x2 = x.chunk(2, dim=1)
        pos1, pos2 = pos.chunk(2, dim=1)

        if self.swap:
            _x1, _x2 = x2, x1
            _pos1, _pos2 = pos2, pos1
        else:
            _x1, _x2 = x1, x2
            _pos1, _pos2 = pos1, pos2

        _x1, logdet1 = self.attention1(_x1, _x2, _pos1, _pos2)
        _x2, logdet2 = self.attention2(_x2, _x1, _pos2, _pos1)

        if self.swap:
            x1, x2 = _x2, _x1
        else:
            x1, x2 = _x1, _x2

        x = torch.cat([x1, x2], dim=1)
        logdet = logdet1 + logdet2
        return x, logdet


    def inverse(self, z, pos):
        z1, z2 = z.chunk(2, dim=1)
        pos1, pos2 = pos.chunk(2, dim=1)

        if self.swap:
            _z1, _z2 = z2, z1
            _pos1, _pos2 = pos2, pos1
        else:
            _z1, _z2 = z1, z2
            _pos1, _pos2 = pos1, pos2

        _z2, logdet2 = self.attention2.inverse(_z2, _z1, _pos2, _pos1)
        _z1, logdet1 = self.attention1.inverse(_z1, _z2, _pos1, _pos2)

        if self.swap:
            z1, z2 = _z2, _z1 
        else:
            z1, z2 = _z1, _z2 

        z = torch.cat([z1, z2], dim=1)
        logdet = logdet1 + logdet2
        return z, logdet


class TransFlowStep(nn.Module):

    def __init__(self, dim, width, pos_dim, c_dim=None, head_dim=32, num_heads=8, swap=False):
        super().__init__()
        if c_dim is not None:
            self.condition_embedding = MLP(c_dim, width, dim * 2)
        else:
            self.condition_embedding = None

        self.pos_embedding = MLP(pos_dim, width, dim * 2)

        self.actnorm = ActNorm(dim)
        self.linear = InvLinear(dim)
        self.affine_coupling = AffineCoupling(dim, width)
        self.attention = AttentionStep(dim, pos_dim, width, head_dim, num_heads, swap)


    def forward(self, x, pos, condition=None):
        pos_hidden = self.pos_embedding(pos)
        scale_drift, bias_drift = pos_hidden[..., 0::2], pos_hidden[..., 1::2]

        if self.condition_embedding is not None and condition is not None:
            c_hidden = self.condition_embedding(condition)
            scale_drift = scale_drift + c_hidden[..., 0::2]
            bias_drift = bias_drift + c_hidden[..., 1::2]

        x, logdet1 = self.actnorm(x, scale_drift, bias_drift)
        x, logdet2 = self.linear(x)
        x, logdet3 = self.affine_coupling(x)
        x, logdet4 = self.attention(x, pos)

        logdet = logdet1 + logdet2 + logdet3 + logdet4
        return x, logdet


    def inverse(self, z, pos, condition=None):
        pos_hidden = self.pos_embedding(pos)
        scale_drift, bias_drift = pos_hidden[..., 0::2], pos_hidden[..., 1::2]

        if self.condition_embedding is not None and condition is not None:
            c_hidden = self.condition_embedding(condition)
            scale_drift = scale_drift + c_hidden[..., 0::2]
            bias_drift = bias_drift + c_hidden[..., 1::2]

        z, logdet4 = self.attention.inverse(z, pos)
        z, logdet3 = self.affine_coupling.inverse(z)
        z, logdet2 = self.linear.inverse(z)
        z, logdet1 = self.actnorm.inverse(z, scale_drift, bias_drift)
 
        logdet = logdet1 + logdet2 + logdet3 + logdet4
        return z, logdet


class FlowLast(nn.Module):

    def __init__(self, dim, width, pos_dim, c_dim=None):
        super().__init__()
        if c_dim is not None:
            self.condition_embedding = MLP(c_dim, width, dim * 2)
        else:
            self.condition_embedding = None

        self.pos_embedding = MLP(pos_dim, width, dim * 2)
        self.actnorm = ActNorm(dim)
        self.linear = InvLinear(dim)
        self.affine_coupling = AffineCoupling(dim, width)


    def forward(self, x, pos, condition=None):
        pos_hidden = self.pos_embedding(pos)
        scale_drift, bias_drift = pos_hidden[..., 0::2], pos_hidden[..., 1::2]

        if self.condition_embedding is not None and condition is not None:
            c_hidden = self.condition_embedding(condition)
            scale_drift = scale_drift + c_hidden[..., 0::2]
            bias_drift = bias_drift + c_hidden[..., 1::2]

        x, logdet1 = self.actnorm(x, scale_drift, bias_drift)
        x, logdet2 = self.linear(x)
        x, logdet3 = self.affine_coupling(x)

        logdet = logdet1 + logdet2 + logdet3
        return x, logdet


    def inverse(self, z, pos, condition=None):
        pos_hidden = self.pos_embedding(pos)
        scale_drift, bias_drift = pos_hidden[..., 0::2], pos_hidden[..., 1::2]

        if self.condition_embedding is not None and condition is not None:
            c_hidden = self.condition_embedding(condition)
            scale_drift = scale_drift + c_hidden[..., 0::2]
            bias_drift = bias_drift + c_hidden[..., 1::2]

        z, logdet3 = self.affine_coupling.inverse(z)
        z, logdet2 = self.linear.inverse(z)
        z, logdet1 = self.actnorm.inverse(z, scale_drift, bias_drift)
 
        logdet = logdet1 + logdet2 + logdet3
        return z, logdet



class TransFlow(nn.Module):

    def __init__(self, dim, width, pos_dim, seq_max_len, c_dim=None, depth=4, head_dim=256, num_heads=8):
        super().__init__()
        self.dim = dim
        self.flows = nn.ModuleList([
            TransFlowStep(dim, width, pos_dim, c_dim, head_dim, num_heads, swap=i % 2 == 0)
            for i in range(depth)
        ])
        self.flow_last = FlowLast(dim, width, pos_dim, c_dim)
        self.gaussianize = Gaussianize(dim)

        self.pos_embedding = nn.Embedding(seq_max_len, pos_dim)

        self.register_buffer('base_dist_mean', torch.zeros(1))
        self.register_buffer('base_dist_var', torch.ones(1))


    @property
    def base_dist(self):
        return D.Normal(self.base_dist_mean, self.base_dist_var)


    def forward(self, x, pos, condition=None):
        pos = self.pos_embedding(pos)

        # x: [b, n, c]
        sum_logdets = 0
        for flow in self.flows:
            x, logdet = flow(x, pos, condition)
            sum_logdets = sum_logdets + logdet

        x, logdet = self.flow_last(x, pos, condition)
        sum_logdets = sum_logdets + logdet

        z, logdet = self.gaussianize(torch.zeros_like(x), x)
        sum_logdets = sum_logdets + logdet
        return z, sum_logdets


    def inverse(self, z, pos, condition=None):
        pos = self.pos_embedding(pos)

        z, sum_logdets = self.gaussianize.inverse(torch.zeros_like(z), z)

        z, logdet = self.flow_last.inverse(z, pos, condition)
        sum_logdets = sum_logdets + logdet

        for flow in reversed(self.flows):
            z, logdet = flow.inverse(z, pos,condition)
            sum_logdets += logdet

        return z, sum_logdets


    def sampling(self, n_samples, pos, condition=None, z_std=1.):
        b, n,  = pos.shape
        b, ref_n, _ = ref.shape
        pos = pos.unsqueeze(0).repeat(n_samples, 1, 1).view(n_samples * b, n)
        if condition is not None:
            b, cond_n, _ = condition.shape
            condition = condition.unsqueeze(0).repeat(n_samples, 1, 1, 1).view(n_samples * b, cond_n, -1)

        z = z_std * self.base_dist.sample((n_samples * b, n, self.dim)).squeeze(-1)
        x, sum_logdets = self.inverse(z, pos, condition)

        return x, sum_logdets


    def log_prob(self, x, pos, condition=None):
        z, logdet = self.forward(x, pos, condition)
        log_prob = self.base_dist.log_prob(z).flatten(1).sum(-1) + logdet
        log_prob = log_prob / (z.shape[1] * z.shape[2])
        return log_prob
