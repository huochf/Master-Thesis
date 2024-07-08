import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D

from hoi_recon.models.blocks import (ActNorm, LULinear, AdditiveCoupling, MLP,
    Gaussianize, AdditiveCouplingSelfAttention, )


class TransFlowStep(nn.Module):

    def __init__(self, mask_feat, mask_seq, dim, width, pos_dim, c_dim=None, num_blocks=2, head_dim=32, num_heads=8, dropout_probability=0., swap=False):
        super().__init__()
        if c_dim is not None:
            self.condition_embedding = MLP(c_dim, width, dim * 2)
        else:
            self.condition_embedding = None

        self.pos_embedding = MLP(pos_dim, width, dim * 2)

        self.actnorm = ActNorm(dim)
        self.linear = LULinear(dim)
        self.additive_coupling = AdditiveCoupling(mask_feat, dim, width, num_blocks, c_dim, dropout_probability)
        self.attention = AdditiveCouplingSelfAttention(mask_seq, dim, pos_dim, width, c_dim, num_blocks, head_dim, num_heads, dropout_probability)


    def forward(self, x, pos, condition=None):
        b, n, c = x.shape
        attn_mask = x.new_zeros((n, n))
        assert n % 2 == 1
        _n = n // 2
        attn_mask[:_n, :_n] = torch.triu(x.new_ones((_n, _n)), diagonal=0)
        attn_mask[:, _n] = 1
        attn_mask[_n + 1:, _n + 1:] = torch.tril(x.new_ones((_n, _n)), diagonal=0)

        pos_hidden = self.pos_embedding(pos)
        scale_drift, bias_drift = pos_hidden[..., 0::2], pos_hidden[..., 0::2]

        if self.condition_embedding is not None and condition is not None:
            c_hidden = self.condition_embedding(condition)
            scale_drift = scale_drift + c_hidden[..., 0::2]
            bias_drift = bias_drift + c_hidden[..., 1::2]
        x, logdet1 = self.actnorm(x, scale_drift, bias_drift)
        x, logdet2 = self.linear(x)
        x, logdet3 = self.additive_coupling(x, condition)
        x, logdet4 = self.attention(x, pos, attn_mask, condition)
        logdet = logdet1 + logdet2 + logdet3 + logdet4

        return x, logdet


    def inverse(self, z, pos, condition=None):
        b, n, c = z.shape
        attn_mask = z.new_zeros((n, n))
        assert n % 2 == 1
        _n = n // 2
        attn_mask[:_n, :_n] = torch.triu(z.new_ones((_n, _n)), diagonal=0)
        attn_mask[:, _n] = 1
        attn_mask[_n + 1:, _n + 1:] = torch.tril(z.new_ones((_n, _n)), diagonal=0)

        pos_hidden = self.pos_embedding(pos)
        scale_drift, bias_drift = pos_hidden[..., 0::2], pos_hidden[..., 0::2]

        if self.condition_embedding is not None and condition is not None:
            c_hidden = self.condition_embedding(condition)
            scale_drift = scale_drift + c_hidden[..., 0::2]
            bias_drift = bias_drift + c_hidden[..., 1::2]

        z, logdet4 = self.attention.inverse(z, pos, attn_mask, condition)
        z, logdet3 = self.additive_coupling.inverse(z, condition)
        z, logdet2 = self.linear.inverse(z)
        z, logdet1 = self.actnorm.inverse(z, scale_drift, bias_drift)

        logdet = logdet1 + logdet2 + logdet3 + logdet4
        return z, logdet


class FlowLast(nn.Module):

    def __init__(self, mask, dim, width, pos_dim, num_blocks, c_dim=None, dropout_probability=0.):
        super().__init__()
        if c_dim is not None:
            self.condition_embedding = MLP(c_dim, width, dim * 2)
        else:
            self.condition_embedding = None

        self.pos_embedding = MLP(pos_dim, width, dim * 2)
        self.actnorm = ActNorm(dim)
        self.linear = LULinear(dim)
        self.additive_coupling = AdditiveCoupling(mask, dim, width, num_blocks, c_dim, dropout_probability)


    def forward(self, x, pos, condition=None):
        pos_hidden = self.pos_embedding(pos)
        scale_drift, bias_drift = pos_hidden[..., 0::2], pos_hidden[..., 1::2]

        if self.condition_embedding is not None and condition is not None:
            c_hidden = self.condition_embedding(condition)
            scale_drift = scale_drift + c_hidden[..., 0::2]
            bias_drift = bias_drift + c_hidden[..., 1::2]

        x, logdet1 = self.actnorm(x, scale_drift, bias_drift)
        x, logdet2 = self.linear(x)
        x, logdet3 = self.additive_coupling(x, condition)

        logdet = logdet1 + logdet2 + logdet3
        return x, logdet


    def inverse(self, z, pos, condition=None):
        pos_hidden = self.pos_embedding(pos)
        scale_drift, bias_drift = pos_hidden[..., 0::2], pos_hidden[..., 1::2]

        if self.condition_embedding is not None and condition is not None:
            c_hidden = self.condition_embedding(condition)
            scale_drift = scale_drift + c_hidden[..., 0::2]
            bias_drift = bias_drift + c_hidden[..., 1::2]

        z, logdet3 = self.additive_coupling.inverse(z, condition)
        z, logdet2 = self.linear.inverse(z)
        z, logdet1 = self.actnorm.inverse(z, scale_drift, bias_drift)

        logdet = logdet1 + logdet2 + logdet3
        return z, logdet


class TransFlow(nn.Module):

    def __init__(self, dim, width, pos_dim, seq_max_len, c_dim=None, num_blocks_per_layers=2, layers=4, head_dim=256, num_heads=8, dropout_probability=0.):
        super().__init__()
        self.dim = dim
        self.flows = []
        mask_feats = torch.ones(dim)
        mask_feats[::2] = -1
        mask_seq = torch.ones(seq_max_len)
        mask_seq[::2] = -1
        mask_seq[seq_max_len // 2] = -1 # make the reference frame is always identity
        for i in range(layers):
            self.flows.append(TransFlowStep(mask_feats, mask_seq, dim, width, pos_dim, c_dim, 
                num_blocks_per_layers, head_dim, num_heads, dropout_probability, swap=i % 2 == 0))
            mask_feats *= -1
            mask_seq *= -1
            mask_seq[seq_max_len // 2] = -1 # make the reference frame is always identity
        self.flows = nn.ModuleList(self.flows)
        self.flow_last = FlowLast(mask_feats, dim, width, pos_dim, num_blocks_per_layers, c_dim, dropout_probability)
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
            z, logdet = flow.inverse(z, pos, condition)
            sum_logdets += logdet

        return z, sum_logdets


    def sampling(self, n_samples, pos, condition=None, z_std=1.):
        b, n,  = pos.shape
        pos = pos.unsqueeze(1).repeat(1, n_samples, 1).view(n_samples * b, n)
        if condition is not None:
            b, cond_n, _ = condition.shape
            condition = condition.unsqueeze(1).repeat(1, n_samples, 1, 1).view(n_samples * b, cond_n, -1)

        z = z_std * self.base_dist.sample((n_samples * b, n, self.dim)).squeeze(-1)
        x, sum_logdets = self.inverse(z, pos, condition)
        x = x.view(b, n_samples, n, self.dim)
        sum_logdets = sum_logdets.view(b, n_samples, n) / self.dim

        return x, sum_logdets


    def log_prob(self, x, pos, condition=None):
        z, logdet = self.forward(x, pos, condition)
        log_prob = self.base_dist.log_prob(z).sum(-1) + logdet
        log_prob = log_prob / z.shape[-1]
        return log_prob
