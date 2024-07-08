import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D

from hoi_recon.models.blocks import ActNorm, InvLinear, AffineCoupling, MLP, Gaussianize


class FlowStep(nn.Module):

    def __init__(self, dim, width, c_dim=None):
        super().__init__()
        if c_dim is not None:
            self.condition_embedding = MLP(c_dim, width, dim * 2)
        else:
            self.condition_embedding = None

        self.actnorm = ActNorm(dim)
        self.linear = InvLinear(dim)
        self.affine_coupling = AffineCoupling(dim, width)


    def forward(self, x, condition=None):
        # x: [b, dim], 
        if self.condition_embedding is not None and condition is not None:
            c_hidden = self.condition_embedding(condition)
            scale_drift, bias_drift = c_hidden[:, 0::2], c_hidden[:, 1::2]
        else:
            scale_drift = bias_drift = None

        x, logdet1 = self.actnorm(x, scale_drift, bias_drift)
        x, logdet2 = self.linear(x)
        x, logdet3 = self.affine_coupling(x)

        logdet = logdet1 + logdet2 + logdet3
        return x, logdet


    def inverse(self, z, condition=None):
        if self.condition_embedding is not None and condition is not None:
            c_hidden = self.condition_embedding(condition)
            scale_drift, bias_drift = c_hidden[:, 0::2], c_hidden[:, 1::2]
        else:
            scale_drift = bias_drift = None

        z, logdet3 = self.affine_coupling.inverse(z)
        z, logdet2 = self.linear.inverse(z)
        z, logdet1 = self.actnorm.inverse(z, scale_drift, bias_drift)

        logdet = logdet1 + logdet2 + logdet3
        return z, logdet


class Glow(nn.Module):

    def __init__(self, dim, width, c_dim=None, depth=4):
        super().__init__()
        self.dim = dim
        self.flows = nn.ModuleList([
            FlowStep(dim, width, c_dim) for _ in range(depth)
        ])
        self.gaussianize = Gaussianize(dim)
        self.register_buffer('base_dist_mean', torch.zeros(1))
        self.register_buffer('base_dist_var', torch.ones(1))


    @property
    def base_dist(self):
        return D.Normal(self.base_dist_mean, self.base_dist_var)


    def forward(self, x, condition=None, gaussianize=True):
        # x: [b, n]
        sum_logdets = 0
        for flow in self.flows:
            x, logdet = flow(x, condition)
            sum_logdets += logdet

        if gaussianize:
            x, logdet = self.gaussianize(torch.zeros_like(x), x)
            sum_logdets = sum_logdets + logdet

        return x, sum_logdets


    def inverse(self, z, condition=None, gaussianize=True):
        sum_logdets = 0
        if gaussianize:
            z, logdet = self.gaussianize.inverse(torch.zeros_like(z), z)
            sum_logdets = sum_logdets + logdet

        for flow in reversed(self.flows):
            z, logdet = flow.inverse(z, condition)
            sum_logdets += logdet

        return z, sum_logdets


    def sampling(self, n_samples, condition=None, z_std=1.):
        z = z_std * self.base_dist.sample((n_samples, self.dim)).squeeze(-1)
        x, sum_logdets = self.inverse(z, condition)
        return x, sum_logdets


    def log_prob(self, x, condition=None):
        z, logdet = self.forward(x, condition)
        log_prob = self.base_dist.log_prob(z).sum(-1) + logdet
        log_prob = log_prob / z.shape[1]
        return log_prob
