# https://github.com/dnhkng/PCAonGPU/blob/main/gpu_pca/pca_module.py
import math
import torch
import torch.nn as nn


class IncrementalPCA(nn.Module):

    def __init__(self, dim, feat_dim, forget_factor=0.9,):
        super().__init__()

        self.dim = dim
        self.feat_dim = feat_dim
        self.forget_factor = forget_factor

        proj = torch.zeros((dim, feat_dim)).float()
        sigma = torch.zeros(feat_dim).float()
        mean = torch.zeros(dim).float()
        self.register_buffer('proj', proj)
        self.register_buffer('sigma', sigma)
        self.register_buffer('mean', mean)
        self.n = 0


    def forward(self, X):
        m = X.shape[0]

        if self.n == 0:
            self.mean = X.mean(dim=0)
            X = X - self.mean.view(1, -1)
            U, S, Vt = torch.linalg.svd(X, full_matrices=False)
            self.proj = Vt.transpose(1, 0)[:, :self.feat_dim] # [dim, feat_dim]
            self.sigma = S[:self.feat_dim]
            self.n += m

            return

        new_mean = X.mean(dim=0)
        old_mean = self.mean.detach()
        X = X - new_mean.view(1, -1)
        X = torch.cat([self.forget_factor * self.sigma.detach().view(-1, 1) * self.proj.detach().transpose(1, 0), 
                       X, 
                       math.sqrt((self.n * m) / (self.n + m)) * (new_mean - old_mean).unsqueeze(0)]) # [feat_dim + m + 1, dim]

        U, S, Vt = torch.linalg.svd(X, full_matrices=False)
        self.proj = Vt.transpose(1, 0)[:, :self.feat_dim] # [dim, feat_dim]
        self.sigma = S[:self.feat_dim]
        self.mean = new_mean * m / (m + self.forget_factor * self.n) + old_mean * self.forget_factor * self.n / (m + self.forget_factor * self.n)
        self.n += m


    def transform(self, X, normalized=False):
        if normalized:
            proj = self.proj / self.sigma.view(1, -1)
        else:
            proj = self.proj

        x = (X - self.mean) @ proj

        return x


    def inverse(self, x, normalized=False):
        if normalized:
            proj = self.proj * self.sigma.view(1, -1)
        else:
            proj = self.proj

        X = x @ proj.transpose(1, 0) + self.mean
        return X
