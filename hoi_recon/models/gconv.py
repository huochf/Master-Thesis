#################################################################
# codes adapted from https://github.com/noahcao/Pixel2Mesh
#################################################################
import torch
import torch.nn as nn
import torch.nn.functional as F


def batch_mm(matrix, batch):
    return torch.stack([matrix.mm(b) for b in batch], dim=0)


def dot(x, y, sparse=False):
    if sparse:
        return batch_mm(x, y)
    else:
        return torch.matmul(x, y)


class NormLayer(nn.Module):

    def __init__(self, dim):
        super(NormLayer, self).__init__()
        self.dim = dim
        self.scale = nn.Parameter(torch.zeros(1, 1, dim))
        self.bias = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.xavier_uniform_(self.scale)
        nn.init.xavier_uniform_(self.bias)


    def forward(self, x):
        bias = self.bias
        scale = self.scale
        x = (x - bias) * torch.exp(-scale)
        return x


class GConv(nn.Module):

    def __init__(self, in_features, out_features, adj_mat, bias=True):
        super(GConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.register_buffer('adj_mat', adj_mat)

        self.weight = nn.Parameter(torch.zeros((in_features, out_features), dtype=torch.float))
        self.loop_weight = nn.Parameter(torch.zeros((in_features, out_features), dtype=torch.float))
        if bias:
            self.bias = nn.Parameter(torch.zeros((out_features, ), dtype=torch.float))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()


    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight.data)
        nn.init.xavier_uniform_(self.loop_weight.data)


    def zeroize(self, ):
        nn.init.zeros_(self.weight.data)
        nn.init.zeros_(self.loop_weight.data)


    def forward(self, inputs):
        support = torch.matmul(inputs, self.weight)
        support_loop = torch.matmul(inputs, self.loop_weight)
        output = dot(self.adj_mat, support, True) + support_loop
        if self.bias is not None:
            ret = output + self.bias
        else:
            ret = output
        return ret


    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GResBlock(nn.Module):

    def __init__(self, in_dim, hidden_dim, adj_mat, activation=None):
        super(GResBlock, self).__init__()

        self.conv1 = GConv(in_features=in_dim, out_features=hidden_dim, adj_mat=adj_mat)
        self.conv2 = GConv(in_features=hidden_dim, out_features=hidden_dim, adj_mat=adj_mat)
        self.activation = F.relu if activation else None
        self.norm1 = nn.BatchNorm1d(num_features=hidden_dim)
        self.norm2 = nn.BatchNorm1d(num_features=hidden_dim)


    def forward(self, inputs):
        x = self.conv1(inputs)
        if self.activation:
            x = self.activation(x)
        x = self.norm1(x.transpose(2, 1)).transpose(2, 1)
        x = self.conv2(x)
        if self.activation:
            x = self.activation(x)
        x = self.norm2(x.transpose(2, 1)).transpose(2, 1)

        return (inputs + x) * 0.5


class GBottleneck(nn.Module):

    def __init__(self, block_num, in_dim, hidden_dim, out_dim, adj_mat, activation=None):
        super(GBottleneck, self).__init__()

        resblock_layers = [GResBlock(in_dim=hidden_dim, hidden_dim=hidden_dim, adj_mat=adj_mat, activation=activation)
                           for _ in range(block_num)]
        self.blocks = nn.Sequential(*resblock_layers)
        self.conv1 = GConv(in_features=in_dim, out_features=hidden_dim, adj_mat=adj_mat)
        self.conv2 = GConv(in_features=hidden_dim, out_features=out_dim, adj_mat=adj_mat)
        self.conv2.zeroize()
        self.activation = F.relu if activation else None


    def forward(self, inputs):
        x = self.conv1(inputs)
        if self.activation:
            x = self.activation(x)
        x_hidden = self.blocks(x)
        x_out = self.conv2(x_hidden)

        return x_out, x_hidden


class MeshDeformer(nn.Module):

    def __init__(self, sphere, features_dim, coord_dim=3, hidden_dim=192, activation=True, layers_per_stages=6, stages=3):
        super(MeshDeformer, self).__init__()

        self.register_buffer('init_pts', sphere.coord)

        self.gcns = nn.ModuleList([
            GBottleneck(layers_per_stages, coord_dim + hidden_dim, hidden_dim, coord_dim, 
                sphere.adj_mat, activation) for _ in range(stages)
        ])


    def forward(self, features):
        batch_size = features.shape[0]
        n = self.init_pts.shape[0]
        coord_feats = features.unsqueeze(1).repeat(1, n, 1)
        coords = self.init_pts.unsqueeze(0).repeat(batch_size, 1, 1)

        coords_all_stages = []
        for gcn in self.gcns:
            coord_feats = torch.cat([coords, coord_feats], dim=2)
            coord_offsets, coord_feats = gcn(coord_feats)
            coords = coords + coord_offsets
            coords_all_stages.append(coords)

        return coords_all_stages
