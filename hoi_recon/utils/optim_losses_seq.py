import numpy as np
from scipy.spatial.transform import Rotation
import torch
import torch.nn as nn
import torch.nn.functional as F

# import neural_renderer as nr
from pytorch3d.transforms import axis_angle_to_matrix


def perspective_projection(points, trans=None, rotmat=None, focal_length=None, optical_center=None):
    # points: [b, n, 3], trans: [b, 3], rotmat: [b, 3, 3], focal_length: [b, 2], optical_center: [b, 2]
    if rotmat is not None:
        points = torch.matmul(points, rotmat.permute(0, 2, 1))
    if trans is not None:
        points = points + trans[:, None]

    if focal_length is not None:
        u = points[:, :, 0] / points[:, :, 2] * focal_length[:, 0:1]
        v = points[:, :, 1] / points[:, :, 2] * focal_length[:, 1:]
        points_2d = torch.stack([u, v], dim=2)
    else:
        u = points[:, :, 0] / points[:, :, 2]
        v = points[:, :, 1] / points[:, :, 2]
        points_2d = torch.stack([u, v], dim=2)

    if optical_center is not None:
        points_2d = points_2d + optical_center[:, None]

    return points_2d


class SMPLPostPriorLoss(nn.Module):

    def __init__(self, prohmr, smpl_betas, condition_feats):
        super().__init__()
        self.flow = prohmr.flow
        self.visual_feats = condition_feats
        self.smpl_betas = smpl_betas


    def forward(self, hoi_dict, begin_idx, batch_size):
        body_pose = hoi_dict['smpl_body_rotmat']
        b = body_pose.shape[0]
        body_pose = body_pose.reshape(b, -1, 3, 3)[:, :, :, :2].permute(0, 1, 3, 2)
        padding = torch.eye(3).to(body_pose.dtype).to(body_pose.device).reshape(1, 1, 3, 3)
        padding = padding.repeat(b, 2, 1, 1)[:, :, :, :2].permute(0, 1, 3, 2)
        body_pose = torch.cat([body_pose.reshape(b, 1, -1), padding.reshape(b, 1, -1)], dim=2)

        smpl_params = {
            'global_orient': hoi_dict['hoi_rotmat'].reshape(b, 1, 3, 3)[:, :, :, :2].permute(0, 1, 3, 2).reshape(b, 1, -1),
            'body_pose': body_pose,
        }
        log_prob, _ = self.flow.log_prob(smpl_params, self.visual_feats[begin_idx:begin_idx+batch_size])

        loss_beta = F.l1_loss(self.smpl_betas[begin_idx:begin_idx+batch_size], hoi_dict['smpl_betas'])
        return {
            'smpl_prior': - log_prob.mean(),
            'loss_shape_prior': loss_beta,
        }


class ObjectEproPnpLoss(nn.Module):

    def __init__(self, model_points, image_points, pts_confidence, focal_length, optical_center, rescale=True):
        super(ObjectEproPnpLoss, self).__init__()
        self.register_buffer('model_points', model_points)
        self.register_buffer('image_points', image_points)
        self.register_buffer('pts_confidence', pts_confidence)
        self.register_buffer('focal_length', focal_length)
        self.register_buffer('optical_center', optical_center)
        b = model_points.shape[0]
        self.register_buffer('weights', torch.ones(b, dtype=torch.float32))

        self.rescale = rescale


    def set_weights(self, weights):
        self.weights = weights


    def forward(self, hoi_dict,  begin_idx, batch_size):
        obj_rotmat = hoi_dict['obj_rotmat']
        obj_trans = hoi_dict['obj_trans']
        b = obj_rotmat.shape[0]
        reproj_points = perspective_projection(points=self.model_points[begin_idx:begin_idx+batch_size], trans=obj_trans, rotmat=obj_rotmat, 
            focal_length=self.focal_length[begin_idx:begin_idx+batch_size], optical_center=self.optical_center[begin_idx:begin_idx+batch_size])

        loss_obj_reproj = F.l1_loss(self.image_points[begin_idx:begin_idx+batch_size], reproj_points, reduction='none') * self.pts_confidence[begin_idx:begin_idx+batch_size]
        loss_obj_reproj = loss_obj_reproj.reshape(b, -1).mean(-1)
        loss_obj_reproj = loss_obj_reproj * self.weights[begin_idx:begin_idx+batch_size]

        return {
            'object_reproj_loss': loss_obj_reproj,
        }


class SMPLKpsProjLoss(nn.Module):

    def __init__(self, vitpose, cam_Ks):
        super().__init__()
        self.register_buffer('vitpose', vitpose)
        self.openpose_to_wholebody_indices = [0, 16, 15, 18, 17, 5, 2, 6, 3, 7, 4, 12, 9, 13, 10, 14, 11, 19, 20, 21, 22, 23, 24]
        self.register_buffer('cam_Ks', cam_Ks)


    def project(self, points3d, begin_idx, batch_size):
        u = points3d[:, :, 0] / points3d[:, :, 2] * self.cam_Ks[begin_idx:begin_idx+batch_size, 0, 0].unsqueeze(1) + self.cam_Ks[begin_idx:begin_idx+batch_size, 0, 2].unsqueeze(1)
        v = points3d[:, :, 1] / points3d[:, :, 2] * self.cam_Ks[begin_idx:begin_idx+batch_size, 1, 1].unsqueeze(1) + self.cam_Ks[begin_idx:begin_idx+batch_size, 1, 2].unsqueeze(1)
        return torch.stack([u, v], dim=2)


    def forward(self, smpl_out, begin_idx, batch_size):
        smpl_kps = smpl_out['openpose_kps3d'] # [b, n, 3]

        smpl_kps_2d = self.project(smpl_kps, begin_idx, batch_size)
        loss_kps = F.l1_loss(smpl_kps_2d[:, self.openpose_to_wholebody_indices], self.vitpose[begin_idx:begin_idx+batch_size, :23, :2], reduction='none')
        loss_kps = loss_kps * self.vitpose[begin_idx:begin_idx+batch_size, :23, 2:]
        return {
            'loss_body_kps2d': loss_kps.mean()
        }


class SmoothLoss(nn.Module):

    def __init__(self, ):
        super().__init__()


    def forward(self, hoi_dict, begin_idx, batch_size):
        smpl_v = hoi_dict['smpl_v']
        object_v = hoi_dict['object_v']
        object_t = hoi_dict['obj_rel_trans']
        object_r6d = hoi_dict['obj_rel_rot6d']

        if smpl_v.shape[0] > 1:
            loss_smo_smpl_v = ((smpl_v[:-1] - smpl_v[1:]) ** 2).sum(-1).mean()
            loss_smo_obj_v = ((object_v[:-1] - object_v[1:]) ** 2).sum(-1).mean()
            loss_obj_t = ((object_t[:-1] - object_t[1:]) ** 2).sum(-1).mean()
            loss_obj_r = ((object_r6d[:-1] - object_r6d[1:]) ** 2).sum(-1).mean()
        else:
            loss_smo_smpl_v = loss_smo_obj_v = torch.zeros(1).to(smpl_v.device)
            loss_obj_t = loss_obj_r = torch.zeros(1).to(smpl_v.device)


        return {
            'loss_smo_smpl_v': loss_smo_smpl_v,
            'loss_smo_obj_v': loss_smo_obj_v,
            'loss_smo_obj_t': loss_obj_t,
            'loss_smo_obj_r': loss_obj_r,
        }


class HOIKPS3DLoss(nn.Module):

    def __init__(self, model, hoi_feats, object_labels):
        super().__init__()
        self.model = model
        self.hoi_feats = hoi_feats
        self.object_labels = object_labels


    def forward(self, hoi_dict, begin_idx, batch_size):
        smpl_kps = hoi_dict['smpl_J_centered'][:, :22]
        object_kps = hoi_dict['object_kps_centered']
        b = smpl_kps.shape[0]
        hoi_kps = torch.cat([smpl_kps, object_kps], dim=1).reshape(b, -1)

        loss_kps = self.model.log_prob(hoi_kps, self.hoi_feats[begin_idx:begin_idx+batch_size], self.object_labels[begin_idx:begin_idx+batch_size])
        loss_kps = - loss_kps.mean()

        return {
            'loss_hoi_kps3d': loss_kps,
        }


class TransflowKps3DSeqLoss(nn.Module):

    def __init__(self, model, hoi_feats, window_raidus, alpha=1.):
        super().__init__()
        self.model = model
        self.window_raidus = window_raidus
        self.alpha = alpha
        seq_len_total = hoi_feats.shape[0]

        seq_len = model.seq_len
        hoi_feats = hoi_feats.unsqueeze(1).repeat(1, seq_len, 1)
        pos = torch.arange(seq_len).long().to(hoi_feats.device).reshape(1, -1).repeat(seq_len_total, 1)

        self.hoi_feats = hoi_feats
        self.pos = pos


    def get_weights(self, log_prob):
        b, n = log_prob.shape
        assert n % 2 == 1
        window_raidus = n // 2
        weights = (torch.arange(n).float().to(log_prob.device) - window_raidus) / window_raidus
        weights = self.alpha * weights ** 2
        weights = (1 + weights) * torch.exp(- weights)
        return weights.unsqueeze(0).repeat(b, 1)


    def forward(self, hoi_dict, begin_idx, batch_size):
        smpl_kps = hoi_dict['smpl_J_centered'][:, :22]
        object_kps = hoi_dict['object_kps_centered'][:, :8]
        hoi_kps = torch.cat([smpl_kps, object_kps], dim=1)
        interval_len, n_kps, _ = hoi_kps.shape

        n_seq = 2 * self.window_raidus + 1

        hoi_kps_padded = torch.cat([hoi_kps.new_zeros(self.window_raidus, n_kps, 3),
                                    hoi_kps,
                                    hoi_kps.new_zeros(self.window_raidus, n_kps, 3)], dim=0)
        sequence_indices = torch.arange(n_seq).reshape(1, -1) + torch.arange(interval_len).reshape(-1, 1)
        hoi_kps_seq = hoi_kps_padded[sequence_indices.reshape(-1)].reshape(interval_len, n_seq, n_kps, 3)
        mask = hoi_kps_seq.new_ones(interval_len, n_seq)
        mask_prev = torch.triu(hoi_kps_seq.new_ones(self.window_raidus, self.window_raidus), diagonal=1)
        mask_prev = torch.flip(mask_prev, dims=(0, ))
        mask[:min(interval_len, self.window_raidus), :self.window_raidus] = mask_prev[:min(interval_len, self.window_raidus)]
        mask_succ = torch.triu(hoi_kps_seq.new_ones(self.window_raidus, self.window_raidus), diagonal=1)
        mask_succ = torch.flip(mask_succ, dims=(1, ))
        mask[- min(interval_len, self.window_raidus):, - self.window_raidus:] = mask_succ[- min(interval_len, self.window_raidus):]

        x = hoi_kps_seq.reshape(interval_len, n_seq, -1)
        pos = self.pos[begin_idx:begin_idx + batch_size].reshape(interval_len, -1)
        hoi_feats = self.hoi_feats[begin_idx:begin_idx + batch_size].reshape(interval_len, n_seq, -1)
        log_prob = self.model.log_prob(x, pos, hoi_feats)
        print(log_prob.max())
        print(log_prob[32])
        weights = self.get_weights(log_prob)
        mask = mask.reshape(interval_len, n_seq)
        weights = weights * mask
        loss_kps_nll = - (weights * log_prob).sum() / weights.sum()

        return {
            'loss_kps_nll': loss_kps_nll,
        }


class MultiViewPseudoKps2DSeqLoss(nn.Module):

    def __init__(self, model, hoi_feats, window_raidus, alpha=1., n_views=8):
        super().__init__()
        self.model = model
        self.window_raidus = window_raidus
        self.alpha = alpha
        self.n_views = n_views
        seq_len_total = hoi_feats.shape[0]

        seq_len = model.seq_len
        hoi_feats = hoi_feats.unsqueeze(1).repeat(1, seq_len, 1)
        pos = torch.arange(seq_len).long().to(hoi_feats.device).reshape(1, -1).repeat(seq_len_total, 1)

        hoi_kps_samples, log_prob = model.sampling(n_views, pos, hoi_feats, z_std=1.) # [seq_len_total, n_views, seq_len, -1]
        hoi_kps_directions = hoi_kps_samples[..., :-3].detach() 
        self.register_buffer('hoi_kps_directions', hoi_kps_directions)
        cam_pos = hoi_kps_samples[..., -3:].detach() 
        # d = 2.36
        # cam_pos = cam_pos / (torch.norm(cam_pos, dim=-1, keepdim=True) + 1e-8) * d
        self.cam_pos = nn.Parameter(cam_pos) # [seq_len_total, n_views, seq_len, 3]
        self.hoi_feats = hoi_feats
        self.pos = pos


    def get_weights(self, log_prob):
        b, n = log_prob.shape
        assert n % 2 == 1
        window_raidus = n // 2
        weights = (torch.arange(n).float().to(log_prob.device) - window_raidus) / window_raidus
        weights = self.alpha * weights ** 2
        weights = (1 + weights) * torch.exp(- weights)
        return weights.unsqueeze(0).repeat(b, 1)


    def forward(self, hoi_dict, begin_idx, batch_size):
        smpl_kps = hoi_dict['smpl_J_centered'][:, :22]
        object_kps = hoi_dict['object_kps_centered'][:, :8]
        hoi_kps = torch.cat([smpl_kps, object_kps], dim=1)
        interval_len, n_kps, _ = hoi_kps.shape

        n_seq = 2 * self.window_raidus + 1

        hoi_kps_padded = torch.cat([hoi_kps.new_zeros(self.window_raidus, n_kps, 3),
                                    hoi_kps,
                                    hoi_kps.new_zeros(self.window_raidus, n_kps, 3)], dim=0)
        sequence_indices = torch.arange(n_seq).reshape(1, -1) + torch.arange(interval_len).reshape(-1, 1)
        hoi_kps_seq = hoi_kps_padded[sequence_indices.reshape(-1)].reshape(interval_len, n_seq, n_kps, 3)
        mask = hoi_kps_seq.new_ones(interval_len, n_seq)
        mask_prev = torch.triu(hoi_kps_seq.new_ones(self.window_raidus, self.window_raidus), diagonal=1)
        mask_prev = torch.flip(mask_prev, dims=(0, ))
        mask[:min(interval_len, self.window_raidus), :self.window_raidus] = mask_prev[:min(interval_len, self.window_raidus)]
        mask_succ = torch.triu(hoi_kps_seq.new_ones(self.window_raidus, self.window_raidus), diagonal=1)
        mask_succ = torch.flip(mask_succ, dims=(1, ))
        mask[- min(interval_len, self.window_raidus):, - self.window_raidus:] = mask_succ[- min(interval_len, self.window_raidus):]

        n_views = self.n_views
        hoi_kps_seq = hoi_kps_seq.unsqueeze(1).repeat(1, n_views, 1, 1, 1)

        cam_pos = self.cam_pos[begin_idx:begin_idx + batch_size].reshape(interval_len, n_views, n_seq, 1, 3)
        cam_directions = hoi_kps_seq - cam_pos
        cam_directions = cam_directions / (torch.norm(cam_directions, dim=-1, keepdim=True) + 1e-8)
        x = torch.cat([cam_directions, cam_pos], dim=-2).reshape(interval_len * n_views, n_seq, -1)

        pos = self.pos[begin_idx:begin_idx + batch_size].unsqueeze(1).repeat(1, n_views, 1).reshape(interval_len * n_views, -1)
        hoi_feats = self.hoi_feats[begin_idx:begin_idx + batch_size].unsqueeze(1).repeat(1, n_views, 1, 1).reshape(interval_len * n_views, n_seq, -1)
        log_prob = self.model.log_prob(x, pos, hoi_feats)
        weights = self.get_weights(log_prob)
        mask = mask.unsqueeze(1).repeat(1, n_views, 1).reshape(interval_len * n_views, n_seq)
        weights = weights * mask
        # # weights[:, 15] = 10
        loss_kps_nll = - (weights * log_prob).sum() / weights.sum()
        loss_kps_nll = - log_prob[:, 15].mean()

        return {
            'loss_kps_nll': loss_kps_nll,
        }
