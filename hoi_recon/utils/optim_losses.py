import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch3d.transforms import matrix_to_rotation_6d

from hoi_recon.datasets.utils import perspective_projection


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


    def forward(self, hoi_dict):
        obj_rotmat = hoi_dict['obj_rotmat']
        obj_trans = hoi_dict['obj_trans'] + 1e-8
        b = obj_rotmat.shape[0]
        reproj_points = perspective_projection(points=self.model_points, trans=obj_trans, rotmat=obj_rotmat, 
            focal_length=self.focal_length, optical_center=self.optical_center)

        loss_obj_reproj = F.l1_loss(self.image_points, reproj_points, reduction='none') * self.pts_confidence
        loss_obj_reproj = loss_obj_reproj.reshape(b, -1).mean(-1)
        loss_obj_reproj = loss_obj_reproj * self.weights

        return {
            'object_reproj_loss': loss_obj_reproj,
        }


class SMPLKpsProjLoss(nn.Module):

    def __init__(self, vitpose, cam_Ks):
        super().__init__()
        self.register_buffer('vitpose', vitpose)
        self.openpose_to_wholebody_indices = [0, 16, 15, 18, 17, 5, 2, 6, 3, 7, 4, 12, 9, 13, 10, 14, 11, 19, 20, 21, 22, 23, 24]
        self.register_buffer('cam_Ks', cam_Ks)


    def project(self, points3d):
        u = points3d[:, :, 0] / points3d[:, :, 2] * self.cam_Ks[:, 0, 0].unsqueeze(1) + self.cam_Ks[:, 0, 2].unsqueeze(1)
        v = points3d[:, :, 1] / points3d[:, :, 2] * self.cam_Ks[:, 1, 1].unsqueeze(1) + self.cam_Ks[:, 1, 2].unsqueeze(1)
        return torch.stack([u, v], dim=2)


    def forward(self, smpl_out):
        smpl_kps = smpl_out['openpose_kps3d'] # [b, n, 3]

        smpl_kps_2d = self.project(smpl_kps)
        loss_kps = F.l1_loss(smpl_kps_2d[:, self.openpose_to_wholebody_indices], self.vitpose[:, :23, :2], reduction='none')
        loss_kps = loss_kps * self.vitpose[:, :23, 2:]
        return {
            'loss_body_kps2d': loss_kps.mean()
        }


class SMPLPostPriorLoss(nn.Module):

    def __init__(self, prohmr, smpl_betas, condition_feats):
        super().__init__()
        self.flow = prohmr.flow
        self.visual_feats = condition_feats
        self.smpl_betas = smpl_betas


    def forward(self, hoi_dict):
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
        log_prob, _ = self.flow.log_prob(smpl_params, self.visual_feats)

        loss_beta = F.l1_loss(self.smpl_betas, hoi_dict['smpl_betas'])
        return {
            'smpl_prior': - log_prob.mean(),
            'loss_shape_prior': loss_beta,
        }


class HOOffsetLoss(nn.Module):

    def __init__(self, stackflow, hooffset, hoi_feats, object_labels):
        super().__init__()
        self.stackflow = stackflow
        self.hooffset = hooffset
        self.hoi_feats = hoi_feats
        self.object_labels = object_labels


    def forward(self, hoi_dict):
        smpl_orient = hoi_dict['hoi_rot6d']
        smpl_body_pose6d = hoi_dict['smpl_body_pose6d']
        b = smpl_orient.shape[0]
        smpl_pose6d = torch.cat([smpl_orient.unsqueeze(1), smpl_body_pose6d], dim=1).reshape(b, -1)

        smpl_betas = hoi_dict['smpl_betas']
        obj_rel_rotmat = hoi_dict['obj_rel_rotmat']
        obj_rel_trans = hoi_dict['obj_rel_trans']
        smpl_body_rotmat = hoi_dict['smpl_body_rotmat']
        gamma, _ = self.hooffset.encode(smpl_betas, smpl_body_rotmat, obj_rel_rotmat, obj_rel_trans, self.object_labels)

        theta_log_prob, gamma_log_prob = self.stackflow.log_prob(smpl_pose6d, gamma, self.hoi_feats, self.object_labels)
        return {
            'loss_theta_nll': - theta_log_prob.mean(),
            'loss_gamma_nll': - gamma_log_prob.mean(),
        }


class HORTLoss(nn.Module):

    def __init__(self, stackflow, hoi_feats, object_labels):
        super().__init__()
        self.stackflow = stackflow
        self.hoi_feats = hoi_feats
        self.object_labels = object_labels


    def forward(self, hoi_dict):
        smpl_orient = hoi_dict['hoi_rot6d']
        smpl_body_pose6d = hoi_dict['smpl_body_pose6d']
        b = smpl_orient.shape[0]
        smpl_pose6d = torch.cat([smpl_orient.unsqueeze(1), smpl_body_pose6d], dim=1).reshape(b, -1)

        obj_rel_rotmat = hoi_dict['obj_rel_rotmat']
        object_rel_rot6d = matrix_to_rotation_6d(obj_rel_rotmat)
        obj_rel_trans = hoi_dict['obj_rel_trans']
        gamma = torch.cat([obj_rel_trans, object_rel_rot6d], dim=1)

        theta_log_prob, gamma_log_prob = self.stackflow.log_prob(smpl_pose6d, gamma, self.hoi_feats, self.object_labels)
        return {
            'loss_theta_nll': - theta_log_prob.mean(),
            'loss_gamma_nll': - gamma_log_prob.mean(),
        }


class HOIKPS3DLoss(nn.Module):

    def __init__(self, model, hoi_feats, n_obj_kps=0):
        super().__init__()
        self.model = model
        self.hoi_feats = hoi_feats
        self.n_obj_kps = n_obj_kps


    def forward(self, hoi_dict):
        smpl_kps = hoi_dict['smpl_J_centered'][:, :22]
        object_kps = hoi_dict['object_kps_centered'][:, :self.n_obj_kps]
        batch_size = smpl_kps.shape[0]
        hoi_kps = torch.cat([smpl_kps, object_kps], dim=1).reshape(batch_size, -1)

        loss_kps = self.model.log_prob(hoi_kps, self.hoi_feats)

        loss_kps = - loss_kps.mean()

        return {
            'loss_hoi_kps3d': loss_kps,
        }


class MultiViewPseudoKps2DLoss(nn.Module):

    def __init__(self, model, hoi_feats, n_obj_kps=8, n_views=8):
        super().__init__()
        self.model = model
        self.n_obj_kps = n_obj_kps
        self.n_views = n_views
        batch_size = hoi_feats.shape[0]
        hoi_feats = hoi_feats.unsqueeze(1).repeat(1, n_views, 1).reshape(batch_size * n_views, -1)
        hoi_kps_samples, log_prob = model.sampling(batch_size * n_views, hoi_feats, z_std=1.) # [batch_size, n_views, -1]

        cam_pos = hoi_kps_samples.reshape(batch_size, n_views, -1)[..., -3:].detach() 
        self.cam_pos = nn.Parameter(cam_pos) # [batch_size, n_views, 3]
        self.hoi_feats = hoi_feats


    def forward(self, hoi_dict):
        smpl_kps = hoi_dict['smpl_J_centered'][:, :22]
        object_kps = hoi_dict['object_kps_centered'][:, :self.n_obj_kps]
        hoi_kps = torch.cat([smpl_kps, object_kps], dim=1)
        b, n_kps, _ = hoi_kps.shape

        n_views = self.n_views
        hoi_kps_seq = hoi_kps.reshape(b, 1, n_kps, 3).repeat(1, n_views, 1, 1)

        cam_pos = self.cam_pos.reshape(b, n_views, 1, 3)
        cam_directions = hoi_kps_seq - cam_pos
        cam_directions = cam_directions / (torch.norm(cam_directions, dim=-1, keepdim=True) + 1e-8)
        x = torch.cat([cam_directions, cam_pos], dim=-2).reshape(b * n_views, -1)

        log_prob = self.model.log_prob(x, self.hoi_feats)
        loss_kps_nll = - log_prob.mean()

        return {
            'loss_kps_nll': loss_kps_nll,
        }


class TransKps3DLoss(nn.Module):

    def __init__(self, model, hoi_feats, n_obj_kps=8):
        super().__init__()
        self.model = model
        self.n_obj_kps = n_obj_kps
        seq_len_total = hoi_feats.shape[0]

        seq_len = model.seq_len
        self.hoi_feats = hoi_feats.unsqueeze(1)

        pos = torch.arange(seq_len).long().to(hoi_feats.device).reshape(1, -1).repeat(seq_len_total, 1)
        self.pos = pos[:, 15:16]


    def forward(self, hoi_dict):
        smpl_kps = hoi_dict['smpl_J_centered'][:, :22]
        object_kps = hoi_dict['object_kps_centered'][:, :self.n_obj_kps]
        hoi_kps = torch.cat([smpl_kps, object_kps], dim=1)
        b, n_kps, _ = hoi_kps.shape
        x = hoi_kps.reshape(b, 1, -1)

        log_prob = self.model.log_prob(x, self.pos, self.hoi_feats)
        loss_kps_nll = - log_prob.mean()

        return {
            'loss_kps_nll': loss_kps_nll,
        }


class TransMultiViewPseudoKps2DLoss(nn.Module):

    def __init__(self, model, hoi_feats, n_obj_kps=8, n_views=8):
        super().__init__()
        self.model = model
        self.n_obj_kps = n_obj_kps
        self.n_views = n_views
        seq_len_total = hoi_feats.shape[0]

        seq_len = model.seq_len
        hoi_feats = hoi_feats.unsqueeze(1).repeat(1, seq_len, 1)
        pos = torch.arange(seq_len).long().to(hoi_feats.device).reshape(1, -1).repeat(seq_len_total, 1)

        hoi_kps_samples, log_prob = model.sampling(n_views, pos, hoi_feats, z_std=1.) # [seq_len_total, n_views, seq_len, -1]

        cam_pos = hoi_kps_samples[..., 15, -3:].detach() 
        self.cam_pos = nn.Parameter(cam_pos) # [seq_len_total, n_views, 3]
        self.hoi_feats = hoi_feats[:, 15]
        self.pos = pos[:, 15]


    def forward(self, hoi_dict):
        smpl_kps = hoi_dict['smpl_J_centered'][:, :22]
        object_kps = hoi_dict['object_kps_centered'][:, :self.n_obj_kps]
        hoi_kps = torch.cat([smpl_kps, object_kps], dim=1)
        b, n_kps, _ = hoi_kps.shape

        n_views = self.n_views
        hoi_kps_seq = hoi_kps.reshape(b, 1, n_kps, 3).repeat(1, n_views, 1, 1)

        cam_pos = self.cam_pos.reshape(b, n_views, 1, 3)
        cam_directions = hoi_kps_seq - cam_pos
        cam_directions = cam_directions / (torch.norm(cam_directions, dim=-1, keepdim=True) + 1e-8)
        x = torch.cat([cam_directions, cam_pos], dim=-2).reshape(b * n_views, 1, -1)

        pos = self.pos.unsqueeze(1).repeat(1, n_views).reshape(b * n_views, -1)
        hoi_feats = self.hoi_feats.unsqueeze(1).repeat(1, n_views, 1).reshape(b * n_views, 1, -1)
        log_prob = self.model.log_prob(x, pos, hoi_feats)
        loss_kps_nll = - log_prob.mean()

        return {
            'loss_kps_nll': loss_kps_nll,
        }
