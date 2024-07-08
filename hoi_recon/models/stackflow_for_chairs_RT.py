import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from smplx import SMPLXLayer
from pytorch3d.transforms import rotation_6d_to_matrix, matrix_to_rotation_6d, axis_angle_to_matrix

from hoi_recon.datasets.utils import perspective_projection, load_pickle
from hoi_recon.models.condition_flow import ConditionFlow
from hoi_recon.models.incrementalPCA import IncrementalPCA


class Model(nn.Module):

    def __init__(self, cfg):
        super(Model, self).__init__()
        self.cfg = cfg
        resnet = models.resnet50(pretrained=True)
        modules = list(resnet.children())[:-1]
        modules.append(nn.Conv2d(resnet.fc.in_features, cfg.model.visual_feature_dim, kernel_size=1))
        self.backbone = nn.Sequential(*modules)

        self.stackflow = StackFlow(cfg)

        self.optimizer = torch.optim.AdamW(params=list(self.backbone.parameters()) + list(self.stackflow.parameters()),
                                           lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
        self.offset_loss = OffsetLoss(cfg)
        self.flow_loss = FlowLoss(cfg, self.stackflow)

        self.loss_weights = {
            'loss_beta': 0.005,
            'loss_body_pose': 0.05,
            'loss_pose_6d': 0.1,
            'loss_pose_6d_samples': 0.1,
            'loss_gamma': 0.1,
            'loss_offset': 0.01,
            'loss_theta_nll': 0.001,
            'loss_gamma_nll': 0.001,
        }


    def forward(self, batch):
        image = batch['image']
        batch_size = image.shape[0]
        visual_features = self.backbone(image)
        visual_features = visual_features.reshape(batch_size, -1)

        pred_pose = batch['smpler_body_pose'] # [b, 3 * 21]
        b = pred_pose.shape[0]
        pred_smpl_rotmat = axis_angle_to_matrix(pred_pose.reshape(b, -1, 3))
        pred_pose6d = matrix_to_rotation_6d(pred_smpl_rotmat)
        pred_gamma = self.stackflow(visual_features, pred_pose6d.reshape(b, -1))
        out = {
            'visual_features': visual_features,
            'pred_gamma': pred_gamma,
        }
        return out


    def forward_train(self, batch):
        pred = self.forward(batch)

        all_losses = {}
        loss_offset = self.offset_loss(pred, batch)
        loss_flow = self.flow_loss(pred, batch)
        all_losses.update(loss_offset)
        all_losses.update(loss_flow)
        loss = sum([v * self.loss_weights[k] for k, v in all_losses.items()])

        return loss, all_losses


    def inference(self, batch, debug=False):
        pred = self.forward(batch)

        pred_betas = batch['smpler_betas']
        pred_pose = batch['smpler_body_pose'] # [b, 3 * 21]
        b = pred_pose.shape[0]
        pred_pose = pred_pose.reshape(b, -1, 3)
        pred_smpl_rotmat = axis_angle_to_matrix(pred_pose)
        pred_pose6d = matrix_to_rotation_6d(pred_smpl_rotmat)

        pred_gamma = pred['pred_gamma'] # [b, dim]
        object_anchors_org = batch['object_anchors']
        pred_obj_rel_R6d, pred_obj_rel_T = pred_gamma[:, 3:], pred_gamma[:, :3]
        pred_obj_rel_R = rotation_6d_to_matrix(pred_obj_rel_R6d)

        results = {
            'pred_betas': pred_betas, # [b, 10]
            'pred_pose6d': pred_pose6d, # [b, 21, 6]
            'pred_smpl_body_pose': pred_smpl_rotmat, # [b, 21, 3, 3]
            'pred_obj_rel_R': pred_obj_rel_R,
            'pred_obj_rel_T': pred_obj_rel_T,
            'visual_features': pred['visual_features'], # for post-optimization
        }

        return results


    def train_step(self, batch):
        self.optimizer.zero_grad()
        loss, all_losses = self.forward_train(batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters=self.parameters(), max_norm=10, norm_type=2)
        self.optimizer.step()

        return loss, all_losses


    def save_checkpoint(self, epoch, path):
        torch.save({
            'epoch': epoch,
            'backbone': self.backbone.state_dict(),
            'stackflow': self.stackflow.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, path)


    def load_checkpoint(self, path):
        state_dict = torch.load(path)
        self.backbone.load_state_dict(state_dict['backbone'])
        self.stackflow.load_state_dict(state_dict['stackflow'])
        try:
            self.optimizer.load_state_dict(state_dict['optimizer'])
        except:
            print('Warning: Lacking weights for optimizer.')
        return state_dict['epoch']


class StackFlow(nn.Module):

    def __init__(self, cfg):
        super(StackFlow, self).__init__()
        self.cfg = cfg
        self.smpl_in_dim = 21 * 6
        self.offset_c_flow = ConditionFlow(dim=cfg.model.offset.latent_dim,
                                  hidden_dim=cfg.model.offsetflow.hidden_dim,
                                  c_dim=cfg.model.visual_feature_dim,
                                  num_blocks_per_layer=cfg.model.offsetflow.num_blocks_per_layer,
                                  num_layers=cfg.model.offsetflow.num_layers,
                                  dropout_probability=0.5)
        self.smpl_pose_embedding = nn.Linear(self.smpl_in_dim, cfg.model.visual_feature_dim)


    def forward(self, visual_features, theta, gamma_z=None):
        b = visual_features.shape[0]
        if gamma_z is None:
            gamma_z = torch.zeros(b, self.cfg.model.offset.latent_dim, dtype=visual_features.dtype, device=visual_features.device)

        offset_condition = visual_features
        offset_condition = offset_condition + self.smpl_pose_embedding(theta)
        gamma, _ = self.offset_c_flow.inverse(gamma_z, offset_condition)

        return gamma


    def gaussianize(self, gamma, visual_features, theta):
        offset_condition = visual_features
        offset_condition = offset_condition + self.smpl_pose_embedding(theta)
        gamma_z = self.offset_c_flow.forward(gamma, offset_condition)

        return gamma_z


    def sampling(self, n_samples, visual_features, theta, z_std=1.):
        offset_condition = visual_features
        offset_condition = offset_condition + self.smpl_pose_embedding(theta)
        gamma, _ = self.offset_c_flow.sampling(n_samples, offset_condition, z_std)

        return gamma


    def log_prob(self, gamma, visual_features, theta):
        offset_condition = visual_features
        offset_condition = offset_condition + self.smpl_pose_embedding(theta)
        gamma_log_prob = self.offset_c_flow.log_prob(gamma, offset_condition)

        return gamma_log_prob


class OffsetLoss(nn.Module):

    def __init__(self, cfg):
        super(OffsetLoss, self).__init__()
        self.cfg = cfg
        self.params_loss = nn.MSELoss(reduction='none')


    def forward(self, preds, targets):
        pred_gamma = preds['pred_gamma'] # [b, dim]
        b = pred_gamma.shape[0]

        object_anchors_org = targets['object_anchors']
        smpl_body_pose_rotmat = targets['smpl_pose_rotmat']
        smpl_betas = targets['smpl_betas']
        object_rel_rotmat = targets['object_rel_rotmat']
        object_rel_trans = targets['object_rel_trans']

        object_rel_rot6d = matrix_to_rotation_6d(object_rel_rotmat)
        gt_gamma = torch.cat([object_rel_trans, object_rel_rot6d], dim=1)

        loss_gamma = F.l1_loss(pred_gamma, gt_gamma)

        pred_obj_rel_R6d, pred_obj_rel_T = pred_gamma[:, 3:], pred_gamma[:, :3]
        pred_obj_rel_R6d = pred_obj_rel_R6d.reshape(-1, 2, 3).permute(0, 2, 1)
        loss_offset = self.params_loss(torch.matmul(pred_obj_rel_R6d.permute(0, 2, 1), pred_obj_rel_R6d),
            torch.eye(2, dtype=pred_obj_rel_R6d.dtype, device=pred_obj_rel_R6d.device).unsqueeze(0)).mean()

        losses = {
            'loss_gamma': loss_gamma,
            'loss_offset': loss_offset,
        }

        return losses


class FlowLoss(nn.Module):

    def __init__(self, cfg, flow):
        super(FlowLoss, self).__init__()
        self.cfg = cfg
        self.flow = flow 

        self.joint_loss = nn.L1Loss(reduction='none')
        self.params_loss = nn.MSELoss(reduction='none')
        self.smpl = SMPLXLayer('data/models/smplx/', gender='male', use_pca=False, batch_size=cfg.train.batch_size)


    def forward(self, preds, targets):
        visual_features = preds['visual_features']
        gt_smpl_rotmat = targets['smpl_pose_rotmat']
        b = gt_smpl_rotmat.shape[0]
        gt_pose_6d = matrix_to_rotation_6d(gt_smpl_rotmat).reshape(b, -1)

        object_anchors_org = targets['object_anchors']
        smpl_body_pose_rotmat = targets['smpl_pose_rotmat']
        smpl_betas = targets['smpl_betas']
        object_rel_rotmat = targets['object_rel_rotmat']
        object_rel_trans = targets['object_rel_trans']
        object_rel_rot6d = matrix_to_rotation_6d(object_rel_rotmat)

        gt_gamma = torch.cat([object_rel_trans, object_rel_rot6d], dim=1)
        gt_pose_6d = gt_pose_6d + torch.randn_like(gt_pose_6d) * 0.001
        gt_gamma = gt_gamma + torch.randn_like(gt_gamma) * 0.001
        gamma_log_prob = self.flow.log_prob(gt_gamma, visual_features, gt_pose_6d)
        loss_gamma_nll = - gamma_log_prob.mean()

        losses = {
            'loss_gamma_nll': loss_gamma_nll,
        }

        return losses
