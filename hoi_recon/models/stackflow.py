import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from smplx import SMPLHLayer
from pytorch3d.transforms import rotation_6d_to_matrix, matrix_to_rotation_6d

from hoi_recon.datasets.behave_extend_metadata import BEHAVEExtendMetaData
from hoi_recon.datasets.utils import perspective_projection, load_pickle
from hoi_recon.models.condition_flow import ConditionFlow


class Model(nn.Module):

    def __init__(self, cfg):
        super(Model, self).__init__()
        self.cfg = cfg
        resnet = models.resnet50(pretrained=True)
        modules = list(resnet.children())[:-1]
        modules.append(nn.Conv2d(resnet.fc.in_features, cfg.model.visual_feature_dim, kernel_size=1))
        self.backbone = nn.Sequential(*modules)

        self.header = FCHeader(cfg)
        self.stackflow = StackFlow(cfg)

        self.optimizer = torch.optim.AdamW(params=list(self.backbone.parameters()) + list(self.header.parameters()) + list(self.stackflow.parameters()),
                                           lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
        self.smpl_loss = SMPLLoss(cfg)
        self.offset_loss = OffsetLoss(cfg)
        self.flow_loss = FlowLoss(cfg, self.stackflow)

        self.loss_weights = {
            'loss_beta': 0.005,
            'loss_global_orient': 0.01,
            'loss_body_pose': 0.05,
            'loss_pose_6d': 0.1,
            'loss_pose_6d_samples': 0.1,
            'loss_trans': 0.1,
            'loss_joint_3d': 0.05,
            'loss_joint_2d': 0.01,
            'loss_gamma': 0.1,
            'loss_offset': 0.01,
            'object_reproj_loss': 0.01,
            'loss_theta_nll': 0.001,
            'loss_gamma_nll': 0.001,
            'loss_joint_2d_sample': 0.01,
            'object_sample_reproj_loss': 0.01,
        }


    def forward(self, batch):
        image = batch['image']
        batch_size = image.shape[0]
        visual_features = self.backbone(image)
        visual_features = visual_features.reshape(batch_size, -1)

        pred_betas, pred_cam = self.header(visual_features)
        pred_theta, pred_gamma = self.stackflow(visual_features, batch['object_labels'])
        out = {
            'visual_features': visual_features,
            'pred_betas': pred_betas,
            'pred_cam': pred_cam,
            'pred_theta': pred_theta,
            'pred_gamma': pred_gamma,
        }
        return out


    def forward_train(self, batch):
        pred = self.forward(batch)

        all_losses = {}
        loss_smpl = self.smpl_loss(pred, batch)
        loss_offset = self.offset_loss(pred, batch)
        loss_flow = self.flow_loss(pred, batch)
        all_losses.update(loss_smpl)
        all_losses.update(loss_offset)
        all_losses.update(loss_flow)
        loss = sum([v * self.loss_weights[k] for k, v in all_losses.items()])

        return loss, all_losses


    def inference(self, batch, debug=False):
        pred = self.forward(batch)

        pred_betas = pred['pred_betas']
        pred_pose6d = pred['pred_theta'] # [b, 6 * (21 + 1)]
        b = pred_pose6d.shape[0]
        pred_pose6d = pred_pose6d.reshape(b, -1, 6)
        pred_smpl_rotmat = rotation_6d_to_matrix(pred_pose6d)

        box_size = batch['box_size'].reshape(b, )
        box_center = batch['box_center'].reshape(b, 2)
        optical_center = batch['optical_center'].reshape(b, 2)
        focal_length = batch['focal_length'].reshape(b, 2)

        pred_cam = pred['pred_cam']
        x = pred_cam[:, 1] * box_size + box_center[:, 0] - optical_center[:, 0]
        y = pred_cam[:, 2] * box_size + box_center[:, 1] - optical_center[:, 1]
        z = focal_length[:, 0] / (box_size * pred_cam[:, 0] + 1e-9)
        x = x / focal_length[:, 0] * z
        y = y / focal_length[:, 1] * z
        hoi_trans = torch.stack([x, y, z], dim=-1) # [b, 3]
        hoi_rotmat = pred_smpl_rotmat[:, 0] # [b, 3, 3]

        pred_gamma = pred['pred_gamma'] # [b, dim]
        object_labels = batch['object_labels']
        pred_offsets = self.offset_loss.hooffset.decode(pred_gamma, object_labels)
        pred_obj_rel_R, pred_obj_rel_T = self.offset_loss.hooffset.decode_object_RT(pred_offsets, pred_betas, pred_smpl_rotmat[:, 1:], object_labels)

        results = {
            'pred_betas': pred_betas, # [b, 10]
            'pred_pose6d': pred_pose6d, # [b, 22, 6]
            'pred_smpl_body_pose': pred_smpl_rotmat[:, 1:], # [b, 21, 3, 3]
            'hoi_trans': hoi_trans, # [b, 3]
            'hoi_rotmat': hoi_rotmat, # [b, 3, 3]
            'pred_offsets': pred_offsets,
            'pred_obj_rel_R': pred_obj_rel_R,
            'pred_obj_rel_T': pred_obj_rel_T,

            'visual_features': pred['visual_features'], # for post-optimization
        }

        if debug:
            smpl_pred_out = self.smpl_loss.smpl(global_orient=pred_smpl_rotmat[:, 0:1], body_pose=pred_smpl_rotmat[:, 1:], betas=pred_betas)
            pred_joint_3d = smpl_pred_out.joints[:, :22]
            pred_joint_3d = pred_joint_3d - pred_joint_3d[:, :1]

            pred_cam_t_local = torch.stack([pred_cam[:, 1] / (pred_cam[:, 0] + 1e-9),
                                            pred_cam[:, 2] * focal_length[:, 0] / (focal_length[:, 1] * pred_cam[:, 0] + 1e-9),
                                            focal_length[:, 0] / (self.cfg.dataset.img_size * pred_cam[:, 0] + 1e-9 )], dim=-1)

            pred_joint_2d = perspective_projection(pred_joint_3d, trans=pred_cam_t_local, focal_length=focal_length / self.cfg.dataset.img_size)

            pred_obj_rel_R = batch['object_rel_rotmat']
            pred_obj_rel_T = batch['object_rel_trans']
            pred_obj_R = torch.matmul(hoi_rotmat, pred_obj_rel_R)
            pred_obj_t = torch.matmul(hoi_rotmat, pred_obj_rel_T.unsqueeze(-1)).squeeze(-1) + pred_cam_t_local

            object_keypoints = self.offset_loss.object_loss.object_keypoints[object_labels]
            object_keypoints_2d = perspective_projection(object_keypoints, trans=pred_obj_t, rotmat=pred_obj_R, focal_length=focal_length / self.cfg.dataset.img_size)

            results['pred_joint_2d'] = pred_joint_2d
            results['object_keypoints_2d'] = object_keypoints_2d

            num_samples = 4
            visual_features = pred['visual_features']

            visual_features = visual_features.unsqueeze(1).repeat(1, num_samples, 1).reshape(b * num_samples, -1)
            object_labels = object_labels.unsqueeze(1).repeat(1, num_samples).reshape(-1, )
            theta_samples, gamma_samples = self.flow_loss.flow.sampling(b * num_samples, visual_features, object_labels)

            pred_cam_t_local = pred_cam_t_local.unsqueeze(1).repeat(1, num_samples, 1).reshape(b * num_samples, 3)
            focal_length = focal_length.unsqueeze(1).repeat(1, num_samples, 1).reshape(b * num_samples, 2)
            pred_betas = pred_betas.unsqueeze(1).repeat(1, num_samples, 1).reshape(b * num_samples, 10)
            hoi_rotmat = hoi_rotmat.unsqueeze(1).repeat(1, num_samples, 1, 1).reshape(b * num_samples, 3, 3)

            pred_pose6d = theta_samples.reshape(b * num_samples, -1, 6)
            pred_smpl_rotmat = rotation_6d_to_matrix(pred_pose6d)
            smpl_pred_out = self.smpl_loss.smpl(global_orient=pred_smpl_rotmat[:, 0:1], body_pose=pred_smpl_rotmat[:, 1:], betas=pred_betas)
            pred_joint_3d = smpl_pred_out.joints[:, :22]
            pred_joint_3d = pred_joint_3d - pred_joint_3d[:, :1]
            pred_joint_2d = perspective_projection(pred_joint_3d, trans=pred_cam_t_local, focal_length=focal_length / self.cfg.dataset.img_size)

            gamma_samples = gamma_samples.reshape(b * num_samples, -1)
            pred_offsets = self.offset_loss.hooffset.decode(gamma_samples, object_labels)
            pred_obj_rel_R, pred_obj_rel_T = self.offset_loss.hooffset.decode_object_RT(pred_offsets, pred_betas, pred_smpl_rotmat[:, 1:], object_labels)
        
            pred_obj_R = torch.matmul(hoi_rotmat, pred_obj_rel_R)
            pred_obj_t = torch.matmul(hoi_rotmat, pred_obj_rel_T.unsqueeze(-1)).squeeze(-1) + pred_cam_t_local

            object_keypoints = self.offset_loss.object_loss.object_keypoints[object_labels]
            object_keypoints_2d = perspective_projection(object_keypoints, trans=pred_obj_t, rotmat=pred_obj_R, focal_length=focal_length / self.cfg.dataset.img_size)

            results['pred_joint_2d_samples'] = pred_joint_2d.reshape(b, num_samples, -1, 2)
            results['object_keypoints_2d_samples'] = object_keypoints_2d.reshape(b, num_samples, -1, 2)

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
            'header': self.header.state_dict(),
            'stackflow': self.stackflow.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, path)


    def load_checkpoint(self, path):
        state_dict = torch.load(path)
        self.backbone.load_state_dict(state_dict['backbone'])
        self.header.load_state_dict(state_dict['header'])
        self.stackflow.load_state_dict(state_dict['stackflow'])
        try:
            self.optimizer.load_state_dict(state_dict['optimizer'])
        except:
            print('Warning: Lacking weights for optimizer.')
        return state_dict['epoch']


class FCHeader(nn.Module):

    def __init__(self, cfg):
        super(FCHeader, self).__init__()
        self.layers = nn.Sequential(nn.Linear(cfg.model.visual_feature_dim, cfg.model.cam_head_dim),
                                    nn.ReLU(inplace=False),
                                    nn.Linear(cfg.model.cam_head_dim, 13))
        nn.init.xavier_uniform_(self.layers[2].weight, gain=0.02)

        init_cam = cfg.dataset.init_cam_translation
        self.register_buffer('init_cam', torch.tensor(init_cam, dtype=torch.float32).reshape(1, 3))


    def forward(self, c):
        # c: [b, dim]
        offset = self.layers(c)
        pred_betas = offset[:, :10]
        pred_cam = self.init_cam + offset[:, 10:]

        return pred_betas, pred_cam


class StackFlow(nn.Module):

    def __init__(self, cfg):
        super(StackFlow, self).__init__()
        self.cfg = cfg
        self.smpl_in_dim = (21 + 1) * 6
        self.smpl_c_flow = ConditionFlow(dim=self.smpl_in_dim, 
                                hidden_dim=cfg.model.smplflow.hidden_dim,
                                c_dim=cfg.model.visual_feature_dim,
                                num_blocks_per_layer=cfg.model.smplflow.num_blocks_per_layer,
                                num_layers=cfg.model.smplflow.num_layers,
                                dropout_probability=0.5)
        self.offset_c_flow = ConditionFlow(dim=cfg.model.offset.latent_dim,
                                  hidden_dim=cfg.model.offsetflow.hidden_dim,
                                  c_dim=cfg.model.visual_feature_dim,
                                  num_blocks_per_layer=cfg.model.offsetflow.num_blocks_per_layer,
                                  num_layers=cfg.model.offsetflow.num_layers,
                                  dropout_probability=0.5)
        self.object_embedding = nn.Embedding(cfg.dataset.num_object, cfg.model.visual_feature_dim)
        self.smpl_pose_embedding = nn.Linear(self.smpl_in_dim, cfg.model.visual_feature_dim)


    def forward(self, visual_features, object_labels, theta_z=None, gamma_z=None):
        b = visual_features.shape[0]
        if theta_z is None:
            theta_z = torch.zeros(b, self.smpl_in_dim, dtype=visual_features.dtype, device=visual_features.device)
        if gamma_z is None:
            gamma_z = torch.zeros(b, self.cfg.model.offset.latent_dim, dtype=visual_features.dtype, device=visual_features.device)

        theta, _ = self.smpl_c_flow.inverse(theta_z, visual_features)

        offset_condition = visual_features
        offset_condition = offset_condition + self.object_embedding(object_labels)
        offset_condition = offset_condition + self.smpl_pose_embedding(theta)
        gamma, _ = self.offset_c_flow.inverse(gamma_z, offset_condition)

        return theta, gamma


    def gaussianize(self, theta, gamma, visual_features, object_labels):
        theta_z = self.smpl_c_flow.forward(theta, visual_features)

        offset_condition = visual_features
        offset_condition = offset_condition + self.object_embedding(object_labels)
        offset_condition = offset_condition + self.smpl_pose_embedding(theta)
        gamma_z = self.offset_c_flow.forward(gamma, offset_condition)

        return theta_z, gamma_z


    def sampling(self, n_samples, visual_features, object_labels, z_std=1.):
        theta, _ = self.smpl_c_flow.sampling(n_samples, visual_features, z_std)

        offset_condition = visual_features
        offset_condition = offset_condition + self.object_embedding(object_labels)
        offset_condition = offset_condition + self.smpl_pose_embedding(theta)
        gamma, _ = self.offset_c_flow.sampling(n_samples, offset_condition, z_std)

        return theta, gamma


    def log_prob(self, theta, gamma, visual_features, object_labels):
        theta_log_prob = self.smpl_c_flow.log_prob(theta, visual_features)

        offset_condition = visual_features
        offset_condition = offset_condition + self.object_embedding(object_labels)
        offset_condition = offset_condition + self.smpl_pose_embedding(theta)
        gamma_log_prob = self.offset_c_flow.log_prob(gamma, offset_condition)

        return theta_log_prob, gamma_log_prob


class HOOffset(nn.Module):

    def __init__(self, cfg):
        super(HOOffset, self).__init__()

        self.smpl = SMPLHLayer(model_path=cfg.model.smplh_dir, gender='male')
        self.dataset_metadata = BEHAVEExtendMetaData(cfg.dataset.root_dir)
        pca_models_path = 'data/datasets/behave_extend_pca_models_n{}_{}_d{}.pkl'.format(cfg.model.smpl_anchor_num, cfg.model.object_anchor_num, cfg.model.pca_dim)

        obj_templates = self.dataset_metadata.obj_mesh_templates
        pca_models = load_pickle(pca_models_path)
        object_anchors = []
        pca_means = []
        pca_components = []
        for object_idx in sorted(self.dataset_metadata.OBJECT_IDX2NAME.keys()):
            object_name = self.dataset_metadata.OBJECT_IDX2NAME[object_idx]
            verts = obj_templates[object_name][0]
            anchors = verts[pca_models[object_name]['object_anchor_indices']]
            object_anchors.append(anchors)
            pca_means.append(pca_models[object_name]['mean'])
            pca_components.append(pca_models[object_name]['components'])
        object_anchors = torch.tensor(object_anchors, dtype=torch.float32)
        self.register_buffer('object_anchors', object_anchors)
        smpl_anchor_indices = torch.tensor(pca_models[object_name]['smpl_anchor_indices'], dtype=torch.int64).reshape(-1)
        self.register_buffer('smpl_anchor_indices', smpl_anchor_indices)
        pca_means = torch.tensor(pca_means, dtype=torch.float32)
        self.register_buffer('pca_means', pca_means)
        pca_components = torch.tensor(pca_components, dtype=torch.float32)
        self.register_buffer('pca_components', pca_components)


    def encode(self, smpl_betas, smpl_body_pose_rotmat, object_rel_rotmat, object_rel_trans, object_labels):
        b = smpl_betas.shape[0]
        smpl_out = self.smpl(body_pose=smpl_body_pose_rotmat, betas=smpl_betas)
        smpl_J = smpl_out.joints
        smpl_v = smpl_out.vertices
        smpl_v = smpl_v - smpl_J[:, :1]
        smpl_anchors = torch.stack([v[self.smpl_anchor_indices] for v in smpl_v], dim=0)

        object_anchors = self.object_anchors[object_labels]
        object_anchors = torch.matmul(object_anchors, object_rel_rotmat.permute(0, 2, 1)) + object_rel_trans.unsqueeze(1)
        offsets = object_anchors.unsqueeze(1) - smpl_anchors.unsqueeze(2)
        offsets = offsets.reshape(b, -1)

        gamma = torch.matmul(self.pca_components[object_labels], (offsets - self.pca_means[object_labels]).unsqueeze(-1)).squeeze(-1)

        return gamma, offsets


    def decode(self, gamma, object_labels):
        offsets = torch.matmul(self.pca_components[object_labels].permute(0, 2, 1), gamma.unsqueeze(-1)).squeeze(-1) + self.pca_means[object_labels]
        return offsets


    def decode_object_RT(self, offsets, smpl_betas, smpl_body_pose_rotmat, object_labels):
        b = smpl_betas.shape[0]
        smpl_out = self.smpl(body_pose=smpl_body_pose_rotmat, betas=smpl_betas)
        smpl_J = smpl_out.joints
        smpl_v = smpl_out.vertices
        smpl_v = smpl_v - smpl_J[:, :1]
        smpl_anchors = torch.stack([v[self.smpl_anchor_indices] for v in smpl_v], dim=0)

        object_anchors = self.object_anchors[object_labels]
        m, n = smpl_anchors.shape[1], object_anchors.shape[1]
        offsets = offsets.reshape(b, m, n, 3)
        smpl_anchors = smpl_anchors.reshape(b, m, 1, 3).repeat(1, 1, n, 1)
        object_p = smpl_anchors + offsets
        P = object_p.reshape(b, -1, 3)
        object_q = object_anchors.reshape(b, 1, n, 3).repeat(1, m, 1, 1)
        Q = object_q.reshape(b, -1, 3)
        center_Q = Q.mean(1).reshape(b, -1, 3)
        Q = Q - center_Q
        svd_mat = P.transpose(1, 2) @ Q
        svd_mat = svd_mat.double() # [b, 3, 3]
        u, _, v = torch.svd(svd_mat)
        d = torch.det(u @ v.transpose(1, 2)) # [b, ]
        d = torch.cat([
            torch.ones(b, 2, device=u.device),
            d.unsqueeze(-1)], axis=-1) # [b, 3]
        d = torch.eye(3, device=u.device).unsqueeze(0) * d.view(-1, 1, 3)
        obj_rotmat = u @ d @ v.transpose(1, 2)
        obj_rotmat_pred = obj_rotmat.to(object_q.dtype) # (b * n, 3, 3)
        _Q = Q + center_Q
        obj_trans_pred = (P.transpose(1, 2) - obj_rotmat_pred @ _Q.transpose(1, 2)).mean(dim=2) # (n * b, 3)
        object_rel_R = obj_rotmat_pred.reshape(b, 3, 3)
        object_rel_T = obj_trans_pred.reshape(b, 3)

        return object_rel_R, object_rel_T


class SMPLLoss(nn.Module):

    def __init__(self, cfg):
        super(SMPLLoss, self).__init__()
        self.cfg = cfg
        self.params_loss = nn.MSELoss(reduction='none')
        self.joint_loss = nn.L1Loss(reduction='none')
        self.smpl = SMPLHLayer(model_path=cfg.model.smplh_dir, gender='male')


    def forward(self, preds, targets):

        pred_cam = preds['pred_cam'] # [b, 3]
        b = pred_cam.shape[0]
        pred_betas = preds['pred_betas'] # [b, 10]
        pred_pose6d = preds['pred_theta'] # [b, 6 * (21 + 1)]
        pred_pose6d = pred_pose6d.reshape(b, -1, 6)
        pred_smpl_rotmat = rotation_6d_to_matrix(pred_pose6d)

        gt_smpl_betas = targets['smpl_betas']
        gt_smpl_rotmat = targets['smpl_pose_rotmat']

        loss_beta = self.params_loss(pred_betas, gt_smpl_betas).sum() / b
        loss_global_orient = self.params_loss(pred_smpl_rotmat[:, 0], gt_smpl_rotmat[:, 0]).sum() / b
        loss_body_pose = self.params_loss(pred_smpl_rotmat[:, 1:], gt_smpl_rotmat[:, 1:]).sum() / b
        pred_pose6d = pred_pose6d.reshape(-1, 2, 3).permute(0, 2, 1)
        loss_pose_6d = self.params_loss(torch.matmul(pred_pose6d.permute(0, 2, 1), pred_pose6d),
            torch.eye(2, dtype=pred_pose6d.dtype, device=pred_pose6d.device).unsqueeze(0)).mean()

        gt_trans = targets['smpl_trans']
        box_size = targets['box_size'].reshape(b, )
        box_center = targets['box_center'].reshape(b, 2)
        optical_center = targets['optical_center'].reshape(b, 2)
        focal_length = targets['focal_length'].reshape(b, 2)

        x = pred_cam[:, 1] * box_size + box_center[:, 0] - optical_center[:, 0]
        y = pred_cam[:, 2] * box_size + box_center[:, 1] - optical_center[:, 1]
        z = focal_length[:, 0] / (box_size * pred_cam[:, 0] + 1e-9)
        x = x / focal_length[:, 0] * z
        y = y / focal_length[:, 1] * z
        pred_cam_t_global = torch.stack([x, y, z], dim=-1)
        loss_trans = F.l1_loss(pred_cam_t_global, gt_trans)

        smpl_pred_out = self.smpl(global_orient=pred_smpl_rotmat[:, 0:1], body_pose=pred_smpl_rotmat[:, 1:], betas=pred_betas)
        pred_joint_3d = smpl_pred_out.joints[:, :22]
        pred_joint_3d = pred_joint_3d - pred_joint_3d[:, :1]

        pred_cam_t_local = torch.stack([pred_cam[:, 1] / (pred_cam[:, 0] + 1e-9),
                                        pred_cam[:, 2] * focal_length[:, 0] / (focal_length[:, 1] * pred_cam[:, 0] + 1e-9),
                                        focal_length[:, 0] / (self.cfg.dataset.img_size * pred_cam[:, 0] + 1e-9 )], dim=-1)

        pred_joint_2d = perspective_projection(pred_joint_3d, trans=pred_cam_t_local, focal_length=focal_length / self.cfg.dataset.img_size)

        gt_joint_3d = targets['person_joint_3d']
        gt_joint_2d = targets['person_joint_2d']
        loss_joint_3d = self.joint_loss(pred_joint_3d - pred_joint_3d[:, 0:1], gt_joint_3d - gt_joint_3d[:, 0:1]).sum() / b
        loss_joint_2d = self.joint_loss(pred_joint_2d, gt_joint_2d).sum() / b

        losses = {
            'loss_beta': loss_beta,
            'loss_global_orient': loss_global_orient,
            'loss_body_pose': loss_body_pose,
            'loss_pose_6d': loss_pose_6d,
            'loss_trans': loss_trans,
            'loss_joint_3d': loss_joint_3d,
            'loss_joint_2d': loss_joint_2d,
        }

        return losses


class OffsetLoss(nn.Module):

    def __init__(self, cfg):
        super(OffsetLoss, self).__init__()
        self.cfg = cfg
        self.hooffset = HOOffset(cfg)
        self.object_loss = ObjectLoss(cfg)


    def forward(self, preds, targets):
        pred_gamma = preds['pred_gamma'] # [b, dim]
        b = pred_gamma.shape[0]

        object_labels = targets['object_labels']
        smpl_body_pose_rotmat = targets['smpl_pose_rotmat'][:, 1:]
        smpl_betas = targets['smpl_betas']
        object_rel_rotmat = targets['object_rel_rotmat']
        object_rel_trans = targets['object_rel_trans']

        gt_gamma, gt_offsets = self.hooffset.encode(smpl_betas, smpl_body_pose_rotmat, object_rel_rotmat, object_rel_trans, object_labels)

        loss_gamma = F.l1_loss(pred_gamma, gt_gamma)

        pred_betas = preds['pred_betas'] # [b, 10]
        pred_pose6d = preds['pred_theta'] # [b, 6 * (21 + 1)]
        pred_pose6d = pred_pose6d.reshape(b, -1, 6)
        pred_rotmat = rotation_6d_to_matrix(pred_pose6d)
        pred_global_pose_rotmat = pred_rotmat[:, 0]
        pred_smpl_body_rotmat = pred_rotmat[:, 1:]
        pred_offsets = self.hooffset.decode(pred_gamma, object_labels)
        pred_obj_rel_R, pred_obj_rel_T = self.hooffset.decode_object_RT(pred_offsets, pred_betas, pred_smpl_body_rotmat, object_labels)

        loss_offset = F.l1_loss(gt_offsets, pred_offsets)

        focal_length = targets['focal_length'].reshape(b, 2)
        pred_cam = preds['pred_cam'] # [b, 3]
        pred_cam_t_local = torch.stack([pred_cam[:, 1] / (pred_cam[:, 0] + 1e-9),
                                        pred_cam[:, 2] * focal_length[:, 0] / (focal_length[:, 1] * pred_cam[:, 0] + 1e-9),
                                        focal_length[:, 0] / (self.cfg.dataset.img_size * pred_cam[:, 0] + 1e-9)], dim=-1)

        pred_obj_R = torch.matmul(pred_global_pose_rotmat, pred_obj_rel_R)
        pred_obj_t = torch.matmul(pred_global_pose_rotmat, pred_obj_rel_T.unsqueeze(-1)).squeeze(-1) + pred_cam_t_local

        object_reproj_loss = self.object_loss(pred_obj_R, pred_obj_t, object_labels, targets['object_kpts_2d'], targets['object_kpts_weights'], focal_length / self.cfg.dataset.img_size)

        losses = {
            'loss_gamma': loss_gamma,
            'object_reproj_loss': object_reproj_loss,
            'loss_offset': loss_offset,
        }

        return losses


class ObjectLoss(nn.Module):

    def __init__(self, cfg):
        super(ObjectLoss, self).__init__()

        self.dataset_metadata = BEHAVEExtendMetaData(cfg.dataset.root_dir)

        num_object = len(self.dataset_metadata.OBJECT_IDX2NAME)
        object_keypoints = np.zeros((num_object, self.dataset_metadata.object_max_keypoint_num, 3))
        for idx, object_idx in enumerate(sorted(self.dataset_metadata.OBJECT_IDX2NAME.keys())):
            object_name = self.dataset_metadata.OBJECT_IDX2NAME[object_idx]
            keypoints = self.dataset_metadata.load_object_keypoints(object_name)
            object_keypoints[idx, :len(keypoints)] = keypoints
        object_keypoints = torch.tensor(object_keypoints, dtype=torch.float32)
        self.register_buffer('object_keypoints', object_keypoints)


    def forward(self, obj_R, obj_t, object_labels, gt_keypoints, keypoints_weights, focal_length):
        object_keypoints = self.object_keypoints[object_labels].clone()
        object_keypoints_2d = perspective_projection(object_keypoints, trans=obj_t, rotmat=obj_R, focal_length=focal_length)
        obj_reproj_loss = F.l1_loss(object_keypoints_2d, gt_keypoints, reduction='none')
        keypoints_weights = keypoints_weights.unsqueeze(-1).repeat(1, 1, 2)
        obj_reproj_loss = obj_reproj_loss * keypoints_weights
        obj_reproj_loss = obj_reproj_loss.sum() / keypoints_weights.sum()

        return obj_reproj_loss


class FlowLoss(nn.Module):

    def __init__(self, cfg, flow):
        super(FlowLoss, self).__init__()
        self.cfg = cfg
        self.flow = flow 

        self.joint_loss = nn.L1Loss(reduction='none')
        self.params_loss = nn.MSELoss(reduction='none')
        self.smpl = SMPLHLayer(model_path=cfg.model.smplh_dir, gender='male')
        self.hooffset = HOOffset(cfg)
        self.object_loss = ObjectLoss(cfg)


    def forward(self, preds, targets):
        visual_features = preds['visual_features']
        gt_smpl_rotmat = targets['smpl_pose_rotmat']
        b = gt_smpl_rotmat.shape[0]
        gt_pose_6d = matrix_to_rotation_6d(gt_smpl_rotmat).reshape(b, -1)

        object_labels = targets['object_labels']
        smpl_body_pose_rotmat = targets['smpl_pose_rotmat'][:, 1:]
        smpl_betas = targets['smpl_betas']
        object_rel_rotmat = targets['object_rel_rotmat']
        object_rel_trans = targets['object_rel_trans']

        gt_gamma, gt_offset = self.hooffset.encode(smpl_betas, smpl_body_pose_rotmat, object_rel_rotmat, object_rel_trans, object_labels)
        gt_pose_6d = gt_pose_6d + torch.randn_like(gt_pose_6d) * 0.001
        gt_gamma = gt_gamma + torch.randn_like(gt_gamma) * 0.01
        theta_log_prob, gamma_log_prob = self.flow.log_prob(gt_pose_6d, gt_gamma, visual_features, object_labels)
        loss_theta_nll = - theta_log_prob.mean()
        loss_gamma_nll = - gamma_log_prob.mean()

        num_samples = self.cfg.train.num_samples
        visual_features = visual_features.unsqueeze(1).repeat(1, num_samples, 1).reshape(b * num_samples, -1)
        object_labels = object_labels.unsqueeze(1).repeat(1, num_samples).reshape(-1 )
        theta_samples, gamma_samples = self.flow.sampling(b * num_samples, visual_features, object_labels)
        
        theta_samples = theta_samples.reshape(b * num_samples, -1)
        gamma_samples = gamma_samples.reshape(b * num_samples, -1)

        pred_betas = preds['pred_betas'].unsqueeze(1).repeat(1, num_samples, 1).reshape(b * num_samples, -1) # [b * n, 10]
        pred_pose6d = theta_samples.reshape(b * num_samples, -1, 6)
        pred_smpl_rotmat = rotation_6d_to_matrix(pred_pose6d)
        smpl_pred_out = self.smpl(global_orient=pred_smpl_rotmat[:, 0:1], body_pose=pred_smpl_rotmat[:, 1:], betas=pred_betas)
        pred_joint_3d = smpl_pred_out.joints[:, :22]
        pred_joint_3d = pred_joint_3d - pred_joint_3d[:, :1]

        pred_pose6d = pred_pose6d.reshape(-1, 2, 3).permute(0, 2, 1)
        loss_pose_6d_samples = self.params_loss(torch.matmul(pred_pose6d.permute(0, 2, 1), pred_pose6d), 
            torch.eye(2, dtype=pred_pose6d.dtype, device=pred_pose6d.device).unsqueeze(0)).mean()

        pred_cam = preds['pred_cam'].unsqueeze(1).repeat(1, num_samples, 1).reshape(b * num_samples, -1) # [b * n, 3]
        focal_length = targets['focal_length'].unsqueeze(1).repeat(1, num_samples, 1).reshape(b * num_samples, 2) # [b * n, 2]
        pred_cam_t_local = torch.stack([pred_cam[:, 1] / (pred_cam[:, 0] + 1e-9),
                                        pred_cam[:, 2] * focal_length[:, 0] / (focal_length[:, 1] * pred_cam[:, 0] + 1e-9),
                                        focal_length[:, 0] / (self.cfg.dataset.img_size * pred_cam[:, 0] + 1e-9 )], dim=-1)
        pred_joint_2d = perspective_projection(pred_joint_3d, trans=pred_cam_t_local, focal_length=focal_length / self.cfg.dataset.img_size)

        gt_joint_2d = targets['person_joint_2d'].unsqueeze(1).repeat(1, num_samples, 1, 1)
        gt_joint_2d = gt_joint_2d.reshape(b * num_samples, -1, 2)
        loss_joint_2d = self.joint_loss(pred_joint_2d, gt_joint_2d).mean()

        pred_offsets = self.hooffset.decode(gamma_samples, object_labels)
        pred_obj_rel_R, pred_obj_rel_T = self.hooffset.decode_object_RT(pred_offsets, pred_betas, pred_smpl_rotmat[:, 1:], object_labels)

        pred_global_pose_rotmat = pred_smpl_rotmat[:, 0]
        pred_obj_R = torch.matmul(pred_global_pose_rotmat, pred_obj_rel_R)
        pred_obj_t = torch.matmul(pred_global_pose_rotmat, pred_obj_rel_T.unsqueeze(-1)).squeeze(-1) + pred_cam_t_local

        gt_object_keypoints = targets['object_kpts_2d'].unsqueeze(1).repeat(1, num_samples, 1, 1).reshape(b * num_samples, -1, 2)
        keypoints_weights = targets['object_kpts_weights'].unsqueeze(1).repeat(1, num_samples, 1).reshape(b * num_samples, -1)

        object_reproj_loss = self.object_loss(pred_obj_R, pred_obj_t, object_labels, gt_object_keypoints, keypoints_weights, focal_length / self.cfg.dataset.img_size)

        losses = {
            'loss_theta_nll': loss_theta_nll,
            'loss_gamma_nll': loss_gamma_nll,
            'loss_joint_2d_sample': loss_joint_2d,
            'loss_pose_6d_samples': loss_pose_6d_samples,
            'object_sample_reproj_loss': object_reproj_loss,
        }

        return losses
