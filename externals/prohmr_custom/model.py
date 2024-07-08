import torch
import torch.nn.functional as F
import numpy as np
import pytorch_lightning as pl
from typing import Any, Dict, Tuple
from yacs.config import CfgNode

from smplx import SMPLHLayer, SMPLXLayer

from prohmr.utils import SkeletonRenderer
from prohmr.utils.geometry import aa_to_rotmat, perspective_projection
from prohmr.models.backbones import create_backbone
from prohmr.models.heads import SMPLFlow
from prohmr.models.losses import Keypoint3DLoss, Keypoint2DLoss, ParameterLoss


class ProHMR(pl.LightningModule):

    def __init__(self, cfg: CfgNode):
        super().__init__()
        self.cfg = cfg
        self.backbone = create_backbone(cfg)
        self.flow = SMPLFlow(cfg)

        self.depth_mean = cfg.DATASETS.CONFIG.SMPL_DEPTH_MEAN

        self.keypoint_3d_loss = Keypoint3DLoss(loss_type='l1')
        self.keypoint_2d_loss = Keypoint2DLoss(loss_type='l1')
        self.smpl_parameter_loss = ParameterLoss()

        if cfg.DATASETS.CONFIG.NAME == 'InterCap':
            self.smpl = SMPLXLayer(model_path='./data/models/smplx', gender='neutral')
        else: # BEHAVE
            self.smpl = SMPLHLayer(model_path='./data/models/smplh', gender='male')

        self.register_buffer('initialized', torch.tensor(False))
        self.automatic_optimization = False


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(params=list(self.backbone.parameters()) + list(self.flow.parameters()),
            lr=self.cfg.TRAIN.LR, weight_decay=self.cfg.TRAIN.WEIGHT_DECAY)

        return optimizer


    def forward_step(self, batch: Dict, train: bool = False) -> Dict:

        if train:
            num_samples = self.cfg.TRAIN.NUM_TRAIN_SAMPLES
        else:
            num_samples = self.cfg.TRAIN.NUM_TEST_SAMPLES

        x = batch['img']
        batch_size = x.shape[0]

        conditioning_feats = self.backbone(x)

        # assert self.initialized.items() # we have loaded pretrained weights

        if num_samples > 1:
            pred_smpl_params, pred_cam, log_prob, _, pred_pose_6d = self.flow(conditioning_feats, num_samples=num_samples - 1)
            z_0 = torch.zeros(batch_size, 1, self.cfg.MODEL.FLOW.DIM, device=x.device)
            pred_smpl_params_mode, pred_cam_mode, log_prob_mode, _, pred_pose_6d_mode = self.flow(conditioning_feats, z=z_0)
            pred_smpl_params = {k: torch.cat([pred_smpl_params_mode[k], v], dim=1) for k, v in pred_smpl_params.items()}
            pred_cam = torch.cat([pred_cam_mode, pred_cam], dim=1)
            log_prob = torch.cat([log_prob_mode, log_prob], dim=1)
            pred_pose_6d = torch.cat([pred_pose_6d_mode, pred_pose_6d], dim=1)
        else:
            z_0 = torch.zeros(batch_size, 1, self.cfg.MODEL.FLOW.DIM, device=x.device)
            pred_smpl_params, pred_cam, log_prob, _, pred_pose_6d = self.flow(conditioning_feats, z=z_0)

        output = {}
        output['pred_cam'] = pred_cam
        output['pred_smpl_params'] = {k: v.clone() for k, v in pred_smpl_params.items()}
        output['log_prob'] = log_prob.detach()
        output['conditioning_feats'] = conditioning_feats
        output['pred_pose_6d'] = pred_pose_6d[:, :, :-6 * 2]

        focal_length = batch['focal_length'].unsqueeze(1).repeat(1, num_samples, 1)
        pred_cam_t = torch.stack([pred_cam[:, :, 1], pred_cam[:, :, 2], 
            2 * focal_length[:, :, 0] / (self.cfg.MODEL.IMAGE_SIZE * pred_cam[:, :, 0] + 1e-9)], dim=-1)
        output['pred_cam_t'] = pred_cam_t

        pred_smpl_params['global_orient'] = pred_smpl_params['global_orient'].reshape(batch_size * num_samples, -1, 3, 3)
        pred_smpl_params['body_pose'] = pred_smpl_params['body_pose'].reshape(batch_size * num_samples, -1, 3, 3)[:, :21]
        pred_smpl_params['betas'] = pred_smpl_params['betas'].reshape(batch_size * num_samples, -1)
        smpl_output = self.smpl(**{k: v.float() for k, v in pred_smpl_params.items()}, pose2rot=False)

        pred_keypoints_3d = smpl_output.joints[:, :22]
        pred_keypoints_3d = pred_keypoints_3d - pred_keypoints_3d[:, :1]
        pred_vertices = smpl_output.vertices
        output['pred_keypoints_3d'] = pred_keypoints_3d.reshape(batch_size, num_samples, -1, 3)
        output['pred_vertices'] = pred_vertices.reshape(batch_size, num_samples, -1, 3)

        pred_cam_t = pred_cam_t.reshape(-1, 3)
        focal_length = focal_length.reshape(-1, 2)
        pred_keypoints_2d = perspective_projection(pred_keypoints_3d, translation=pred_cam_t, focal_length=focal_length / self.cfg.MODEL.IMAGE_SIZE)
        output['pred_keypoints_2d'] = pred_keypoints_2d.reshape(batch_size, num_samples, -1, 2)

        roi_center = batch['roi_center'].unsqueeze(1).repeat(1, num_samples, 1)
        roi_size = batch['roi_size'].unsqueeze(1).repeat(1, num_samples, 1)
        focal_length = focal_length.reshape(batch_size, num_samples, 2)
        pred_cam = pred_cam.reshape(batch_size, num_samples, 3)
        pred_cam_t = pred_cam_t.reshape(batch_size, num_samples, 3)
        K = batch['K'].unsqueeze(1).repeat(1, num_samples, 1, 1)
        u = pred_cam_t[:, :, 0] / pred_cam_t[:, :, 2] * focal_length[:, :, 0] / self.cfg.MODEL.IMAGE_SIZE * roi_size[:, :, 0] + roi_center[:, :, 0]
        v = pred_cam_t[:, :, 1] / pred_cam_t[:, :, 2] * focal_length[:, :, 1] / self.cfg.MODEL.IMAGE_SIZE * roi_size[:, :, 1] + roi_center[:, :, 1]

        z = pred_cam_t[:, :, 2] * self.cfg.MODEL.IMAGE_SIZE / roi_size[:, :, 0]
        x = (u - K[:, :, 0, 2]) / focal_length[:, :, 0] * z
        y = (v - K[:, :, 1, 2]) / focal_length[:, :, 1] * z
        global_cam_t = torch.stack([x, y, z], dim=2)
        output['global_cam_t'] = global_cam_t

        return output


    def compute_loss(self, batch: Dict, output: Dict, train: bool = True) -> torch.Tensor:
        pred_smpl_params = output['pred_smpl_params']
        pred_pose_6d = output['pred_pose_6d']
        conditioning_feats = output['conditioning_feats']
        pred_keypoints_2d = output['pred_keypoints_2d']
        pred_keypoints_3d = output['pred_keypoints_3d']

        batch_size = pred_smpl_params['body_pose'].shape[0]
        num_samples = pred_smpl_params['body_pose'].shape[1]
        device = pred_smpl_params['body_pose'].device
        dtype = pred_smpl_params['body_pose'].dtype

        gt_keypoints_2d = batch['keypoints_2d']
        gt_keypoints_3d = batch['keypoints_3d']
        gt_smpl_params = batch['smpl_params']
        has_smpl_params = batch['has_smpl_params']
        is_axis_angle = batch['smpl_params_is_axis_angle']

        loss_keypoints_2d = self.keypoint_2d_loss(pred_keypoints_2d, gt_keypoints_2d.unsqueeze(1).repeat(1, num_samples, 1, 1))
        loss_keypoints_3d = self.keypoint_3d_loss(pred_keypoints_3d, gt_keypoints_3d.unsqueeze(1).repeat(1, num_samples, 1, 1), pelvis_id=0)

        loss_smpl_params = {}
        for k, pred in pred_smpl_params.items():
            gt = gt_smpl_params[k].unsqueeze(1).repeat(1, num_samples, 1).view(batch_size * num_samples, -1)
            if is_axis_angle[k].all():
                gt = aa_to_rotmat(gt.reshape(-1, 3)).view(batch_size * num_samples, -1, 3, 3)
            has_gt = has_smpl_params[k].unsqueeze(1).repeat(1, num_samples)
            loss_smpl_params[k] = self.smpl_parameter_loss(pred.reshape(batch_size, num_samples, -1), gt.reshape(batch_size, num_samples, -1), has_gt)

        loss_keypoints_2d_mode = loss_keypoints_2d[:, [0]].sum() / batch_size
        if loss_keypoints_2d.shape[1] > 1:
            loss_keypoints_2d_exp = loss_keypoints_2d[:, 1:].sum() / (batch_size * (num_samples - 1))
        else:
            loss_keypoints_2d_exp = torch.tensor(0., device=device, dtype=dtype)

        loss_keypoints_3d_mode = loss_keypoints_3d[:, [0]].sum() / batch_size
        if loss_keypoints_3d.shape[1] > 1:
            loss_keypoints_3d_exp = loss_keypoints_3d[:, 1:].sum() / (batch_size * (num_samples - 1))
        else:
            loss_keypoints_3d_exp = torch.tensor(0., device=device, dtype=dtype)
        loss_smpl_params_mode = {k: v[:, [0]].sum() / batch_size for k, v in loss_smpl_params.items()}
        if loss_smpl_params['body_pose'].shape[1] > 1:
            loss_smpl_params_exp = {k: v[:, 1:].sum() / (batch_size * (num_samples - 1)) for k, v in loss_smpl_params.items()}
        else:
            loss_smpl_params_exp = {k: torch.tensor(0., device=device, dtype=dtype) for k, v in loss_smpl_params.items()}

        smpl_params = {k: v.clone() for k, v in gt_smpl_params.items()}
        smpl_params['body_pose'] = aa_to_rotmat(smpl_params['body_pose'].reshape(-1, 3)).reshape(batch_size, -1, 3, 3)[:, :, :, :2].permute(0, 1, 3, 2).reshape(batch_size, 1, -1)
        smpl_params['global_orient'] = aa_to_rotmat(smpl_params['global_orient'].reshape(-1, 3)).reshape(batch_size, -1, 3, 3)[:, :, :, :2].permute(0, 1, 3, 2).reshape(batch_size, 1, -1)
        smpl_params['betas'] = smpl_params['betas'].unsqueeze(1)

        if train:
            smpl_params = {k: v + self.cfg.TRAIN.SMPL_PARAM_NOISE_RATIO * torch.randn_like(v) for k, v in smpl_params.items()}

        if smpl_params['body_pose'].shape[0] > 0:
            log_prob, _ = self.flow.log_prob(smpl_params, conditioning_feats)
        else:
            log_prob = torch.zeros(1, device=device, dtype=dtype)
        loss_nll = - log_prob.mean()

        pred_pose_6d = pred_pose_6d.reshape(-1, 2, 3).permute(0, 2, 1)
        loss_pose_6d = ((torch.matmul(pred_pose_6d.permute(0, 2, 1), pred_pose_6d) - torch.eye(2, device=pred_pose_6d.device, dtype=pred_pose_6d.dtype).unsqueeze(0)) ** 2)
        loss_pose_6d = loss_pose_6d.reshape(batch_size, num_samples, -1)
        loss_pose_6d_mode = loss_pose_6d[:, 0].mean()
        loss_pose_6d_exp = loss_pose_6d[:, 1:].mean()

        pred_cam_t = output['global_cam_t']
        gt_cam_t = batch['cam_t'].unsqueeze(1).repeat(1, num_samples, 1)
        loss_cam_t = F.l1_loss(pred_cam_t, gt_cam_t)

        loss = self.cfg.LOSS_WEIGHTS['KEYPOINTS_3D_EXP'] * loss_keypoints_3d_exp+\
               self.cfg.LOSS_WEIGHTS['KEYPOINTS_2D_EXP'] * loss_keypoints_2d_exp+\
               self.cfg.LOSS_WEIGHTS['NLL'] * loss_nll+\
               self.cfg.LOSS_WEIGHTS['ORTHOGONAL'] * (loss_pose_6d_exp+loss_pose_6d_mode)+\
               sum([loss_smpl_params_exp[k] * self.cfg.LOSS_WEIGHTS[(k+'_EXP').upper()] for k in loss_smpl_params_exp])+\
               self.cfg.LOSS_WEIGHTS['KEYPOINTS_3D_MODE'] * loss_keypoints_3d_mode+\
               self.cfg.LOSS_WEIGHTS['KEYPOINTS_2D_MODE'] * loss_keypoints_2d_mode+\
               sum([loss_smpl_params_mode[k] * self.cfg.LOSS_WEIGHTS[(k+'_MODE').upper()] for k in loss_smpl_params_mode]) +\
               self.cfg.LOSS_WEIGHTS['CAM_T'] * loss_cam_t

        losses = dict(loss=loss.detach(),
                      loss_nll=loss_nll.detach(),
                      loss_pose_6d_exp=loss_pose_6d_exp,
                      loss_pose_6d_mode=loss_pose_6d_mode,
                      loss_keypoints_2d_exp=loss_keypoints_2d_exp.detach(),
                      loss_keypoints_3d_exp=loss_keypoints_3d_exp.detach(),
                      loss_keypoints_2d_mode=loss_keypoints_2d_mode.detach(),
                      loss_keypoints_3d_mode=loss_keypoints_3d_mode.detach(),
                      loss_cam_t=loss_cam_t.detach())

        for k, v in loss_smpl_params_exp.items():
            losses['loss_' + k + '_exp'] = v.detach()
        for k, v in loss_smpl_params_mode.items():
            losses['loss_' + k + '_mode'] = v.detach()

        output['losses'] = losses

        return loss


    def forward(self, batch: Dict) -> Dict:
        return self.forward_step(batch, train=False)


    def training_step(self, batch: Dict, batch_idx: int,) -> Dict:
        optimizer = self.optimizers(use_pl_optimizer=True)
        batch_size = batch['img'].shape[0]
        output = self.forward_step(batch, train=True)
        pred_smpl_params = output['pred_smpl_params']
        num_samples = pred_smpl_params['body_pose'].shape[1]
        loss = self.compute_loss(batch, output, train=True)
        optimizer.zero_grad()
        self.manual_backward(loss)
        optimizer.step()

        if self.global_step > 0 and self.global_step % self.cfg.GENERAL.LOG_STEPS == 0:
            self.tensorboard_logging(batch, output, self.global_step, train=True)

        return output


    def validation_step(self, batch: Dict, batch_idx: int) -> Dict:
        batch_size = batch['img'].shape[0]
        output = self.forward_step(batch, train=False)
        pred_smpl_params = output['pred_smpl_params']
        num_samples = pred_smpl_params['body_pose'].shape[1]
        loss = self.compute_loss(batch, output, train=False)
        output['loss'] = loss
        self.tensorboard_logging(batch, output, self.global_step, train=False)

        return output


    def tensorboard_logging(self, batch: Dict, output: Dict, step_count: int, train: bool = True) -> None:

        mode = 'train' if train else 'val'
        summary_writer = self.logger.experiment
        batch_size = batch['keypoints_2d'].shape[0]
        images = batch['img']
        images = images * torch.tensor([0.229, 0.224, 0.225], device=images.device).reshape(1,3,1,1)
        images = images + torch.tensor([0.485, 0.456, 0.406], device=images.device).reshape(1,3,1,1)
        images = 255 * images.permute(0, 2, 3, 1).cpu().numpy()
        num_samples = self.cfg.TRAIN.NUM_TRAIN_SAMPLES if mode == 'train' else self.cfg.TRAIN.NUM_TEST_SAMPLES

        pred_keypoints_3d = output['pred_keypoints_3d'].detach().reshape(batch_size, num_samples, -1, 3)
        gt_keypoints_3d = batch['keypoints_3d']
        gt_keypoints_2d = batch['keypoints_2d']
        losses = output['losses']
        pred_cam = output['pred_cam'].detach().reshape(batch_size, num_samples, 3)
        pred_keypoints_2d = output['pred_keypoints_2d'].detach().reshape(batch_size, num_samples, -1, 2)

        for loss_name, val in losses.items():
            summary_writer.add_scalar(mode +'/' + loss_name, val.detach().item(), step_count)
