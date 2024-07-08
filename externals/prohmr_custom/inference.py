import os
import argparse
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from pytorch3d.transforms import matrix_to_axis_angle, axis_angle_to_matrix, matrix_to_quaternion, quaternion_to_matrix
from smplx import SMPLHLayer

from prohmr.configs import get_config
from prohmr.datasets import ProHMRDataModule
from prohmr_custom.model import ProHMR
from prohmr_custom.datasets.behave_extend_image_dataset import HumanImageDataset
from transflow.datasets.utils import load_json, save_pickle, load_J_regressor, load_pickle
from transflow.datasets.behave_extend_metadata import BEHAVEExtendMetaData



def smooth_sequence(sequence, windows=5):
    sequence = np.stack(sequence, axis=0)
    seq_n, kps_n, d = sequence.shape

    confidence_score_org = sequence[:, :, 2:]
    sequence = np.concatenate([np.zeros((windows // 2, kps_n, 3)), sequence, np.zeros((windows // 2, kps_n, 3))], axis=0) # [seq_n + windows - 1, n, 3]
    confidence_score = sequence[:, :, 2:]

    smooth_kps = np.stack([
        sequence[i: seq_n + i] for i in range(windows)
    ], axis=0)    
    confidence_score = np.stack([
        confidence_score[i: seq_n + i] for i in range(windows)
    ], axis=0)
    smooth_kps = (smooth_kps * confidence_score).sum(0) / (confidence_score.sum(0) + 1e-8)
    smooth_kps[:, :, 2:] = confidence_score_org
    return smooth_kps


class SMPLHInstance(nn.Module):

    def __init__(self, betas, body_pose, global_orient, transl):
        super(SMPLHInstance, self).__init__()

        self.smpl = SMPLHLayer('/public/home/huochf/projects/3D_HOI/hoiYouTube/data/smpl/smplh/', gender='male')

        self.betas = nn.Parameter(betas)
        self.global_orient = nn.Parameter(matrix_to_axis_angle(global_orient))
        self.body_pose = nn.Parameter(matrix_to_axis_angle(body_pose))
        self.transl = nn.Parameter(transl)

        openpose_regressor = load_J_regressor('/public/home/huochf/projects/3D_HOI/hoiYouTube/data/smpl/J_regressor_body25_smplh.txt')
        self.register_buffer('openpose_regressor', torch.tensor(openpose_regressor).float())


    def get_optimizer(self, lr=1e-3):
        param_list = [self.betas, self.global_orient, self.body_pose, self.transl]
        optimizer = torch.optim.Adam(param_list, lr=lr, betas=(0.9, 0.999))
        return optimizer


    def forward(self, batch_idx, batch_size):
        global_orient_rotmat = axis_angle_to_matrix(self.global_orient[batch_idx: batch_idx + batch_size])
        body_pose_rotmat = axis_angle_to_matrix(self.body_pose[batch_idx: batch_idx + batch_size])
        smpl_out = self.smpl(betas=self.betas[batch_idx: batch_idx + batch_size], 
                               global_orient=global_orient_rotmat,
                               body_pose=body_pose_rotmat)

        smpl_joints = smpl_out.joints
        smpl_v = smpl_out.vertices
        smpl_v = smpl_v - smpl_joints[:, :1]
        smpl_v = smpl_v + self.transl.reshape(-1, 1, 3)[batch_idx: batch_idx + batch_size]
        smpl_joints = smpl_joints - smpl_joints[:, :1]
        smpl_joints = smpl_joints + self.transl.reshape(-1, 1, 3)[batch_idx: batch_idx + batch_size]

        openpose_kps3d = self.openpose_regressor.unsqueeze(0) @ smpl_v

        results = {
            'betas': self.betas[batch_idx: batch_idx + batch_size],
            'global_orient_rotmat': global_orient_rotmat,
            'body_pose_rotmat': body_pose_rotmat,
            'transl': self.transl[batch_idx: batch_idx + batch_size],
            'global_orient_axis_angle': self.global_orient[batch_idx: batch_idx + batch_size],
            'body_pose_axis_angle': self.body_pose[batch_idx: batch_idx + batch_size],
            'smpl_joints': smpl_joints,
            'openpose_kps3d': openpose_kps3d,
            'smpl_v': smpl_v,
        }
        return results


def load_vitpose(seq_name):
    kps_all = {}
    vitpose_dir = '/inspurfs/group/wangjingya/huochf/datasets_hot_data/BEHAVE_extend/vitpose/'
    vitpose_load = load_pickle(os.path.join(vitpose_dir, '{}.pkl'.format(seq_name)))
    for cam_id in vitpose_load:
        vitpose = vitpose_load[cam_id]
        kps = [item['kps'] for item in vitpose]
        if len(kps) == 0:
            kps_all[cam_id] = None
        else:
            kps_all[cam_id] = smooth_sequence(kps)

    return kps_all


class Keypoint2DLoss(nn.Module):

    def __init__(self, vitpose, cam_Ks):
        super().__init__()
        self.register_buffer('vitpose', torch.tensor(vitpose).float()) # [n_seq, -1, 3]
        self.openpose_to_wholebody_indices = [0, 16, 15, 18, 17, 5, 2, 6, 3, 7, 4, 12, 9, 13, 10, 14, 11, 19, 20, 21, 22, 23, 24]
        self.register_buffer('cam_Ks', cam_Ks)


    def project(self, points3d, batch_idx, batch_size):
        u = points3d[:, :, 0] / points3d[:, :, 2] * self.cam_Ks[batch_idx : batch_idx + batch_size, 0, 0].unsqueeze(1) + self.cam_Ks[batch_idx : batch_idx + batch_size, 0, 2].unsqueeze(1)
        v = points3d[:, :, 1] / points3d[:, :, 2] * self.cam_Ks[batch_idx : batch_idx + batch_size, 1, 1].unsqueeze(1) + self.cam_Ks[batch_idx : batch_idx + batch_size, 1, 2].unsqueeze(1)
        return torch.stack([u, v], dim=2)


    def forward(self, smpl_out, batch_idx, batch_size):
        smpl_kps = smpl_out['openpose_kps3d'] # [b, n, 3]

        smpl_kps_2d = self.project(smpl_kps, batch_idx, batch_size)
        loss_kps = F.l1_loss(smpl_kps_2d[:, self.openpose_to_wholebody_indices], self.vitpose[batch_idx:batch_idx+batch_size, :23, :2], reduction='none')
        loss_kps = loss_kps * self.vitpose[batch_idx:batch_idx+batch_size, :23, 2:]
        return {
            'loss_body_kps2d': loss_kps.mean()
        }


class SMPLPostPriorLoss(nn.Module):

    def __init__(self, prohmr, smpl_betas, visual_feats):
        super().__init__()
        self.flow = prohmr.flow
        self.smpl_betas = smpl_betas
        self.visual_feats = visual_feats


    def forward(self, smpl_out, batch_idx, batch_size):
        body_pose = smpl_out['body_pose_rotmat']
        b = body_pose.shape[0]
        body_pose = body_pose.reshape(b, -1, 3, 3)[:, :, :, :2].permute(0, 1, 3, 2)
        padding = torch.eye(3).to(body_pose.dtype).to(body_pose.device).reshape(1, 1, 3, 3)
        padding = padding.repeat(b, 2, 1, 1)[:, :, :, :2].permute(0, 1, 3, 2)
        body_pose = torch.cat([body_pose.reshape(b, 1, -1), padding.reshape(b, 1, -1)], dim=2)

        smpl_params = {
            'global_orient': smpl_out['global_orient_rotmat'].reshape(b, 1, 3, 3)[:, :, :, :2].permute(0, 1, 3, 2).reshape(b, 1, -1),
            'body_pose': body_pose,
        }
        log_prob, _ = self.flow.log_prob(smpl_params, self.visual_feats[batch_idx:batch_idx+batch_size])
        loss_beta = F.l1_loss(smpl_out['betas'], self.smpl_betas[batch_idx:batch_idx+batch_size])
        return {
            'smpl_prior': - log_prob.mean(),
            'beta_prior': loss_beta,
        }


class SmoothLoss(nn.Module):

    def __init__(self, ):
        super().__init__()


    def forward(self, smpl_out, batch_idx, batch_size):
        smpl_v = smpl_out['smpl_v']
        smpl_J = smpl_out['smpl_joints']

        if smpl_v.shape[0] > 1:
            loss_smooth_v = ((smpl_v[:-1] - smpl_v[1:]) ** 2).sum(-1).mean()
            loss_smooth_joints = ((smpl_J[:-1] - smpl_J[1:]) ** 2).sum(-1).mean()
        else:
            loss_smooth_v = torch.zeros(1, dtype=smpl_v.dtype, device=smpl_v.device)
            loss_smooth_joints = torch.zeros(1, dtype=smpl_v.dtype, device=smpl_v.device)

        return {
            'loss_smooth_v': loss_smooth_v,
            'loss_smooth_joints': loss_smooth_joints,
        }


def inference(args):
    device = torch.device('cuda')

    cfg = get_config(args.cfg_file)

    model = ProHMR.load_from_checkpoint(args.checkpoint, strict=False, cfg=cfg)
    model = model.to(device)
    model.eval()

    metadata = BEHAVEExtendMetaData(args.root_dir)
    img_ids_by_seq = metadata.get_all_image_by_sequence(split='all')
    output_dir = '/inspurfs/group/wangjingya/huochf/datasets_hot_data/BEHAVE_extend/prohmr'
    os.makedirs(output_dir, exist_ok=True)
    for seq_name in tqdm(img_ids_by_seq):
        if os.path.exists(os.path.join(output_dir, '{}.pkl'.format(seq_name))):
            continue
        if int(seq_name[5]) < args.begin_idx or int(seq_name[5]) >= args.end_idx:
            continue

        prohmr_all = {}
        vitpose_all = load_vitpose(seq_name)
        for cam_id in img_ids_by_seq[seq_name]:
            img_ids = img_ids_by_seq[seq_name][cam_id]
            if vitpose_all[cam_id] is None:
                print('lack annotation for {}'.format(seq_name))
                continue
            dataset = HumanImageDataset(cfg, seq_name, cam_id, img_ids, metadata)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, num_workers=8, shuffle=False, drop_last=False)

            smpl_betas, smpl_body_pose, smpl_global_orient, trans, condition_feats = [], [], [], [], []
            cam_Ks = []
            for item in tqdm(dataloader):
                images, bbox, cam_K = item
                images = images.to(device)
                bbox = bbox.to(device)
                cam_K =  cam_K.to(device)
                focal_length = torch.stack([cam_K[:, 0, 0], cam_K[:, 1, 1]], dim=1)
                batch = {'img': images, 'roi_center': bbox[:, :2], 'roi_size': bbox[:, 2:], 'K': cam_K, 'focal_length': focal_length}
                outputs = model.forward_step(batch)
                trans.append(outputs['global_cam_t'][:, 0].detach())
                smpl_betas.append(outputs['pred_smpl_params']['betas'][:, 0].detach()) # [b, 10]
                smpl_body_pose.append(outputs['pred_smpl_params']['body_pose'][:, 0, :21].detach()) # [b, 21, 3, 3]
                smpl_global_orient.append(outputs['pred_smpl_params']['global_orient'][:, 0].reshape(-1, 3, 3).detach()) # [b, 1, 3, 3]
                condition_feats.append(outputs['conditioning_feats'].detach()) # [b, n]
                cam_Ks.append(cam_K)
            smpl_betas = torch.cat(smpl_betas)
            smpl_body_pose = torch.cat(smpl_body_pose)
            smpl_global_orient = torch.cat(smpl_global_orient)
            trans = torch.cat(trans)
            condition_feats = torch.cat(condition_feats)
            cam_Ks = torch.cat(cam_Ks)

            seq_len = smpl_betas.shape[0]
            smplh_instance = SMPLHInstance(smpl_betas, smpl_body_pose, smpl_global_orient, trans).to(device)
            optimizer = smplh_instance.get_optimizer(lr=0.02)

            loss_functions = [
                Keypoint2DLoss(vitpose_all[cam_id], cam_Ks).to(device),
                SMPLPostPriorLoss(model, smpl_betas.clone(), condition_feats).to(device),
                SmoothLoss().to(device),
            ]
            iterations = 2
            steps_per_iter = 100
            batch_size = 64

            loss_weights = {
                'loss_body_kps2d': lambda cst, it: 10. ** -1 * cst / (1 + 10 * it),
                'smpl_prior': lambda cst, it: 10. ** 0 * cst / (1 + 10 * it),
                'beta_prior': lambda cst, it: 10. ** 0 * cst / (1 + 10 * it),
                'loss_smooth_v': lambda cst, it: 10. ** 0 * cst / (1 + 10 * it),
                'loss_smooth_joints': lambda cst, it: 10. ** 0 * cst / (1 + 10 * it),
            }
            for it in range(iterations):
                loop = tqdm(range(steps_per_iter))
                for i in loop:
                    optimizer.zero_grad()
                    total_loss = 0
                    for batch_idx in range(0, seq_len, batch_size // 2):
                        smpl_output = smplh_instance.forward(batch_idx, batch_size)
                        losses = {}
                        for f in loss_functions:
                            losses.update(f(smpl_output, batch_idx, batch_size))
                        loss_list = [loss_weights[k](v.mean(), it) for k, v in losses.items()]
                        total_loss += torch.stack(loss_list).sum()
                    total_loss.backward()
                    optimizer.step()

                    l_str = 'Optim. Step {}: Iter: {}'.format(it, i)
                    for k, v in losses.items():
                        l_str += ', {}: {:.4f}'.format(k, v.mean().detach().item())
                        loop.set_description(l_str)

            smpl_out = smplh_instance.forward(0, seq_len)

            prohmr_all[cam_id] = {
                'img_ids': img_ids,
                'betas': smpl_out['betas'].detach().cpu().numpy(),
                'global_orient_rotmat': smpl_out['global_orient_rotmat'].detach().cpu().numpy(),
                'body_pose_rotmat': smpl_out['body_pose_rotmat'].detach().cpu().numpy(),
                'transl': smpl_out['transl'].detach().cpu().numpy(),
                'cam_Ks': cam_Ks.detach().cpu().numpy(),
            } 
        save_pickle(prohmr_all, os.path.join(output_dir, '{}.pkl'.format(seq_name)))
        print('{} done!'.format(seq_name))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', default='/public/home/huochf/datasets/BEHAVE/', type=str, help='Dataset root directory.')
    parser.add_argument('--dataset', default='BEHAVE', type=str, choices=['BEHAVE', 'InterCap'], help='Process behave dataset or intercap dataset.')
    parser.add_argument('--checkpoint', default='outputs/prohmr/behave_extend/checkpoints/epoch=5-step=100000.ckpt', type=str, help='Directory to save logs and checkpoints')
    parser.add_argument('--cfg_file', default='prohmr_custom/configs/prohmr_behave_extend.yaml', type=str, help='Directory to save logs and checkpoints')
    parser.add_argument('--begin_idx', type=int)
    parser.add_argument('--end_idx', type=int)
    args = parser.parse_args()

    inference(args)
