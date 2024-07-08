import os
import pickle
import trimesh
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from smplx import SMPLHLayer, SMPLXLayer
from tqdm import tqdm

from pytorch3d.transforms import matrix_to_axis_angle, matrix_to_rotation_6d, rotation_6d_to_matrix, axis_angle_to_matrix
from hoi_recon.datasets.behave_extend_metadata import BEHAVEExtendMetaData
from hoi_recon.datasets.utils import load_pickle, save_pickle, save_json


def recon_from_offsets(offsets_recon, smpl, object_v_org, object_anchors_org, smpl_anchor_indices, object_anchor_indices):
    device = torch.device('cuda')
    batch_size = offsets_recon.shape[0]
    betas_recon = nn.Parameter(torch.zeros(batch_size, 10).float().to(device))
    body_pose_recon = nn.Parameter(torch.zeros(batch_size, 21, 3).float().to(device))
    object_rel_R6d_recon = nn.Parameter(matrix_to_rotation_6d(torch.eye(3).unsqueeze(0).repeat(batch_size, 1, 1)).float().to(device))
    object_rel_T_recon = nn.Parameter(torch.zeros(batch_size, 3).float().to(device))

    optimizer = torch.optim.Adam([betas_recon, body_pose_recon, object_rel_R6d_recon, object_rel_T_recon], lr=1e-2, betas=(0.9, 0.999))
    loss_weights = {
        'offset_loss': lambda cst, it: 10. ** 0 * cst,
        'beta_norm': lambda cst, it: 10 ** 0 * cst,
        'theta_norm': lambda cst, it: 10. ** 0 * cst,
    }
    iterations = 2
    steps_per_iter = 1000

    for it in range(iterations):
        loop = tqdm(range(steps_per_iter))
        for i in loop:
            optimizer.zero_grad()

            loss_beta_norm = (betas_recon ** 2).mean()
            loss_theta_norm = (body_pose_recon ** 2).mean()

            smpl_out = smpl(betas=betas_recon, body_pose=axis_angle_to_matrix(body_pose_recon))
            smpl_J = smpl_out.joints
            smpl_v = smpl_out.vertices
            smpl_v = smpl_v - smpl_J[:, :1]
            smpl_anchors = smpl_v[:, smpl_anchor_indices]
            object_v = object_v_org @ rotation_6d_to_matrix(object_rel_R6d_recon).transpose(2, 1) + object_rel_T_recon.unsqueeze(1)
            object_anchors = object_anchors_org @ rotation_6d_to_matrix(object_rel_R6d_recon).transpose(2, 1) + object_rel_T_recon.unsqueeze(1)
            offsets = object_anchors.reshape(batch_size, 1, -1, 3) - smpl_anchors.reshape(batch_size, -1, 1, 3)
            offsets = offsets.reshape(batch_size, -1)
            loss_offsets = F.l1_loss(offsets, offsets_recon)

            if it == 0:
                loss = loss_offsets + 1e-2 * loss_beta_norm + 1e-2 * loss_theta_norm
            else:
                loss = loss_offsets + 0 * loss_beta_norm + 0 * loss_theta_norm
                loss = loss * 0.1

            loss.backward()
            optimizer.step()

            l_str = 'Iter: {}, Step: {}'.format(it, i)
            l_str += ', {}: {:0.4f}'.format('loss_offsets', loss_offsets.item())
            l_str += ', {}: {:0.4f}'.format('loss_beta_norm', loss_beta_norm.item())
            l_str += ', {}: {:0.4f}'.format('loss_theta_norm', loss_theta_norm.item())
            loop.set_description(l_str)

    return betas_recon.detach(), axis_angle_to_matrix(body_pose_recon.detach()), rotation_6d_to_matrix(object_rel_R6d_recon).detach(), object_rel_T_recon.detach(), \
        smpl_v.detach(), object_v.detach()


class HOITemplateDataset:

    def __init__(self, annotation_file, obj_name):
        print('loading annotations ...')
        if isinstance(annotation_file, str):
            self.annotations = load_pickle(annotation_file)
        else:
            self.annotations = annotation_file

        self.annotations = [item for item in self.annotations if item['img_id'].split('_')[2] == obj_name]
        random.shuffle(self.annotations)
        self.annotations = self.annotations[:1000]


    def __len__(self, ):
        return len(self.annotations)


    def __getitem__(self, idx):
        annotation = self.annotations[idx]

        betas = annotation['smplh_betas_male']
        body_rotmat = annotation['smplh_pose_rotmat'][1:22]
        object_rel_rotmat = annotation['object_rel_rotmat']
        object_rel_trans = annotation['object_rel_trans']

        return betas, body_rotmat, object_rel_rotmat, object_rel_trans


def evaluate_pca_model(args):
    device = torch.device('cuda')
    annotation_file = 'data/datasets/behave_extend_test_list.pkl'
    annotation_file = load_pickle(annotation_file)

    output_dir = 'outputs/evaluate_pca_models/'
    os.makedirs(output_dir, exist_ok=True)

    dataset_metadata = BEHAVEExtendMetaData(args.root_dir)

    smpl = SMPLHLayer(model_path='data/models/smplh', gender='male', ext='pkl').to(device)
    pca_model_file = 'data/datasets/behave_extend_pca_models_n{}_{}_d{}.pkl'.format(args.smpl_anchor_num, args.object_anchor_num, args.pca_dim)
    pca_models = load_pickle(pca_model_file)

    error_all_objects = {}

    for obj_name in pca_models:
        error_all_objects[obj_name] = {
            'loss_betas': [], 'loss_body_pose': [], 'loss_obj_rel_rot': [], 'loss_obj_rel_T': [],
            'loss_smpl_v': [], 'loss_object_v': [], 'loss_offsets': [],
        }
        pca_model = pca_models[obj_name]
        mean = pca_model['mean']
        components = pca_model['components']
        smpl_anchor_indices = pca_model['smpl_anchor_indices']
        object_anchor_indices = pca_model['object_anchor_indices']
        print('evaluate pca model for object: {}'.format(obj_name))
        mean = torch.from_numpy(mean).float().reshape(1, -1).to(device)
        components = torch.from_numpy(components).float().unsqueeze(0).to(device)

        object_v_org = torch.from_numpy(dataset_metadata.obj_mesh_templates[obj_name][0]).float().to(device)
        object_v_org = object_v_org.reshape(1, -1, 3)
        object_anchors_org = object_v_org[:, object_anchor_indices]

        dataset = HOITemplateDataset(annotation_file, obj_name)
        if len(dataset) == 0:
            continue
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=128, num_workers=8, shuffle=True, drop_last=False)
        for batch in tqdm(dataloader):
            betas, body_rotmat, object_rel_rotmat, object_rel_trans = batch
            betas = betas.float().to(device)
            body_rotmat = body_rotmat.float().to(device)
            object_rel_rotmat = object_rel_rotmat.float().to(device)
            object_rel_trans = object_rel_trans.float().to(device)

            batch_size = betas.shape[0]

            smpl_out = smpl(betas=betas, body_pose=body_rotmat)
            smpl_J = smpl_out.joints.detach()
            smpl_v = smpl_out.vertices.detach()
            smpl_v = smpl_v - smpl_J[:, :1]
            smpl_anchors = smpl_v[:, smpl_anchor_indices]
            object_anchors = object_anchors_org @ object_rel_rotmat.transpose(2, 1) + object_rel_trans.reshape(-1, 1, 3)
            object_v = object_v_org @ object_rel_rotmat.transpose(2, 1) + object_rel_trans.reshape(-1, 1, 3)
            offsets = object_anchors.reshape(batch_size, 1, -1, 3) - smpl_anchors.reshape(batch_size, -1, 1, 3)
            offsets = offsets.reshape(batch_size, -1)

            gamma = (offsets - mean).unsqueeze(1) @ components.transpose(2, 1)
            gamma = gamma.reshape(batch_size, -1)
            offsets_recon = (gamma.unsqueeze(1) @ components).squeeze(1) + mean

            betas_recon, body_rotmat_recon, object_rel_rotmat_recon, object_rel_trans_recon, smpl_v_recon, object_v_recon = \
                recon_from_offsets(offsets_recon, smpl, object_v_org, object_anchors_org, smpl_anchor_indices, object_anchor_indices)

            loss_betas = F.l1_loss(betas_recon, betas)
            loss_body_pose = F.l1_loss(matrix_to_axis_angle(body_rotmat_recon), matrix_to_axis_angle(body_rotmat))
            loss_obj_rel_rot = F.l1_loss(matrix_to_axis_angle(object_rel_rotmat), matrix_to_axis_angle(object_rel_rotmat_recon))
            loss_obj_rel_T = F.l1_loss(object_rel_trans, object_rel_trans_recon)
            loss_smpl_v = F.l1_loss(smpl_v, smpl_v_recon)
            loss_object_v = F.l1_loss(object_v, object_v_recon)
            loss_offsets = F.l1_loss(offsets, offsets_recon)
            print()
            print('loss_betas: {:.4f}, loss_body_pose: {:.4f}, loss_obj_rel_rot: {:.4f}, loss_obj_rel_T: {:.4f}, loss_smpl_v: {:.4f}, loss_object_v: {:.4f}, loss_offsets: {:.4f}'.format(
                loss_betas.item(), loss_body_pose.item(), loss_obj_rel_rot.item(), loss_obj_rel_T.item(), loss_smpl_v.item(), loss_object_v.item(), loss_offsets.item(),))
            error_all_objects[obj_name]['loss_betas'].append(loss_betas.item())
            error_all_objects[obj_name]['loss_body_pose'].append(loss_body_pose.item())
            error_all_objects[obj_name]['loss_obj_rel_rot'].append(loss_obj_rel_rot.item())
            error_all_objects[obj_name]['loss_obj_rel_T'].append(loss_obj_rel_T.item())
            error_all_objects[obj_name]['loss_smpl_v'].append(loss_smpl_v.item())
            error_all_objects[obj_name]['loss_object_v'].append(loss_object_v.item())
            error_all_objects[obj_name]['loss_offsets'].append(loss_offsets.item())

        error_all_objects[obj_name]['loss_betas'] = np.mean(error_all_objects[obj_name]['loss_betas'])
        error_all_objects[obj_name]['loss_body_pose'] = np.mean(error_all_objects[obj_name]['loss_body_pose'])
        error_all_objects[obj_name]['loss_obj_rel_rot'] = np.mean(error_all_objects[obj_name]['loss_obj_rel_rot'])
        error_all_objects[obj_name]['loss_obj_rel_T'] = np.mean(error_all_objects[obj_name]['loss_obj_rel_T'])
        error_all_objects[obj_name]['loss_smpl_v'] = np.mean(error_all_objects[obj_name]['loss_smpl_v'])
        error_all_objects[obj_name]['loss_object_v'] = np.mean(error_all_objects[obj_name]['loss_object_v'])
        error_all_objects[obj_name]['loss_offsets'] = np.mean(error_all_objects[obj_name]['loss_offsets'])

    error_all_objects['avg'] = {}
    error_all_objects['avg']['loss_betas'] = np.mean([
        error_all_objects[obj_name]['loss_betas'] for obj_name in pca_models if obj_name not in ['keyboard', 'basketball']
    ])
    error_all_objects['avg']['loss_body_pose'] = np.mean([
        error_all_objects[obj_name]['loss_body_pose'] for obj_name in pca_models if obj_name not in ['keyboard', 'basketball']
    ])
    error_all_objects['avg']['loss_obj_rel_rot'] = np.mean([
        error_all_objects[obj_name]['loss_obj_rel_rot'] for obj_name in pca_models if obj_name not in ['keyboard', 'basketball']
    ])
    error_all_objects['avg']['loss_obj_rel_T'] = np.mean([
        error_all_objects[obj_name]['loss_obj_rel_T'] for obj_name in pca_models if obj_name not in ['keyboard', 'basketball']
    ])
    error_all_objects['avg']['loss_smpl_v'] = np.mean([
        error_all_objects[obj_name]['loss_smpl_v'] for obj_name in pca_models if obj_name not in ['keyboard', 'basketball']
    ])
    error_all_objects['avg']['loss_object_v'] = np.mean([
        error_all_objects[obj_name]['loss_object_v'] for obj_name in pca_models if obj_name not in ['keyboard', 'basketball']
    ])
    error_all_objects['avg']['loss_offsets'] = np.mean([
        error_all_objects[obj_name]['loss_offsets'] for obj_name in pca_models if obj_name not in ['keyboard', 'basketball']
    ])

    save_json(error_all_objects, os.path.join(output_dir, 'evaluate_pca_model_n{}_{}_d{}.json'.format(args.smpl_anchor_num, args.object_anchor_num, args.pca_dim)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', default='/public/home/huochf/datasets/BEHAVE/', type=str, help='Dataset root directory.')
    parser.add_argument('--smpl_anchor_num', default=32, type=int, help='the number of SMPL anchors per body part')
    parser.add_argument('--object_anchor_num', default=64, type=int, help='the number of object anchors')
    parser.add_argument('--pca_dim', default=32, type=int, help='the number of dimensions of PCA latent space')

    args = parser.parse_args()

    np.random.seed(17) # for reproducibility
    random.seed(17)

    evaluate_pca_model(args)
