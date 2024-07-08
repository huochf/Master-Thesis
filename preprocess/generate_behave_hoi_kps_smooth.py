import os
import sys
file_dir = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(file_dir, '..', ))
import json
from tqdm import tqdm
import argparse
import numpy as np
import torch
from smplx import SMPLH, SMPLX, SMPLHLayer, SMPLXLayer
from pytorch3d.transforms import axis_angle_to_matrix
from scipy.spatial.transform import Rotation as R

from hoi_recon.datasets.behave_extend_metadata import BEHAVEExtendMetaData
from hoi_recon.datasets.utils import save_pickle, load_pickle


def smooth_sequence(sequence, windows=5):
    sequence = np.stack(sequence, axis=0)
    n, d = sequence.shape

    sequence = np.concatenate([np.zeros((windows // 2, d)), sequence, np.zeros((windows // 2, d))], axis=0) # [seq_n + windows - 1, n, 3]
    confidence_score = np.ones([n + windows // 2 * 2, 1])
    confidence_score[:windows // 2] = 0
    confidence_score[- windows // 2:] = 0
    smooth_kps = np.stack([
        sequence[i: n + i, :] for i in range(windows)
    ], axis=0)    
    confidence_score = np.stack([
        confidence_score[i: n + i, :] for i in range(windows)
    ], axis=0)
    smooth_kps = (smooth_kps * confidence_score).sum(0) / (confidence_score.sum(0) + 1e-8)
    return smooth_kps


def generate_kps_behave_extend(args):
    device = torch.device('cuda')
    metadata = BEHAVEExtendMetaData(args.root_dir)

    joints_flip = [0, 2, 1, 3, 5, 4, 6, 8, 7, 9, 11, 10, 12, 14, 13, 15, 17, 16, 19, 18, 21, 20]

    smpl = SMPLH(model_path='data/models/smplh', gender='male').to(device)
    for sequence_name in metadata.go_through_all_sequences(split=args.split):
        day_id, sub_id, obj_name, inter_type = metadata.parse_seq_info(sequence_name)
        if args.object != 'all' and obj_name != args.object:
            continue

        object_v = torch.tensor(metadata.obj_mesh_templates[obj_name][0]).reshape(1, -1, 3).float().to(device)
        obj_keypoints_3d = metadata.load_object_keypoints(obj_name)
        obj_keypoints_3d = torch.tensor(obj_keypoints_3d).reshape(1, -1, 3).float().to(device)
        # output_dir = os.path.join(args.root_dir, 'hoi_vertices', sequence_name)
        output_dir = os.path.join('/inspurfs/group/wangjingya/huochf/datasets_hot_data/BEHAVE_extend/', 'hoi_vertices_smoothed', sequence_name)
        os.makedirs(output_dir, exist_ok=True)

        print('Processing sequence: {}.'.format(sequence_name))
        annotations = load_pickle('./data/datasets/behave_extend_datalist/{}.pkl'.format(sequence_name))

        if len(annotations[0]) == 0:
            continue
        smplh_betas_male_seq = smooth_sequence([item['smplh_betas_male'] for item in annotations[0]])
        smplh_theta_seq = smooth_sequence([item['smplh_theta'] for item in annotations[0]])

        object_rel_rotaxis = smooth_sequence([R.from_matrix(item['object_rel_rotmat']).as_quat() for item in annotations[0]])
        object_rel_trans_seq = smooth_sequence([item['object_rel_trans'] for item in annotations[0]])
        object_rel_rotmat_seq = [R.from_quat(item).as_matrix() for item in object_rel_rotaxis]
        # for annotation in tqdm(annotations[0]): # cam_id = 0
        for idx, _ in enumerate(tqdm(annotations[0])):
            img_id = annotations[0][idx]['img_id']
            if os.path.exists(os.path.join(output_dir, '{}.pkl'.format(img_id))) and not args.redo:
                continue

            smplh_betas = torch.tensor(smplh_betas_male_seq[idx]).unsqueeze(0).float().to(device)
            smplh_pose = torch.tensor(smplh_theta_seq[idx]).unsqueeze(0).float().to(device)

            smpl_out_org = smpl(betas=smplh_betas, body_pose=smplh_pose[:, 3:66])
            joint_3d_org = smpl_out_org.joints.detach()
            smpl_v_org = smpl_out_org.vertices.detach()
            smpl_v_org = smpl_v_org - joint_3d_org[:, :1]
            joint_3d_org = joint_3d_org - joint_3d_org[:, :1]

            smplh_pose_sym = smplh_pose[:, :66].reshape(1, 22, 3)[:, joints_flip]
            smplh_pose_sym[:, :, 1] *= -1
            smplh_pose_sym[:, :, 2] *= -1
            smplh_pose_sym = smplh_pose_sym.reshape(1, 66)
            smpl_out_org_sym = smpl(betas=smplh_betas, body_pose=smplh_pose_sym[:, 3:66])
            joint_3d_org_sym = smpl_out_org_sym.joints.detach()
            smpl_v_org_sym = smpl_out_org_sym.vertices.detach()
            smpl_v_org_sym = smpl_v_org_sym - joint_3d_org_sym[:, :1]
            joint_3d_org_sym = joint_3d_org_sym - joint_3d_org_sym[:, :1]

            object_rel_rotmat = torch.tensor(object_rel_rotmat_seq[idx]).unsqueeze(0).float().to(device)
            object_rel_trans = torch.tensor(object_rel_trans_seq[idx]).unsqueeze(0).float().to(device)

            object_v_org = torch.matmul(object_v, object_rel_rotmat.transpose(2, 1)) + object_rel_trans.reshape(1, 1, 3)
            obj_kps_org = torch.matmul(obj_keypoints_3d, object_rel_rotmat.transpose(2, 1)) + object_rel_trans.reshape(1, 1, 3)

            save_path = os.path.join(output_dir, '{}.pkl'.format(img_id))
            results = {
                'smpl_kps': joint_3d_org[0, :22].detach().cpu().numpy(),
                'smpl_kps_sym': joint_3d_org_sym[0, :22].detach().cpu().numpy(),
                'object_kps': obj_kps_org[0].detach().cpu().numpy(),
                'smpl_v': smpl_v_org[0].detach().cpu().numpy(),
                'smpl_v_sym': smpl_v_org_sym[0].detach().cpu().numpy(),
                'object_v': object_v_org[0].detach().cpu().numpy(),
                'object_rel_rotmat': object_rel_rotmat[0].detach().cpu().numpy(),
                'object_rel_trans': object_rel_trans[0].detach().cpu().numpy(),
                'smpl_orient': smplh_pose[0, :3].detach().cpu().numpy(),
                'smpl_orient_sym': smplh_pose_sym[0, :3].detach().cpu().numpy(),
            }
            save_pickle(results, save_path)
        print('Sequence {} Done!'.format(sequence_name))


def generate_kps(args):
    generate_kps_behave_extend(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate KPS (BEHAVE)')
    parser.add_argument('--root_dir', default='/storage/data/huochf/BEHAVE', type=str)
    parser.add_argument('--object', default='backpack', type=str)
    parser.add_argument('--split', default='train', type=str)
    parser.add_argument('--redo', default=False, action='store_true')
    args = parser.parse_args()

    generate_kps(args)
