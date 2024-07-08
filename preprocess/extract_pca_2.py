import os
import sys
file_dir = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(file_dir, '..', ))
import numpy as np
import random
import trimesh
import argparse
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F

from smplx import SMPLHLayer, SMPLXLayer
from sklearn.decomposition import PCA

from hoi_recon.datasets.behave_extend_metadata import BEHAVEExtendMetaData
from hoi_recon.datasets.utils import load_pickle, save_pickle


class HOOffsetDataList():

    def __init__(self, annotation_file, dataset_root, obj_name, args):
        print('loading annotations ...')
        if isinstance(annotation_file, str):
            annotations = load_pickle(annotation_file)
        else:
            annotations = annotation_file
        self.obj_name = obj_name

        self.dataset_metadata = BEHAVEExtendMetaData(dataset_root)
        self.annotations = [item for item in annotations if item['img_id'].split('_')[2] == obj_name]

        random.shuffle(self.annotations)
        self.annotations = self.annotations[:80000] 
        # self.annotations = self.annotations[:100]# for debug

        object_mesh = os.path.join(dataset_root, 'objects', obj_name, '{}_f1000.ply'.format(obj_name))
        if not os.path.exists(object_mesh):
            object_mesh = os.path.join(dataset_root, 'objects', obj_name, '{}_f2000.ply'.format(obj_name))
        if not os.path.exists(object_mesh):
            object_mesh = os.path.join(dataset_root, 'objects', obj_name, '{}_f2500.ply'.format(obj_name))
        if not os.path.exists(object_mesh):
            object_mesh = os.path.join(dataset_root, 'objects', obj_name, '{}_closed_f1000.ply'.format(obj_name))
        assert os.path.exists(object_mesh)

        object_mesh = trimesh.load(object_mesh, process=False)
        object_v_center = object_mesh.vertices.mean(0)
        object_v = trimesh.sample.sample_surface(object_mesh, 3000)[0]
        object_v = object_v - object_v_center
        self.object_v = object_v.astype(np.float32)


    def __len__(self, ):
        return len(self.annotations)


    def __getitem__(self, idx):
        annotation = self.annotations[idx]
        img_id = annotation['img_id']

        obj_name = img_id.split('_')[2]
        person_beta = annotation['smplh_betas_male'].astype(np.float32)
        person_body_pose = annotation['smplh_pose_rotmat'].astype(np.float32)[1:22]

        object_r = annotation['object_rel_rotmat']
        object_t = annotation['object_rel_trans']

        if np.isnan(person_beta).any():
            return self.__getitem__(np.random.randint(len(self)))

        return img_id, person_beta, person_body_pose, object_r, object_t, self.object_v


def sample_smpl_anchors(args):

    dataset_metadata = BEHAVEExtendMetaData(args.root_dir)
    smpl_pkl = load_pickle('data/models/smplh/SMPLH_MALE.pkl')
    radius = 0.02

    weights = smpl_pkl['weights']
    parts_indices = np.argmax(weights[:, :22], axis=1)

    smpl_anchor_indices = []
    for i in range(22):
        part_anchor_indices = np.where(parts_indices == i)[0]
        part_anchors = part_anchor_indices[np.random.choice(len(part_anchor_indices), args.smpl_anchor_num)]
        smpl_anchor_indices.extend(part_anchors.tolist())

    anchor_indices = {'smpl': smpl_anchor_indices}
    return anchor_indices



def extract_pca_models(args, anchor_indices):
    device = torch.device('cuda')

    dataset_metadata = BEHAVEExtendMetaData(args.root_dir)
    annotation_file = '../StackFLOW/data/datasets/behave_extend_train_list.pkl'
    annotation_file = load_pickle(annotation_file)

    smpl = SMPLHLayer(model_path='data/models/smplh', gender='male', ext='pkl').to(device)

    obj_names = list(dataset_metadata.OBJECT_NAME2IDX.keys())

    pca_models = {}
    for obj_name in obj_names:
        dataset = HOOffsetDataList(annotation_file, args.root_dir, obj_name, args)
        if len(dataset) == 0:
            continue
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, num_workers=4, shuffle=True)
        all_offsets = []
        print('collect offsets for object: {}'.format(obj_name))
        for batch in tqdm(dataloader):
            img_ids, person_beta, person_body_pose, object_r, object_t, object_v = batch

            b = person_beta.shape[0]

            person_beta = person_beta.to(device)
            person_body_pose = person_body_pose.to(device)
            object_r = object_r.to(device)
            object_t = object_t.to(device)
            object_v_org = object_v.to(device)

            smpl_out = smpl(betas=person_beta, body_pose=person_body_pose)
            smpl_v = smpl_out.vertices
            smpl_J = smpl_out.joints
            smpl_v = smpl_v - smpl_J[:, :1]

            object_v = object_v_org @ object_r.transpose(2, 1) + object_t.reshape(-1, 1, 3)
            smpl_anchors = smpl_v[:, anchor_indices['smpl']]

            ho_dists = ((smpl_anchors.unsqueeze(2) - object_v.unsqueeze(1)) ** 2).sum(-1)
            _, indices = ho_dists.min(dim=-1) # [b, n_smpl]
            object_anchors = torch.stack([v[idx] for v, idx in zip(object_v, indices)])
            object_anchors_org = torch.stack([v[idx] for v, idx in zip(object_v_org, indices)])

            ho_offsets = object_anchors - smpl_anchors
            ho_offsets = torch.cat([ho_offsets, object_anchors_org], dim=-1)

            for i in range(b):
                all_offsets.append(ho_offsets[i].reshape(-1).detach().cpu().numpy())

        pca = PCA(n_components=args.pca_dim)
        if len(all_offsets) == 0:
            print('no annotations for {} are found ...'.format(obj_name))
            n_offsets = args.smpl_anchor_num * 22 * 3 * 2
            pca_models[obj_name] = {
                'mean': np.zeros(n_offsets),
                'components': np.zeros((args.pca_dim, n_offsets)),
                'smpl_anchor_indices': anchor_indices['smpl'],
            }
        else:
            all_offsets = np.stack(all_offsets, axis=0)
            print('principle components analysing ...')
            pca.fit_transform(all_offsets)
            # print(pca.explained_variance_ratio_)
            # print(pca.explained_variance_)
            pca_models[obj_name] = {
                'mean': pca.mean_,
                'components': pca.components_,
                'smpl_anchor_indices': anchor_indices['smpl'],
            }
    return pca_models


def evaluate_pca_models(args, anchor_indices, pca_models):
    device = torch.device('cuda')

    dataset_metadata = BEHAVEExtendMetaData(args.root_dir)
    annotation_file = '../StackFLOW/data/datasets/behave_extend_test_list.pkl'
    annotation_file = load_pickle(annotation_file)
    smpl = SMPLHLayer(model_path='data/models/smplh', gender='male', ext='pkl').to(device)

    all_offsets = {}
    for obj_name in pca_models:
        pca_model = pca_models[obj_name]
        mean = torch.from_numpy(pca_model['mean']).float().reshape(1, -1).to(device)
        components = torch.from_numpy(pca_model['components']).float().to(device)
        print('evaluate pca model for object: {}'.format(obj_name))
        dataset = HOOffsetDataList(annotation_file, args.root_dir, obj_name, args)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, num_workers=4, shuffle=True)
        reconstruction_error = []
        for batch in tqdm(dataloader):
            img_id, person_beta, person_body_pose, object_r, object_t, object_v = batch

            b = person_beta.shape[0]

            person_beta = person_beta.to(device)
            person_body_pose = person_body_pose.to(device)
            object_r = object_r.to(device)
            object_t = object_t.to(device)
            object_v_org = object_v.to(device)

            smpl_out = smpl(betas=person_beta, body_pose=person_body_pose)
            smpl_v = smpl_out.vertices
            smpl_J = smpl_out.joints
            smpl_v = smpl_v - smpl_J[:, :1]

            object_v = object_v_org @ object_r.transpose(2, 1) + object_t.reshape(-1, 1, 3)

            smpl_anchors = smpl_v[:, anchor_indices['smpl']]

            ho_dists = ((smpl_anchors.unsqueeze(2) - object_v.unsqueeze(1)) ** 2).sum(-1)
            _, indices = ho_dists.min(dim=-1) # [b, n_smpl]
            object_anchors = torch.stack([v[idx] for v, idx in zip(object_v, indices)])
            object_anchors_org = torch.stack([v[idx] for v, idx in zip(object_v_org, indices)])

            ho_offsets = object_anchors - smpl_anchors
            ho_offsets = torch.cat([ho_offsets, object_anchors_org], dim=-1)

            ho_offsets = ho_offsets.reshape(b, -1)
            latent_code = torch.matmul(ho_offsets - mean, components.transpose(1, 0))
            recon_offset = torch.matmul(latent_code, components) + mean
            recon_error = (ho_offsets - recon_offset).abs().mean()
            reconstruction_error.append(recon_error.item())
        print('reconstruction error: {} +- {}'.format(np.mean(reconstruction_error), np.std(reconstruction_error)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', default='/public/home/huochf/datasets/BEHAVE/', type=str, help='Dataset root directory.')
    parser.add_argument('--smpl_anchor_num', default=32, type=int, help='the number of SMPL anchors per body part')
    parser.add_argument('--pca_dim', default=32, type=int, help='the number of dimensions of PCA latent space')

    args = parser.parse_args()

    np.random.seed(7) # for reproducibility
    random.seed(7)

    anchor_indices = sample_smpl_anchors(args)
    pca_models = extract_pca_models(args, anchor_indices)
    out_path = 'data/datasets/behave_extend_pca_models_v2_n{}_d{}.pkl'.format(args.smpl_anchor_num, args.pca_dim)
    save_pickle(pca_models, out_path)

    evaluate_pca_models(args, anchor_indices, pca_models)
