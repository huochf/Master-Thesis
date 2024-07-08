import os
from tqdm import tqdm
import random
import numpy as np
import torch
import pickle
from scipy.spatial.transform import Rotation

from hoi_recon.datasets.utils import load_pickle, save_pickle


part_name2id = {
    'chair_head': 0,
    'chair_back': 1,
    'chair_arm_left': 2,
    'chair_arm_right': 3,
    'chair_seat': 4,
    'chair_base': 5,
    'footrest': 6,
}
train_object_ids = [17, 24, 25, 26, 30, 33, 44, 46, 48, 49, 
                    59, 60, 64, 68, 75, 81, 83, 85, 87, 96, 
                    98, 99, 103, 104, 110, 111, 116, 117, 118, 121, 
                    123, 130, 131, 141, 142, 143, 147, 151, 152, 154, 
                    156, 157, 158, 162, 166, 168, 171, 173, 176, 180, 
                    181]
test_object_ids = [15, 29, 36, 43, 45, 92, 109, 149, 172]
object_ids = train_object_ids + test_object_ids


def load_meshes():
    object_inter_mesh_dir = 'data/datasets/chairs/object_inter_shapes_2k_alpha_1e6'
    object_inter_meshes = {}
    for file in os.listdir(object_inter_mesh_dir):
        object_id = int(file.split('.')[0])
        object_inter_meshes[object_id] = np.load(os.path.join(object_inter_mesh_dir, file))

    return object_inter_meshes

device = torch.device('cuda')
anchor_indices = load_pickle('data/datasets/chairs_anchor_indices_n32_128.pkl')

object_inter_meshes = load_meshes()
annotations = load_pickle('data/datasets/chairs_train_annotations.pkl')
image_ids = list(annotations.keys())
random.seed(7)
random.shuffle(image_ids)


root_dir = '/storage/data/huochf/CHAIRS'

img_name = np.load(os.path.join(root_dir, 'AHOI_Data', 'DATA_FOLDER', 'img_name.npy'))
__object_ids = np.load(os.path.join(root_dir, 'AHOI_Data', 'DATA_FOLDER', 'object_id.npy'))
human_betas = np.load(os.path.join(root_dir, 'AHOI_Data', 'DATA_FOLDER', 'human_betas.npy'))
human_orient = np.load(os.path.join(root_dir, 'AHOI_Data', 'DATA_FOLDER', 'human_orient.npy'))
human_pose = np.load(os.path.join(root_dir, 'AHOI_Data', 'DATA_FOLDER', 'human_pose.npy'))
human_transl = np.load(os.path.join(root_dir, 'AHOI_Data', 'DATA_FOLDER', 'human_transl.npy'))
object_root_location = np.load(os.path.join(root_dir, 'AHOI_Data', 'DATA_FOLDER', 'object_root_location.npy'))
object_root_rotation = np.load(os.path.join(root_dir, 'AHOI_Data', 'DATA_FOLDER', 'object_root_rotation.npy'))

with open(os.path.join(root_dir, 'AHOI_Data', 'AHOI_ROOT', 'Metas', 'object_meta.pkl'), 'rb') as f:
    object_meda = pickle.load(f) 

item_indices = np.arange(len(object_root_location)).tolist()
random.shuffle(item_indices)

results = {}
for item_idx in tqdm(item_indices[:128]):
    _, seq_id, cam_id, frame_id = img_name[item_idx].split('/')[-1].split('.')[0].split('_')

    object_id = __object_ids[item_idx]
    img_id = '_'.join(['{:03d}'.format(object_id), seq_id, cam_id, frame_id])
    results[img_id] = {}

    smplx_betas = human_betas[item_idx]
    smplx_body_theta = human_pose[item_idx]
    object_rotation = object_root_rotation[item_idx]
    object_location = object_root_location[item_idx]

    object_rel_rotmat = Rotation.from_euler('xyz', [0, np.pi, 0]).as_matrix() @ \
               Rotation.from_euler('xyz', object_rotation).as_matrix()
    object_rel_trans =  Rotation.from_euler('xyz', [0, np.pi, 0]).as_matrix() @ object_location.reshape(3, 1)
    object_rel_trans = object_rel_trans.reshape(1, 3)

    results[img_id]['smplx_betas'] = smplx_betas
    results[img_id]['smplx_body_theta'] = smplx_body_theta
    results[img_id]['object_rel_rotmat'] = object_rel_rotmat
    results[img_id]['object_rel_trans'] = object_rel_trans
    object_anchors = {}
    _obj_anchors = object_inter_meshes[int(object_id)][anchor_indices['object']['sphere_2k']]
    _obj_anchors = _obj_anchors @ object_rel_rotmat.transpose(1, 0) + object_rel_trans.reshape(1, 3)
    object_anchors = _obj_anchors

    for _object_id in object_ids:
        Q = []
        P = []
        _object_anchors = object_inter_meshes[int(_object_id)][anchor_indices['object']['sphere_2k']]
        Q.append(_object_anchors)
        P.append(object_anchors)
        Q = np.concatenate(Q, axis=0)
        P = np.concatenate(P, axis=0)
        Q = torch.from_numpy(Q).float().to(device)
        P = torch.from_numpy(P).float().to(device)
        Q = Q.reshape(1, -1, 3)
        P = P.reshape(1, -1, 3)
        b = 1

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
        obj_rotmat_pred = obj_rotmat.float() # (b * n, 3, 3)
        _Q = Q + center_Q
        obj_trans_pred = (P.transpose(1, 2) - obj_rotmat_pred @ _Q.transpose(1, 2)).mean(dim=2) # (n * b, 3)
        object_rel_R = obj_rotmat_pred.reshape(3, 3)
        object_rel_T = obj_trans_pred.reshape(3)

        results[img_id][_object_id] = {'object_rel_R': object_rel_R.detach().cpu().numpy(),
            'object_rel_T': object_rel_T.detach().cpu().numpy()}
            
save_pickle(results, './interaction_transfer.pkl')
