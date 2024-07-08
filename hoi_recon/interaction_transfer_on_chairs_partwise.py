import os
import random
import numpy as np
import torch
import pickle

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
    root_dir = '/storage/data/huochf/CHAIRS'
    object_info = np.load(os.path.join(root_dir, 'AHOI_Data', 'AHOI_ROOT', 'Metas', 'object_info.npy'), allow_pickle=True)
    vertex_len = object_info.item()['vertex_len']
    object_ids = object_info.item()['object_ids']
    vertex_len = {id_: len_ for id_, len_ in zip(object_ids, vertex_len)}

    object_inter_mesh_dir = 'data/datasets/chairs/object_inter_shapes'
    object_inter_meshes = {}
    for file in os.listdir(object_inter_mesh_dir):
        object_id = int(file.split('.')[0])
        object_inter_meshes[object_id] = np.load(os.path.join(object_inter_mesh_dir, file))

    return vertex_len, object_inter_meshes

device = torch.device('cuda')
anchor_indices = load_pickle('data/datasets/chairs_anchor_indices_n32_128.pkl')

vertex_len, object_inter_meshes = load_meshes()
annotations = load_pickle('data/datasets/chairs_train_annotations.pkl')
image_ids = list(annotations.keys())
random.seed(7)
random.shuffle(image_ids)

with open('/storage/data/huochf/CHAIRS/AHOI_Data/AHOI_ROOT/Metas/object_meta.pkl', 'rb') as f:
    object_metas = pickle.load(f)
object_parts_init_shift = {}
for object_id in object_ids:
    object_parts_init_shift[object_id] = []
    for part_name in part_name2id.keys():
        if part_name in object_metas[str(object_id)]['init_shift']:
            object_parts_init_shift[object_id].append(object_metas[str(object_id)]['init_shift'][part_name])
        else:
            object_parts_init_shift[object_id].append(np.zeros(3))
    object_parts_init_shift[object_id] = np.stack(object_parts_init_shift[object_id], axis=0)

results = {}
for img_id in image_ids[:128]:
    results[img_id] = {}
    object_id, seq_id, cam_id, frame_id = img_id.split('_')

    annotation = annotations[img_id]
    smplx_betas = annotation['smplx_betas']
    smplx_body_theta = annotation['smplx_body_theta']
    smplx_body_rotmat = annotation['smplx_body_rotmat']
    object_rel_rotmat = annotation['object_rel_rotmat']
    object_rel_trans = annotation['object_rel_trans']

    results[img_id]['smplx_betas'] = smplx_betas
    results[img_id]['smplx_body_theta'] = smplx_body_theta
    results[img_id]['object_rel_rotmat'] = object_rel_rotmat
    results[img_id]['object_rel_trans'] = object_rel_trans
    object_anchors = {}
    for part_id in np.nonzero(vertex_len[int(object_id)])[0]:
        if not np.isnan(object_rel_rotmat[part_id]).any():
            _obj_anchors = object_inter_meshes[int(object_id)][part_id][anchor_indices['object']['sphere_1k']]
            _obj_anchors = _obj_anchors @ object_rel_rotmat[part_id].transpose(1, 0) + object_rel_trans[part_id].reshape(1, 3)
            object_anchors[part_id] = _obj_anchors
        else:
            object_anchors[part_id] = None

    for _object_id in object_ids:
        Q = []
        P = []
        for part_id in object_anchors:
            if object_anchors[part_id] is None:
                continue
            if vertex_len[int(_object_id)][part_id] <= 0:
                continue
            init_shift = object_parts_init_shift[_object_id][part_id] - object_parts_init_shift[_object_id][4]
            _object_anchors = object_inter_meshes[int(_object_id)][part_id][anchor_indices['object']['sphere_2k']]
            Q.append(_object_anchors + init_shift)
            P.append(object_anchors[part_id])
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
