import os
import sys
file_dir = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(file_dir, '..', ))
import numpy as np
import argparse
import json
import cv2
import pickle
import trimesh
from trimesh.voxel import creation
import random
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
from smplx import SMPLXLayer
from pytorch3d.transforms import matrix_to_rotation_6d, rotation_6d_to_matrix

import torch
import torch.nn as nn
torch.multiprocessing.set_sharing_strategy('file_system')
from hoi_recon.configs.stackflow_chairs import load_config
from hoi_recon.models.stackflow_for_chairs import Model
# from hoi_recon.configs.stackflow_chairs_RT import load_config
# from hoi_recon.models.stackflow_for_chairs_RT import Model
from hoi_recon.utils.evaluator import chamfer_distance
from hoi_recon.datasets.utils import load_pickle, save_pickle, save_json, generate_image_patch, get_augmentation_params
from hoi_recon.datasets.chairs_hoi_dataset import train_object_ids, test_object_ids, part_name2id


def to_device(batch, device):
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(device)
    return batch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class ImageDataset:

    def __init__(self, root_dir, split='test'):
        self.root_dir = root_dir
        self.split = split
        self.image_padding_ratio = 0.5
        self.img_size = 256
        self.annotations = self.load_bboxes()
        self.img_ids = list(self.annotations.keys())
        self.vertex_len, self.object_meshes = self.load_meshes()
        self.img_ids = self.filter_image_ids()
        print('loaded {} items'.format(len(self.img_ids)))
        random.shuffle(self.img_ids)
        self.img_ids = self.img_ids[:5000]

        anchor_indices = load_pickle('data/datasets/chairs_anchor_indices_n32_128.pkl')
        self.object_indices = anchor_indices['object']['sphere_4k']
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        with open(root_dir + '/AHOI_Data/AHOI_ROOT/Metas/object_meta.pkl', 'rb') as f:
            object_metas = pickle.load(f)
        object_parts_init_shift = {}
        for object_id in test_object_ids:
            object_parts_init_shift[object_id] = []
            for part_name in part_name2id.keys():
                if part_name in object_metas[str(object_id)]['init_shift']:
                    object_parts_init_shift[object_id].append(
                        object_metas[str(object_id)]['init_shift'][part_name] - object_metas[str(object_id)]['init_shift']['chair_seat'])
                else:
                    object_parts_init_shift[object_id].append(np.zeros(3))
            object_parts_init_shift[object_id] = np.stack(object_parts_init_shift[object_id], axis=0)
        self.object_parts_init_shift = object_parts_init_shift


    def load_bboxes(self, ):
        if self.split == 'train':
            annotations = load_pickle('data/datasets/chairs_train_annotations.pkl')
        else:
            annotations = load_pickle('data/datasets/chairs_test_annotations.pkl')
        
        return annotations


    def filter_image_ids(self, ):
        img_ids = []
        for img_id in self.img_ids:
            _, seq_id, cam_id, frame_id = img_id.split('_')
            if seq_id in ['0151', '0152', '0153']:
                continue
            annotation = self.annotations[img_id]
            # print(img_id, np.linalg.norm(annotation['object_rel_trans'][4]))
            if np.linalg.norm(annotation['object_rel_trans'][4]) > 1:
                continue
            img_ids.append(img_id)
        return img_ids


    def load_meshes(self, ):
        object_info = np.load(os.path.join(self.root_dir, 'AHOI_Data', 'AHOI_ROOT', 'Metas', 'object_info.npy'), allow_pickle=True)
        vertex_len = object_info.item()['vertex_len']
        object_ids = object_info.item()['object_ids']
        vertex_len = {id_: len_ for id_, len_ in zip(object_ids, vertex_len)}

        object_inter_mesh_dir = 'data/datasets/chairs/object_inter_shapes_alpha_100_4k'
        object_inter_meshes = {}
        for file in os.listdir(object_inter_mesh_dir):
            if file.split('.')[-1] != 'npy':
                continue
            object_id = int(file.split('.')[0])
            object_inter_meshes[object_id] = np.load(os.path.join(object_inter_mesh_dir, file))

        return vertex_len, object_inter_meshes


    def __len__(self,):
        return len(self.img_ids)


    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        annotation = self.annotations[img_id]
        bbox = annotation['person_bb_xyxy']

        object_id, seq_id, cam_id, frame_id = img_id.split('_')

        image_path = os.path.join(self.root_dir, 'rimage', seq_id, cam_id, 'rgb_{}_{}_{}.jpg'.format(seq_id, cam_id, frame_id))
        if not os.path.exists(image_path):
            image_path = os.path.join(self.root_dir, 'rimage_extra', seq_id, cam_id, 'rgb_{}_{}_{}.jpg'.format(seq_id, cam_id, frame_id))
        assert os.path.exists(image_path), image_path
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)

        box_width, box_height = bbox[2:] - bbox[:2]
        box_size = max(box_width, box_height)
        box_size = box_size * (self.image_padding_ratio + 1)
        box_center_x, box_center_y = (bbox[2:] + bbox[:2]) / 2

        tx, ty, rot, scale, color_scale = 0., 0., 0., 1., [1., 1., 1.]

        box_center_x += tx * box_width
        box_center_y += ty * box_width
        out_size = self.img_size
        box_size = box_size * scale

        img_patch, img_trans = generate_image_patch(image, box_center_x, box_center_y, box_size, out_size, rot, color_scale)
        img_patch = img_patch[:, :, ::-1].astype(np.float32)
        img_patch = img_patch.transpose((2, 0, 1))
        img_patch = img_patch / 256

        for n_c in range(3):
            img_patch[n_c, :, :] = np.clip(img_patch[n_c, :, :] * color_scale[n_c], 0, 1)
            img_patch[n_c, :, :] = (img_patch[n_c, :, :] - self.mean[n_c]) / self.std[n_c]

        object_vertices_all_parts = self.object_meshes[int(object_id)]
        object_vertices_all_parts = object_vertices_all_parts[:, self.object_indices]

        results = {}
        results['img_id'] = img_id
        results['image'] = img_patch
        results['part_labels'] = 4
        results['object_anchors'] = object_vertices_all_parts[4].astype(np.float32)
        results['object_anchors_all_parts'] = object_vertices_all_parts.astype(np.float32)
        # results['smpler_betas'] = annotation['smpler_betas'].reshape(-1).astype(np.float32)
        # results['smpler_body_pose'] = annotation['smpler_body_pose'][:21].reshape(-1).astype(np.float32)
        results['smpler_betas'] = annotation['smplx_betas'].astype(np.float32) # annotation['smpler_betas'].reshape(-1).astype(np.float32)
        results['smpler_body_pose'] = annotation['smplx_body_theta'].astype(np.float32) # annotation['smpler_body_pose'][:21].reshape(-1).astype(np.float32)

        results['smpl_pose_rotmat'] = annotation['smplx_body_rotmat'].astype(np.float32)
        results['smpl_betas'] = annotation['smplx_betas'].astype(np.float32)
        results['object_rel_rotmat_all_parts'] = annotation['object_rel_rotmat'].astype(np.float32)
        results['object_rel_trans_all_parts'] = annotation['object_rel_trans'].astype(np.float32)
        results['object_parts_init_shift'] = self.object_parts_init_shift[int(object_id)].astype(np.float32)

        return results


def calculate_metrics(cfg, results_all):

    smpl = SMPLXLayer('data/models/smplx/', gender='male', use_pca=False, batch_size=1)
    smpl_f = np.array(smpl.faces).astype(np.int64)
    object_meshes_all = {}
    object_mesh_dir = os.path.join(cfg.dataset.root_dir, 'AHOI_Data', 'AHOI_ROOT', 'Meshes_wt')
    for object_id in test_object_ids:
        object_meshes_all[object_id] = {}
        for file in os.listdir(os.path.join(object_mesh_dir, str(object_id))):
            if file.split('.')[-1] != 'obj':
                continue
            if file.split('.')[0] not in part_name2id:
                continue
            part_id = part_name2id[file.split('.')[0]]
            object_mesh = trimesh.load(os.path.join(object_mesh_dir, str(object_id), file), process=False)
            object_meshes_all[object_id][part_id] = object_mesh

    sample_num = 5000

    metrics_all = {}
    print('calculating metrics ...')
    for item_id in tqdm(results_all):
        results = results_all[item_id]
        object_id, seq_id, cam_id, frame_id, part_id = item_id.split('_')

        gt_obj_rel_R = results['gt_obj_rel_R']
        gt_obj_rel_T = results['gt_obj_rel_T']
        gt_betas = results['gt_betas']
        gt_smpl_pose_rotmat = results['gt_smpl_pose_rotmat']
        gt_smpl_v = results['gt_smpl_v']

        pred_obj_rel_R = results['pred_obj_rel_R']
        pred_obj_rel_T = results['pred_obj_rel_T']
        pred_betas = results['pred_betas']
        pred_smpl_pose_rotmat = results['pred_smpl_pose_rotmat']
        pred_smpl_v = results['pred_smpl_v']

        smpl_mesh_recon = trimesh.Trimesh(pred_smpl_v, smpl_f)
        smpl_mesh_gt = trimesh.Trimesh(gt_smpl_v, smpl_f)

        error_trans = np.sqrt(((gt_obj_rel_T - pred_obj_rel_T) ** 2).sum())
        error_rot = np.arccos((np.trace(gt_obj_rel_R @ pred_obj_rel_R.T) - 1) / 2)
        error_rot = error_rot * 180 / np.pi

        object_v = object_meshes_all[int(object_id)][int(part_id)].vertices
        object_f = object_meshes_all[int(object_id)][int(part_id)].faces
        object_v_recon = object_v @ pred_obj_rel_R.T + pred_obj_rel_T.reshape(1, 3)
        object_v_gt = object_v @ gt_obj_rel_R.T + gt_obj_rel_T.reshape(1, 3)
        object_mesh_recon = trimesh.Trimesh(object_v_recon, object_f)
        object_mesh_gt = trimesh.Trimesh(object_v_gt, object_f)

        recon_smpl_points = smpl_mesh_recon.sample(sample_num)
        recon_object_points = object_mesh_recon.sample(sample_num)
        gt_smpl_points = smpl_mesh_gt.sample(sample_num)
        gt_object_points = object_mesh_gt.sample(sample_num)
        chamfer_smpl_dist = chamfer_distance(recon_smpl_points, gt_smpl_points)
        chamfer_object_dist = chamfer_distance(recon_object_points, gt_object_points)

        recon_object_points = object_mesh_recon.sample(1000) # to speed up
        signed_distance = trimesh.proximity.signed_distance(smpl_mesh_recon, recon_object_points)

        contact_error = min(np.abs(signed_distance).min(), 0.2)
        penetration_error = max(signed_distance.max(), 0)

        # object_v_combined = np.concatenate([object_v_gt, object_v_recon], axis=0)
        # v_min = object_v_combined.min(axis=0)
        # v_max = object_v_combined.max(axis=0)
        # center = (v_max + v_min) / 2
        # radius = (v_max - v_min).max() / 2
        # object_mesh_combined = trimesh.Trimesh(object_v_combined, np.concatenate([object_f, object_f + object_v_gt.shape[0]], axis=0))
        # union_voxels = creation.local_voxelize(mesh=object_mesh_combined, point=center, pitch=radius / 64, radius=64, fill=True)
        # voxel_single = creation.local_voxelize(mesh=object_mesh_gt, point=center, pitch=radius / 64, radius=64, fill=True)
        mesh_iou = 0 # (2 * voxel_single.volume - union_voxels.volume) / union_voxels.volume

        metrics_all[item_id] = {
            'error_rot': float(error_rot),
            'error_trans': float(error_trans),
            'chamfer_object_dist': float(chamfer_object_dist),
            'chamfer_smpl_dist': float(chamfer_smpl_dist),
            'contact_error': float(contact_error),
            'penetration_error': float(penetration_error),
            'mesh_iou': float(mesh_iou),
        }
        print(item_id, metrics_all[item_id])

    avg_metrics = {
        'error_rot': float(np.mean([metrics_all[id_]['error_rot'] for id_ in metrics_all])),
        'error_trans': float(np.mean([metrics_all[id_]['error_trans'] for id_ in metrics_all])),
        'chamfer_object_dist': float(np.mean([metrics_all[id_]['chamfer_object_dist'] for id_ in metrics_all])),
        'chamfer_smpl_dist': float(np.mean([metrics_all[id_]['chamfer_smpl_dist'] for id_ in metrics_all])),
        'contact_error': float(np.mean([metrics_all[id_]['contact_error'] for id_ in metrics_all])),
        'penetration_error': float(np.mean([metrics_all[id_]['penetration_error'] for id_ in metrics_all])),
        'mesh_iou': float(np.mean([metrics_all[id_]['mesh_iou'] for id_ in metrics_all])),
    }
    metrics_all['avg'] = avg_metrics
    print(metrics_all['avg'])

    return metrics_all


def tune_object_pose(model, batch, outputs, valid_parts):
    pred_obj_rel_R = outputs['pred_obj_rel_R'].detach()
    pred_obj_rel_T = outputs['pred_obj_rel_T'].detach()

    batch_size = pred_obj_rel_T.shape[0]

    visual_feats = outputs['visual_features'].detach().reshape(batch_size, 1, -1).repeat(1, 7, 1).reshape(batch_size * 7, -1)
    smpl_betas = outputs['pred_betas'].detach().reshape(batch_size, 1, 10).repeat(1, 7, 1).reshape(batch_size * 7, 10)
    smpl_body_pose_rotmat = outputs['pred_smpl_body_pose'].detach().reshape(batch_size, 1, 21, 3, 3).repeat(1, 7, 1, 1, 1).reshape(batch_size * 7, 21, 3, 3)
    smpl_body_pose_rot6d = matrix_to_rotation_6d(smpl_body_pose_rotmat).reshape(batch_size * 7, 21 * 6)
    object_anchors_all_parts = batch['object_anchors_all_parts'].reshape(batch_size * 7, -1, 3)
    parts_labels = torch.arange(7).reshape(1, 7).repeat(batch_size, 1).reshape(batch_size * 7).to(torch.int64).to(smpl_betas.device)

    pred_obj_rel_R6d = matrix_to_rotation_6d(pred_obj_rel_R).reshape(batch_size, 6)
    pred_obj_rel_T = pred_obj_rel_T.reshape(batch_size, 3)
    pred_obj_rel_R6d = nn.Parameter(pred_obj_rel_R6d)
    pred_obj_rel_T = nn.Parameter(pred_obj_rel_T)

    optimizer = torch.optim.Adam([pred_obj_rel_R6d, pred_obj_rel_T], lr=0.01, betas=(0.9, 0.999))
    iterations = 0
    steps_per_iter = 200
    for it in range(iterations):
        loop = tqdm(range(steps_per_iter))

        for i in loop:
            optimizer.zero_grad()

            object_rel_R_all_parts = rotation_6d_to_matrix(pred_obj_rel_R6d).reshape(batch_size, 1, 3, 3).repeat(1, 7, 1, 1)
            object_rel_R_all_parts = object_rel_R_all_parts.reshape(batch_size * 7, 3, 3)
            init_shift = object_rel_R_all_parts.reshape(batch_size * 7, 3, 3) @ batch['object_parts_init_shift'].reshape(batch_size * 7, 3, 1)
            object_rel_T_all_parts = pred_obj_rel_T.reshape(batch_size, 1, 3).repeat(1, 7, 1) + init_shift.reshape(batch_size, 7, 3)
            object_rel_T_all_parts = object_rel_T_all_parts.reshape(batch_size * 7, 3)
            pred_gamma, _ = model.flow_loss.hooffset.encode(
                smpl_betas, smpl_body_pose_rotmat, object_rel_R_all_parts, object_rel_T_all_parts, object_anchors_all_parts, parts_labels)
            log_prob = model.stackflow.log_prob(pred_gamma, visual_feats, smpl_body_pose_rot6d, parts_labels)
            loss_nll = - (log_prob.reshape(-1) * valid_parts.reshape(-1)).sum() / valid_parts.sum()
            loss_nll = loss_nll / 10**it
            loss_nll.backward()
            optimizer.step()

            l_str = 'Optim. Step {}: Iter: {}, loss: {:.4f}'.format(it, i, loss_nll.item())
            loop.set_description(l_str)

    object_rel_R_all_parts = rotation_6d_to_matrix(pred_obj_rel_R6d).reshape(batch_size, 1, 3, 3).repeat(1, 7, 1, 1)
    init_shift = object_rel_R_all_parts.reshape(batch_size * 7, 3, 3) @ batch['object_parts_init_shift'].reshape(batch_size * 7, 3, 1)
    object_rel_T_all_parts = pred_obj_rel_T.reshape(batch_size, 1, 3).repeat(1, 7, 1) + init_shift.reshape(batch_size, 7, 3)
    return object_rel_R_all_parts, object_rel_T_all_parts


def evaluate(cfg):
    # device = torch.device('cuda')

    # dataset_root_dir = cfg.dataset.root_dir
    # dataset = ImageDataset(dataset_root_dir, split='test')
    # dataloader = torch.utils.data.DataLoader(dataset, 
    #                                                batch_size=cfg.train.batch_size,
    #                                                num_workers=cfg.train.num_workers,
    #                                                shuffle=False,
    #                                                drop_last=False)

    # model = Model(cfg)
    # model.to(device)
    # model.eval()
    # model.load_checkpoint(cfg.train.resume)

    # smpl = SMPLXLayer('data/models/smplx/', gender='male', use_pca=False, batch_size=cfg.train.batch_size).to(device)

    # results_all = {}
    # for idx, batch in enumerate(tqdm(dataloader)):
    #     batch = to_device(batch, device)
    #     outputs = model.inference(batch)

    #     smpl_out_pred = smpl(betas=outputs['pred_betas'], body_pose=outputs['pred_smpl_body_pose'])
    #     smpl_J_pred = smpl_out_pred.joints
    #     smpl_v_pred = smpl_out_pred.vertices
    #     smpl_v_pred = smpl_v_pred - smpl_J_pred[:, :1]

    #     smpl_out_gt = smpl(betas=batch['smpl_betas'], body_pose=batch['smpl_pose_rotmat'])
    #     smpl_J_gt = smpl_out_gt.joints
    #     smpl_v_gt = smpl_out_gt.vertices
    #     smpl_v_gt = smpl_v_gt - smpl_J_gt[:, :1]

    #     batch_size = smpl_v_gt.shape[0]
    #     valid_parts = torch.zeros(batch_size, 7).float().to(device)
    #     # valid_parts[:, 4] = 1
    #     for batch_idx in range(batch_size):
    #         for part_idx in range(7):
    #             if not torch.isnan(batch['object_rel_rotmat_all_parts'][batch_idx, part_idx]).any():
    #                 valid_parts[batch_idx, part_idx] = 1
    #     object_rel_R_all_parts, object_rel_T_all_parts = tune_object_pose(model, batch, outputs, valid_parts)

    #     for idx, img_id in enumerate(batch['img_id']):
    #         for i in range(7):
    #             if valid_parts[idx, i] <= 0:
    #                 continue
    #             item_id = img_id + '_' + str(i)

    #             results_all[item_id] = {
    #                 'pred_betas': outputs['pred_betas'][idx].detach().cpu().numpy(),
    #                 'pred_pose6d': outputs['pred_pose6d'][idx].detach().cpu().numpy(),
    #                 'pred_smpl_pose_rotmat': outputs['pred_smpl_body_pose'][idx].detach().cpu().numpy(),
    #                 'pred_obj_rel_R': object_rel_R_all_parts[idx, i].detach().cpu().numpy(),
    #                 'pred_obj_rel_T': object_rel_T_all_parts[idx, i].detach().cpu().numpy(),
    #                 'pred_smpl_v': smpl_v_pred[idx].detach().cpu().numpy(),
    #                 'gt_betas': batch['smpl_betas'][idx].detach().cpu().numpy(),
    #                 'gt_smpl_pose_rotmat': batch['smpl_pose_rotmat'][idx].detach().cpu().numpy(),
    #                 'gt_obj_rel_R': batch['object_rel_rotmat_all_parts'][idx, i].detach().cpu().numpy(),
    #                 'gt_obj_rel_T': batch['object_rel_trans_all_parts'][idx, i].detach().cpu().numpy(),
    #                 'gt_smpl_v': smpl_v_gt[idx].detach().cpu().numpy(),
    #             }
    #     # if len(results_all) > 10:
    #     #     break

    # os.makedirs(cfg.train.output_dir, exist_ok=True)
    # save_pickle(results_all, os.path.join(cfg.train.output_dir, 'inference_results_sit.pkl'))

    results_all = load_pickle(os.path.join(cfg.train.output_dir, 'inference_results_sit.pkl'))
    evaluate_metrics = calculate_metrics(cfg, results_all)
    save_json(evaluate_metrics, os.path.join(cfg.train.output_dir, 'evaluate_metrics_sit.json'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root_dir', default='/storage/data/huochf/CHAIRS', type=str)
    args = parser.parse_args()

    cfg = load_config()
    cfg.dataset.root_dir = args.dataset_root_dir
    cfg.freeze()
    set_seed(7)
    evaluate(cfg)
