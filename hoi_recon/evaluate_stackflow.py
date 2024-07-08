import os
import argparse
import numpy as np
import cv2
import trimesh
from tqdm import tqdm
import random
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

from smplx import SMPLHLayer
from pytorch3d.transforms import matrix_to_rotation_6d, axis_angle_to_matrix, quaternion_to_matrix

from hoi_recon.datasets.utils import generate_image_patch
from hoi_recon.datasets.behave_extend_metadata import BEHAVEExtendMetaData
from hoi_recon.datasets.utils import load_json, load_pickle, save_pickle, save_json
from hoi_recon.configs.stackflow_config import load_config
from hoi_recon.models.stackflow import Model
from hoi_recon.models.hoi_instance import HOIInstance
from hoi_recon.utils.evaluator import get_recon_meshes, get_gt_meshes, ReconEvaluator
from hoi_recon.utils.post_optimization import post_optimization
from hoi_recon.utils.optim_losses import SMPLPostPriorLoss, SMPLKpsProjLoss, ObjectEproPnpLoss, HOOffsetLoss


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def extract_bbox_from_mask(mask):
    try:
        indices = np.array(np.nonzero(np.array(mask)))
        y1 = np.min(indices[0, :])
        y2 = np.max(indices[0, :])
        x1 = np.min(indices[1, :])
        x2 = np.max(indices[1, :])

        return np.array([x1, y1, x2, y2])
    except:
        return np.zeros(4)


class TestDataset:

    def __init__(self, metadata, img_ids):
        self.metadata = metadata
        self.img_ids = img_ids
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])
        self.annotations = self.load_annotations(train=False)


    def load_annotations(self, train):
        sequence_names = list(self.metadata.go_through_all_sequences(split='train' if train else 'test'))
        all_annotations = {}
        for seq_name in tqdm(sequence_names):
            annotations = load_pickle('./data/datasets/behave_extend_datalist/{}.pkl'.format(seq_name))
            for cam_id in annotations:
                for item in annotations[cam_id]:
                    img_id = item['img_id']
                    all_annotations[img_id] = {
                        'person_bb_xyxy': item['person_bb_xyxy'],
                        'hoi_bb_xyxy': item['hoi_bb_xyxy'],
                        'smplh_theta': item['smplh_theta'],
                        'smplh_betas_male': item['smplh_betas_male'],
                        'smplh_joints_2d': item['smplh_joints_2d'],
                        'smplh_joints_3d': item['smplh_joints_3d'],
                        'cam_K': item['cam_K'],
                        'hoi_trans': item['hoi_trans'],
                    }
        return all_annotations


    def __len__(self, ):
        return len(self.img_ids)


    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        day_id, sub_id, obj_name, inter_type, frame_id, cam_id = img_id.split('_')
        object_label = self.metadata.OBJECT_NAME2IDX[obj_name]

        image = cv2.imread(self.metadata.get_image_path(img_id), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)

        fx, fy, cx, cy = self.metadata.cam_intrinsics[int(cam_id)][:4]
        cam_K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]).astype(np.float32)

        annotation = self.annotations[img_id]
        hoi_bbox = annotation['hoi_bb_xyxy']
        x1, y1, x2, y2 = hoi_bbox
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        box_size = max(1, 1.2 * max(x2 - x1, y2 - y1))
        hoi_image, _ = generate_image_patch(image, cx, cy, box_size, 256, 0, [1., 1., 1.])
        hoi_image = hoi_image[:, :, ::-1].astype(np.float32) # convert to RGB
        hoi_image = hoi_image.transpose((2, 0, 1))
        hoi_image = hoi_image / 256
        for n_c in range(3):
            hoi_image[n_c, :, :] = np.clip(hoi_image[n_c, :, :], 0, 1)
            hoi_image[n_c, :, :] = (hoi_image[n_c, :, :] - self.mean[n_c]) / self.std[n_c]

        crop_bbox = np.array([cx, cy, box_size, box_size]).astype(np.float32)

        # load vitpose here
        vitpose_load = load_json(self.metadata.get_vitpose_path(img_id))
        vitpose = np.array(vitpose_load['kps']).astype(np.float32)

        epro_pnp_coor = self.metadata.get_pred_coor_map_path(img_id)
        epro_pnp_coor = load_pickle(epro_pnp_coor)
        obj_x3d = epro_pnp_coor['x3d']
        obj_x2d = epro_pnp_coor['x2d']
        obj_w2d = epro_pnp_coor['w2d']

        object_corr_load = load_pickle(self.metadata.get_pred_dino_coor_map_path(img_id))
        pose_init = object_corr_load['pose_init'].astype(np.float32)
        obj_rot_init = pose_init[3:].astype(np.float32)
        obj_trans_init = pose_init[:3].astype(np.float32)

        return img_id, object_label, hoi_image, crop_bbox, cam_K, vitpose, obj_x3d, obj_x2d, obj_w2d, obj_rot_init, obj_trans_init


def evaluate_framewise_on_behave(cfg):
    device = torch.device('cuda')

    stackflow = Model(cfg)
    stackflow.to(device)
    stackflow.load_checkpoint(cfg.train.resume)
    stackflow.eval()

    output_dir = os.path.dirname(cfg.train.resume)

    metadata = BEHAVEExtendMetaData(cfg.dataset.root_dir)
    all_sequences = metadata.get_all_image_by_sequence(split='test')

    smpl = SMPLHLayer(model_path='data/models/smplh', gender='male').to(device)
    object_verts_all = np.zeros((20, metadata.object_max_vertices_num, 3))
    object_kps_all = np.zeros((20, metadata.object_max_keypoint_num, 3))
    object_kps_masks_all = np.zeros((20, metadata.object_max_keypoint_num))
    for object_name, object_idx in metadata.OBJECT_NAME2IDX.items():
        object_v, object_f = metadata.obj_mesh_templates[object_name]
        object_verts_all[object_idx, :object_v.shape[0]] = object_v

        object_kps = metadata.load_object_keypoints(object_name)
        object_kps_all[object_idx, :object_kps.shape[0]] = object_kps
        object_kps_masks_all[object_idx, :object_kps.shape[0]] = 1
    object_verts_all = torch.tensor(object_verts_all).float().to(device)
    object_kps_all = torch.tensor(object_kps_all).float().to(device)

    keyframes = load_pickle('data/datasets/behave-split-30fps-keyframes.pkl')['test']
    seq_renames = {'Date02_Sub02_monitor_move2': 'Date02_Sub02_monitor_move', 
                   'Date02_Sub02_toolbox_part2': 'Date02_Sub02_toolbox',
                   'Date03_Sub04_boxtiny_part2': 'Date03_Sub04_boxtiny',
                   'Date03_Sub04_yogaball_play2': 'Date03_Sub04_yogaball_play',
                   'Date03_Sub05_chairwood_part2': 'Date03_Sub05_chairwood',
                   'Date04_Sub05_monitor_part2': 'Date04_Sub05_monitor'}
    img_id_keyframes = []
    for path in keyframes:
        seq_name, frame_name, file_name = path.split('/')
        if seq_name in seq_renames:
            seq_name = seq_renames[seq_name]
        day_id, sub_id, obj_name, inter_type = metadata.parse_seq_info(seq_name)
        # if obj_name != 'backpack':
        #     continue
        frame_id = frame_name[2:]
        cam_id = file_name[1]
        img_id = '_'.join([day_id, sub_id, obj_name, inter_type, frame_id, cam_id])
        img_id_keyframes.append(img_id)

    test_dataset = TestDataset(metadata, img_id_keyframes)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=32, num_workers=4, shuffle=False, drop_last=False)

    recon_results = {}
    for idx, batch in enumerate(tqdm(test_dataloader)):
        img_ids, object_labels, hoi_images, hoi_bboxes, cam_Ks, vitposes, obj_x3ds, obj_x2ds, obj_w2ds, obj_rot_init, obj_trans_init = batch

        batch_size = hoi_images.shape[0]
        stackflow_out = stackflow.inference({
            'image': hoi_images.to(device), 
            'box_size': hoi_bboxes[:, 2].to(device), 
            'box_center': hoi_bboxes[:, :2].to(device),
            'optical_center':  torch.stack([cam_Ks[:, 0, 2], cam_Ks[:, 1, 2]], dim=-1).to(device),
            'focal_length': torch.stack([cam_Ks[:, 0, 0], cam_Ks[:, 1, 1]], dim=-1).to(device),
            'object_labels': object_labels.to(device)}, debug=False)

        smpl_betas = stackflow_out['pred_betas'].detach()
        smpl_body_pose6d = stackflow_out['pred_pose6d'][:, 1:].detach()
        obj_rel_trans_init = stackflow_out['pred_obj_rel_T'].detach()
        obj_rel_rotmat_init = stackflow_out['pred_obj_rel_R'].detach()
        hoi_trans = stackflow_out['hoi_trans'].detach()
        hoi_rot6d = matrix_to_rotation_6d(stackflow_out['hoi_rotmat'].detach())
        hoi_feats = stackflow_out['visual_features'].detach()

        object_labels = object_labels.to(device)
        object_kps = object_kps_all[object_labels]
        object_v = object_verts_all[object_labels]
        hoi_instance = HOIInstance(smpl, object_kps, object_v, smpl_betas, smpl_body_pose6d, obj_rel_trans_init, obj_rel_rotmat_init, hoi_trans, hoi_rot6d).to(device)

        loss_functions = [
            SMPLKpsProjLoss(vitposes, cam_Ks).to(device),
            ObjectEproPnpLoss(model_points=obj_x3ds, 
                             image_points=obj_x2ds, 
                             pts_confidence=obj_w2ds, 
                             focal_length=torch.stack([cam_Ks[:, 0, 0], cam_Ks[:, 1, 1]], dim=-1).to(device),
                             optical_center=torch.stack([cam_Ks[:, 0, 2], cam_Ks[:, 1, 2]], dim=-1).to(device),).to(device),
            HOOffsetLoss(stackflow.stackflow, stackflow.flow_loss.hooffset, hoi_feats, object_labels).to(device),
        ]
        param_dicts = [
            {"params": [hoi_instance.smpl_body_pose6d, hoi_instance.obj_rel_trans, hoi_instance.obj_rel_rot6d,
            hoi_instance.hoi_trans, hoi_instance.hoi_rot6d]},
        ]
        parameters = [hoi_instance.smpl_body_pose6d, hoi_instance.obj_rel_trans, hoi_instance.obj_rel_rot6d,
            hoi_instance.hoi_trans, hoi_instance.hoi_rot6d]
        optimizer = torch.optim.Adam(param_dicts, lr=0.05, betas=(0.9, 0.999))

        loss_weights = {
            'object_reproj_loss': lambda cst, it: 10. ** 1 * cst / (1 + 10 * it),
            'loss_body_kps2d': lambda cst, it: 10. ** -1 * cst / (1 + 10 * it),
            'loss_theta_nll': lambda cst, it: 10. ** 0 * cst / (1 + 10 * it),
            'loss_gamma_nll': lambda cst, it: 10. ** 0 * cst / (1 + 10 * it),
        }
        itertations = 2
        steps_per_iter = 500
        post_optimization(hoi_instance, optimizer, parameters, loss_functions, loss_weights, itertations, steps_per_iter)

        hoi_outputs = hoi_instance.forward()
        for batch_idx in range(batch_size):
            img_id = img_ids[batch_idx]
            recon_results[img_id] = {
                'betas': hoi_outputs['smpl_betas'][batch_idx].reshape(10, ).detach().cpu().numpy(),
                'body_pose_rotmat': hoi_outputs['smpl_body_rotmat'][batch_idx].reshape(-1, 3, 3)[:21].detach().cpu().numpy(),
                'hoi_trans': hoi_outputs['hoi_trans'][batch_idx].reshape(3, ).detach().cpu().numpy(),
                'hoi_rotmat': hoi_outputs['hoi_rotmat'][batch_idx].reshape(3, 3).detach().cpu().numpy(),
                'obj_rel_R': hoi_outputs['obj_rel_rotmat'][batch_idx].reshape(3, 3).detach().cpu().numpy(),
                'obj_rel_T': hoi_outputs['obj_rel_trans'][batch_idx].reshape(3, ).detach().cpu().numpy(),
                'cam_K': cam_Ks[batch_idx].reshape(3, 3).detach().cpu().numpy(),
            }

    save_pickle(recon_results, os.path.join(output_dir, 'recon_results_framewise_optim.pkl'))
    smpl_male = SMPLHLayer(model_path='data/models/smplh', gender='male').to(device)
    smpl_female = SMPLHLayer(model_path='data/models/smplh', gender='female').to(device)

    gt_annotations = {}
    for file in os.listdir('data/datasets/behave_extend_datalist'):
        annotation_load = load_pickle(os.path.join('data/datasets/_behave_extend_datalist', file))
        for cam_id in annotation_load:
            for item in annotation_load[cam_id]:
                gt_annotations[item['img_id']] = item

    hoi_evaluator = ReconEvaluator(align_mesh=False, smpl_only=False)
    hoi_evaluator_aligned = ReconEvaluator(align_mesh=True, smpl_only=False)
    smpl_evaluator_aligned = ReconEvaluator(align_mesh=True, smpl_only=True)

    evaluate_results = {}
    for img_id in tqdm(recon_results):

        recon_smpl, recon_object = get_recon_meshes(metadata, img_id, smpl_male, recon_results[img_id])
        gt_smpl, gt_object = get_gt_meshes(metadata, img_id, smpl_male, smpl_female, gt_annotations)

        smpl_error, object_error = hoi_evaluator.compute_errors([gt_smpl, gt_object], [recon_smpl, recon_object])
        hoi_smpl_error, hoi_obj_error = hoi_evaluator_aligned.compute_errors([gt_smpl, gt_object], [recon_smpl, recon_object])
        smpl_aligned_error, _ = smpl_evaluator_aligned.compute_errors([gt_smpl, gt_object], [recon_smpl, recon_object])
        evaluate_results[img_id] = {
            'hoi_smpl_error': hoi_smpl_error,
            'hoi_obj_error': hoi_obj_error,
            'smpl_error': smpl_error,
            'object_error': object_error,
            'smpl_aligned_error': smpl_aligned_error,
        }
        print(evaluate_results[img_id])

    all_hoi_smpl_errors = [item['hoi_smpl_error'] for item in evaluate_results.values()]
    all_hoi_obj_errors = [item['hoi_obj_error'] for item in evaluate_results.values()]
    all_smpl_errors = [item['smpl_error'] for item in evaluate_results.values()]
    all_object_errors = [item['object_error'] for item in evaluate_results.values()]
    all_smpl_aligned_errors = [item['smpl_aligned_error'] for item in evaluate_results.values()]

    evaluate_results['avg'] = {
        'hoi_smpl_error': np.mean(all_hoi_smpl_errors),
        'hoi_obj_error': np.mean(all_hoi_obj_errors),
        'smpl_error': np.mean(all_smpl_errors),
        'object_error': np.mean(all_object_errors),
        'smpl_aligned_error': np.mean(all_smpl_aligned_errors),
    }
    evaluate_results['std'] = {
        'hoi_smpl_error': np.std(all_hoi_smpl_errors),
        'hoi_obj_error': np.std(all_hoi_obj_errors),
        'smpl_error': np.std(all_smpl_errors),
        'object_error': np.std(all_object_errors),
        'smpl_aligned_error': np.std(all_smpl_aligned_errors),
    }
    print(evaluate_results['avg'])

    save_json(evaluate_results, os.path.join(output_dir, 'evaluate_results_framewise_optim.json'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root_dir', default='/storage/data/huochf/BEHAVE', type=str)
    args = parser.parse_args()

    cfg = load_config()
    cfg.dataset.root_dir = args.dataset_root_dir
    cfg.freeze()
    set_seed(7)
    evaluate_framewise_on_behave(cfg)
