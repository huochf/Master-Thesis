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

from prohmr.configs import get_config as get_prohmr_config
from prohmr.datasets.utils import generate_image_patch, convert_cvimg_to_tensor
from externals.prohmr_custom.model import ProHMR

from hoi_recon.datasets.behave_extend_metadata import BEHAVEExtendMetaData
from hoi_recon.datasets.utils import generate_image_patch as generate_hoi_image_patch
from hoi_recon.datasets.utils import load_json, load_pickle, save_pickle, save_json
from hoi_recon.models.hoi_instance import HOIInstance
from hoi_recon.utils.evaluator import get_recon_meshes, get_gt_meshes, ReconEvaluator
from hoi_recon.utils.optim_losses import SMPLPostPriorLoss, SMPLKpsProjLoss, ObjectEproPnpLoss, HOIKPS3DLoss
from hoi_recon.utils.post_optimization import post_optimization

from .train_c_flow_kps3d import Model


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
        person_mask = cv2.imread(self.metadata.get_person_mask_path(img_id), cv2.IMREAD_GRAYSCALE) / 255
        person_bbox = extract_bbox_from_mask(person_mask)
        annotation = self.annotations[img_id]
        person_bbox = annotation['person_bb_xyxy']
        x1, y1, x2, y2 = person_bbox
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        s = max(1, 1.2 * max(x2 - x1, y2 - y1))

        focal_length = np.array([annotation['cam_K'][0, 0], annotation['cam_K'][1, 1]]).astype(np.float32)
        roi_center = np.array([cx, cy]).astype(np.float32)
        roi_size = np.array([s, s]).astype(np.float32)

        person_image, _ = generate_image_patch(image, cx, cy, s, s, 224, 224, False, 1.0, 0.)
        person_image = person_image[:, :, ::-1]
        person_image = convert_cvimg_to_tensor(person_image)

        for n_c in range(3):
            person_image[n_c, :, :] = np.clip(person_image[n_c, :, :], 0, 255)
            person_image[n_c, :, :] = (person_image[n_c, :, :] - 255 * self.mean[n_c]) / (self.std[n_c] * 255)

        crop_bbox = np.array([cx, cy, s, s]).astype(np.float32)

        fx, fy, cx, cy = self.metadata.cam_intrinsics[int(cam_id)][:4]
        cam_K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]).astype(np.float32)

        hoi_bbox = annotation['hoi_bb_xyxy']
        x1, y1, x2, y2 = hoi_bbox
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        box_size = 1.2 * max(x2 - x1, y2 - y1)
        hoi_image, _ = generate_hoi_image_patch(image, cx, cy, box_size, 256, 0, [1., 1., 1.])
        hoi_image = hoi_image[:, :, ::-1].astype(np.float32) # convert to RGB
        hoi_image = hoi_image.transpose((2, 0, 1))
        hoi_image = hoi_image / 256
        for n_c in range(3):
            hoi_image[n_c, :, :] = np.clip(hoi_image[n_c, :, :], 0, 1)
            hoi_image[n_c, :, :] = (hoi_image[n_c, :, :] - self.mean[n_c]) / self.std[n_c]

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

        return img_id, object_label, person_image.astype(np.float32), hoi_image, crop_bbox, cam_K, vitpose, obj_x3d, obj_x2d, obj_w2d, obj_rot_init, obj_trans_init


def evaluate_framewise_on_behave(args):
    device = torch.device('cuda')

    prohmr_cfg = get_prohmr_config('externals/prohmr_custom/configs/prohmr_behave_extend.yaml')
    checkpoint = 'outputs/prohmr/behave_extend/checkpoints_cam_t/epoch=2-step=40000.ckpt'
    prohmr = ProHMR.load_from_checkpoint(checkpoint, strict=False, cfg=prohmr_cfg)
    prohmr.eval()

    metadata = BEHAVEExtendMetaData(args.dataset_root_dir)
    all_sequences = metadata.get_all_image_by_sequence(split='test')

    flow_dim = (22 + metadata.object_num_keypoints[args.object]) * 3
    model = Model(flow_dim=flow_dim, 
                  flow_width=512, 
                  c_dim=256,
                  num_blocks_per_layers=2,
                  layers=4,
                  dropout_probability=1.).to(device)
    state_dict = torch.load('./outputs/cflow_kps3d/checkpoint_{}.pth'.format(args.object))
    model.load_state_dict(state_dict['model'])
    model.eval()
    output_dir = 'outputs/cflow_kps3d/'

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
        if obj_name != args.object:
            continue
        frame_id = frame_name[2:]
        cam_id = file_name[1]
        img_id = '_'.join([day_id, sub_id, obj_name, inter_type, frame_id, cam_id])
        img_id_keyframes.append(img_id)

    test_dataset = TestDataset(metadata, img_id_keyframes)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=32, num_workers=4, shuffle=False, drop_last=False)

    recon_results = {}
    for idx, batch in enumerate(tqdm(test_dataloader)):

        img_ids, object_labels, person_images, hoi_images, person_bboxes, cam_Ks, vitposes, obj_x3ds, obj_x2ds, obj_w2ds, obj_rot_init, obj_trans_init = batch

        batch_size = person_images.shape[0]
        prohmr_outputs = prohmr.forward_step({'img': person_images.to(device), 'K': cam_Ks.to(device), 
            'roi_center': person_bboxes[:, :2].to(device), 'roi_size': person_bboxes[:, 2:].to(device), 
            'focal_length': torch.stack([cam_Ks[:, 0, 0], cam_Ks[:, 1, 1]], dim=-1).to(device)})
        smpl_betas = prohmr_outputs['pred_smpl_params']['betas'][:, 0].detach()
        smpl_body_rotmat = prohmr_outputs['pred_smpl_params']['body_pose'][:, 0, :21].detach()
        smpl_body_pose6d = matrix_to_rotation_6d(smpl_body_rotmat)
        hoi_trans = prohmr_outputs['global_cam_t'][:, 0].detach()
        hoi_rotmat = prohmr_outputs['pred_smpl_params']['global_orient'][:, 0].detach().reshape(batch_size, 3, 3)
        hoi_rot6d = matrix_to_rotation_6d(hoi_rotmat)

        person_feats = prohmr_outputs['conditioning_feats'].detach()
        hoi_feats = model.image_embedding(hoi_images.to(device)).detach().reshape(batch_size, -1)

        obj_trans_init = obj_trans_init.to(device)
        obj_rot_init = obj_rot_init.to(device)
        obj_rotmat_init = axis_angle_to_matrix(obj_rot_init)
        obj_smpl_dist, _ = F.l1_loss(hoi_trans, obj_trans_init, reduction='none').max(-1)
        obj_trans_init[obj_smpl_dist > 1.5] = hoi_trans[obj_smpl_dist > 1.5]
        obj_rel_rotmat_init = hoi_rotmat.transpose(-2, -1) @ obj_rotmat_init
        obj_rel_trans_init = torch.matmul((obj_trans_init - hoi_trans).reshape(-1, 1, 3), hoi_rotmat).reshape(-1, 3)

        object_labels = object_labels.to(device)
        object_kps = object_kps_all[object_labels]
        object_v = object_verts_all[object_labels]

        hoi_instance = HOIInstance(smpl, object_kps, object_v, smpl_betas, smpl_body_pose6d, obj_rel_trans_init, obj_rel_rotmat_init, hoi_trans, hoi_rot6d).to(device)

        optimizer = torch.optim.Adam([hoi_instance.smpl_betas, hoi_instance.smpl_body_pose6d, hoi_instance.obj_rel_trans, hoi_instance.obj_rel_rot6d,
            hoi_instance.hoi_trans, hoi_instance.hoi_rot6d], lr=0.05, betas=(0.9, 0.999))

        loss_functions = [
            SMPLPostPriorLoss(prohmr, smpl_betas.clone(), person_feats).to(device),
            SMPLKpsProjLoss(vitposes, cam_Ks).to(device),
            ObjectEproPnpLoss(model_points=obj_x3ds, 
                             image_points=obj_x2ds, 
                             pts_confidence=obj_w2ds, 
                             focal_length=torch.stack([cam_Ks[:, 0, 0], cam_Ks[:, 1, 1]], dim=-1).to(device),
                             optical_center=torch.stack([cam_Ks[:, 0, 2], cam_Ks[:, 1, 2]], dim=-1).to(device),).to(device),
            HOIKPS3DLoss(model, hoi_feats, n_obj_kps=metadata.object_num_keypoints[args.object]).to(device),
        ]
        loss_weights = {
            'smpl_prior': lambda cst, it: 10. ** -1 * cst / (1 + 10 * it),
            'loss_shape_prior': lambda cst, it: 10. ** -1 * cst / (1 + 10 * it),
            'object_reproj_loss': lambda cst, it: 10. ** 1 * cst / (1 + 10 * it),
            'loss_body_kps2d': lambda cst, it: 10. ** -1 * cst / (1 + 10 * it),
            'loss_hoi_kps3d': lambda cst, it: 10. ** 0 * cst / (1 + 10 * it),
        }
        parameters = [hoi_instance.smpl_betas, hoi_instance.smpl_body_pose6d, hoi_instance.obj_rel_trans, hoi_instance.obj_rel_rot6d,
            hoi_instance.hoi_trans, hoi_instance.hoi_rot6d]
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

    save_pickle(recon_results, os.path.join(output_dir, 'recon_results_framewise_optim_{}.pkl'.format(args.object)))
    smpl_male = SMPLHLayer(model_path='data/models/smplh', gender='male').to(device)
    smpl_female = SMPLHLayer(model_path='data/models/smplh', gender='female').to(device)

    gt_annotations = {}
    for file in os.listdir('data/datasets/behave_extend_datalist'):
        annotation_load = load_pickle(os.path.join('data/datasets/behave_extend_datalist', file))
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

    save_json(evaluate_results, os.path.join(output_dir, 'evaluate_results_framewise_optim_{}.json'.format(args.object)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root_dir', default='/storage/data/huochf/BEHAVE', type=str)
    parser.add_argument('--object', default='backpack', type=str)
    args = parser.parse_args()

    set_seed(7)
    evaluate_framewise_on_behave(args)
