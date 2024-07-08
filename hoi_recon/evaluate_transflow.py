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
from pytorch3d.transforms import matrix_to_rotation_6d, axis_angle_to_matrix, quaternion_to_matrix, rotation_6d_to_matrix

from prohmr.configs import get_config as get_prohmr_config
from prohmr.datasets.utils import generate_image_patch, convert_cvimg_to_tensor
from externals.prohmr_custom.model import ProHMR

from hoi_recon.datasets.behave_extend_metadata import BEHAVEExtendMetaData
from hoi_recon.datasets.utils import generate_image_patch as generate_hoi_image_patch
from hoi_recon.datasets.utils import load_json, load_pickle, save_pickle, save_json
from hoi_recon.models.hoi_instance_seq import HOIInstanceSeq
from hoi_recon.utils.sequence_evaluator import ReconEvaluator, get_recon_meshes, get_gt_meshes
from hoi_recon.utils.optim_losses_seq import SMPLPostPriorLoss, SMPLKpsProjLoss, ObjectEproPnpLoss, SmoothLoss, MultiViewPseudoKps2DSeqLoss, TransflowKps3DSeqLoss
from hoi_recon.utils.post_optimization import post_optimization

# from .train_transflow_pseudo_kps2d_seq import Model
from .train_transflow_kps3d_seq import Model


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


def smooth_kps(wholebody_kps_seq, windows=5):
    # [n, 133, 3]

    seq_n, kps_n, _ = wholebody_kps_seq.shape
    smooth_kps = torch.cat([torch.zeros((windows // 2, kps_n, 3)).float().to(wholebody_kps_seq.device), 
                            wholebody_kps_seq, 
                            torch.zeros((windows // 2, kps_n, 3)).float().to(wholebody_kps_seq.device)], dim=0) # [seq_n + windows - 1, n, 3]
    confidence_score = torch.stack([
        smooth_kps[i: seq_n + i, :, 2:] for i in range(windows)
    ], dim=0)
    smooth_kps = torch.stack([
        smooth_kps[i: seq_n + i, :, :2] for i in range(windows)
    ], dim=0)
    smooth_kps = (smooth_kps * confidence_score).sum(0) / (confidence_score.sum(0) + 1e-8)
    smooth_kps = torch.cat([smooth_kps, wholebody_kps_seq[:, :, 2:]], dim=2)

    return smooth_kps


class TestDataset:

    def __init__(self, metadata, img_ids):
        self.metadata = metadata
        self.img_ids = img_ids
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])
        self.annotations = self.load_annotations(train=True)


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
                        'object_rel_trans': item['object_rel_trans'],
                        'object_rel_rotmat': item['object_rel_rotmat'],
                        'smplh_pose_rotmat': item['smplh_pose_rotmat'],
                        'cam_K': item['cam_K'],
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
        s = 1.2 * max(x2 - x1, y2 - y1)

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
        # vitpose_load = load_json(self.metadata.get_vitpose_path(img_id))
        # vitpose = np.array(vitpose_load['kps']).astype(np.float32)

        # epro_pnp_coor = self.metadata.get_pred_coor_map_path(img_id)
        # epro_pnp_coor = load_pickle(epro_pnp_coor)
        # obj_x3d = epro_pnp_coor['x3d']
        # obj_x2d = epro_pnp_coor['x2d']
        # obj_w2d = epro_pnp_coor['w2d']

        # object_corr_load = load_pickle(self.metadata.get_pred_dino_coor_map_path(img_id))
        # pose_init = object_corr_load['pose_init'].astype(np.float32)
        # obj_rot_init = pose_init[3:].astype(np.float32)
        # obj_trans_init = pose_init[:3].astype(np.float32)

        return img_id, object_label, person_image.astype(np.float32), hoi_image, crop_bbox, cam_K, \
            annotation['smplh_betas_male'], annotation['smplh_pose_rotmat'], annotation['object_rel_trans'], annotation['object_rel_rotmat'] # vitpose, obj_x3d, obj_x2d, obj_w2d, obj_rot_init, obj_trans_init,


def evaluate_sequence_on_behave(args):
    device = torch.device('cuda')

    prohmr_cfg = get_prohmr_config('externals/prohmr_custom/configs/prohmr_behave_extend.yaml')
    checkpoint = 'outputs/prohmr/behave_extend/checkpoints_cam_t/epoch=2-step=40000.ckpt'
    # checkpoint = 'outputs/prohmr/_behave_extend/checkpoints_cam_t/epoch=3-step=70000.ckpt'
    prohmr = ProHMR.load_from_checkpoint(checkpoint, strict=False, cfg=prohmr_cfg)
    prohmr.eval()

    metadata = BEHAVEExtendMetaData(args.dataset_root_dir)
    all_sequences = metadata.get_all_image_by_sequence(split='train')

    # flow_dim = (22 + metadata.object_num_keypoints['backpack'] + 1) * 3
    flow_dim = (22 + metadata.object_num_keypoints['backpack']) * 3
    model = Model(flow_dim=flow_dim, 
                  flow_width=512, 
                  pos_dim=256, 
                  c_dim=256, 
                  seq_len=31,
                  num_blocks_per_layers=2,
                  layers=4,
                  head_dim=256,
                  num_heads=8,
                  dropout_probability=0.).to(device)
    output_dir = 'outputs/transflow_kps3d_seq/'
    state_dict = torch.load(os.path.join(output_dir, 'checkpoint_{}.pth'.format(args.object)))
    model.load_state_dict(state_dict['model'])
    model.eval()

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

    recon_results = {}
    for seq_name in all_sequences:
        day_id, sub_id, obj_name, inter_type = metadata.parse_seq_info(seq_name)
        if args.object != 'all' and obj_name != args.object:
            continue

        recon_results[seq_name] = {}
        for cam_id in all_sequences[seq_name]:
            if int(cam_id) != 1: # following CHORE, we only test on one camera
                continue
            img_ids = all_sequences[seq_name][cam_id][:200]

            test_dataset = TestDataset(metadata, img_ids)
            test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, num_workers=4, shuffle=False, drop_last=False)

            smpl_betas_seq = []
            smpl_body_pose6d_seq = []
            hoi_rot6d_seq = []
            hoi_trans_seq = []
            person_feats_seq = []
            vitposes_seq = []
            obj_trans_seq = []
            obj_rot_seq = []
            obj_x3d_seq = []
            obj_x2d_seq = []
            obj_w2d_seq = []
            cam_K_seq = []
            hoi_feats_seq = []
            object_labels_seq = []

            smpl_betas_seq_debug = []
            smpl_body_pose6d_seq_debug = []
            obj_rel_rotmat_seq_debug = []
            obj_rel_trans_seq_debug = []

            for idx, batch in enumerate(tqdm(test_dataloader)):
                _, object_labels, person_images, hoi_images, person_bboxes, cam_Ks, \
                    smpl_betas_gt, smpl_pose_rotmat_gt, object_rel_trans_gt, object_rel_rotmat_gt = batch
                # vitposes, obj_x3ds, obj_x2ds, obj_w2ds, obj_rot_init, obj_trans_init, \
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
                hoi_feats_seq.append(hoi_feats)
                # vitposes_seq.append(vitposes)

                # obj_trans_init = obj_trans_init.to(device)
                # obj_rot_init = obj_rot_init.to(device)
                # obj_smpl_dist, _ = F.l1_loss(hoi_trans, obj_trans_init, reduction='none').max(-1)
                # obj_trans_init[obj_smpl_dist > 1.5] = hoi_trans[obj_smpl_dist > 1.5]

                object_labels = object_labels.to(device)
                object_labels_seq.append(object_labels)
                object_kps = object_kps_all[object_labels]
                object_v = object_verts_all[object_labels]

                smpl_betas_seq.append(smpl_betas)
                smpl_body_pose6d_seq.append(smpl_body_pose6d)
                hoi_rot6d_seq.append(hoi_rot6d)
                hoi_trans_seq.append(hoi_trans)
                person_feats_seq.append(person_feats)
                # obj_trans_seq.append(obj_trans_init)
                # obj_rot_seq.append(obj_rot_init)
                # obj_x3d_seq.append(obj_x3ds)
                # obj_x2d_seq.append(obj_x2ds)
                # obj_w2d_seq.append(obj_w2ds)
                cam_K_seq.append(cam_Ks)

                smpl_betas_seq_debug.append(smpl_betas_gt)
                smpl_body_pose6d_seq_debug.append(smpl_pose_rotmat_gt[:, 1:22])
                obj_rel_trans_seq_debug.append(object_rel_trans_gt)
                obj_rel_rotmat_seq_debug.append(object_rel_rotmat_gt)

            smpl_betas_seq = torch.cat(smpl_betas_seq, dim=0)
            smpl_body_pose6d_seq = torch.cat(smpl_body_pose6d_seq, dim=0)
            hoi_rot6d_seq = torch.cat(hoi_rot6d_seq, dim=0)
            hoi_rotmat_seq = rotation_6d_to_matrix(hoi_rot6d_seq)
            hoi_trans_seq = torch.cat(hoi_trans_seq, dim=0)
            person_feats_seq = torch.cat(person_feats_seq, dim=0)
            # vitposes_seq = torch.cat(vitposes_seq, dim=0)
            # obj_x3d_seq = torch.cat(obj_x3d_seq, dim=0)
            # obj_x2d_seq = torch.cat(obj_x2d_seq, dim=0)
            # obj_w2d_seq = torch.cat(obj_w2d_seq, dim=0)
            cam_K_seq = torch.cat(cam_K_seq, dim=0)

            # vitposes_seq = smooth_kps(vitposes_seq)
            hoi_feats_seq = torch.cat(hoi_feats_seq, dim=0)
            object_labels_seq = torch.cat(object_labels_seq, dim=0)

            n_seq = smpl_betas_seq.shape[0]
            # obj_rot6d_seq = matrix_to_rotation_6d(axis_angle_to_matrix(torch.cat(obj_rot_seq, dim=0).reshape(n_seq, 3))).detach().cpu().numpy()
            # obj_trans_seq = torch.cat(obj_trans_seq, dim=0).reshape(n_seq, 3).detach().cpu().numpy()
            # obj_rot6d_seq, obj_trans_seq = smooth_obj_RT(obj_rot6d_seq, obj_trans_seq)

            # obj_rot6d_seq = torch.tensor(obj_rot6d_seq).float().to(device)
            # obj_trans_seq = torch.tensor(obj_trans_seq).float().to(device)
            # obj_rel_rotmat_seq = hoi_rotmat_seq.transpose(-2, -1) @ rotation_6d_to_matrix(obj_rot6d_seq)
            # obj_rel_trans_seq = torch.matmul((obj_trans_seq - hoi_trans_seq).reshape(-1, 1, 3), hoi_rotmat_seq).reshape(-1, 3)

            smpl_betas_seq = torch.cat(smpl_betas_seq_debug).float().to(device)
            smpl_body_pose6d_seq_debug = torch.cat(smpl_body_pose6d_seq_debug).float().to(device)
            obj_rel_rotmat_seq = torch.cat(obj_rel_rotmat_seq_debug).float().to(device)
            obj_rel_trans_seq = torch.cat(obj_rel_trans_seq_debug).float().to(device)
            smpl_body_pose6d_seq = matrix_to_rotation_6d(smpl_body_pose6d_seq_debug)

            hoi_instance = HOIInstanceSeq(smpl, object_kps[:1], object_v[:1], smpl_betas_seq, smpl_body_pose6d_seq, 
                obj_rel_trans_seq, obj_rel_rotmat_seq, hoi_trans_seq, hoi_rot6d_seq, ).to(device)

            loss_weights = {
                'smpl_prior': lambda cst, it: 10. ** -1 * cst / (1 + 10 * it),
                'object_reproj_loss': lambda cst, it: 10. ** 1 * cst / (1 + 10 * it),
                'loss_body_kps2d': lambda cst, it: 10. ** -1 * cst / (1 + 10 * it),
                'loss_shape_prior': lambda cst, it: 10. ** -1 * cst / (1 + 10 * it),
                'loss_smo_smpl_v': lambda cst, it: 10. ** 1 * cst / (1 + 10 * it),
                'loss_smo_obj_v': lambda cst, it: 10. ** 1 * cst / (1 + 10 * it),
                'loss_smo_obj_t': lambda cst, it: 10. ** 1 * cst / (1 + 10 * it),
                'loss_smo_obj_r': lambda cst, it: 10. ** 1 * cst / (1 + 10 * it),
                'loss_kps_nll': lambda cst, it: 0 * 10. ** -2 * cst / (1 + 10 * it),
                'loss_kps_samples_l1': lambda cst, it: 10. ** -1 * cst / (1 + 10 * it),
            }
            loss_functions = [
                SMPLPostPriorLoss(prohmr, smpl_betas_seq.clone(), person_feats_seq).to(device),
                # SMPLKpsProjLoss(vitposes_seq, cam_K_seq).to(device),
                # ObjectEproPnpLoss(model_points=obj_x3d_seq, 
                #                  image_points=obj_x2d_seq, 
                #                  pts_confidence=obj_w2d_seq, 
                #                  focal_length=torch.stack([cam_K_seq[:, 0, 0], cam_K_seq[:, 1, 1]], dim=-1).to(device),
                #                  optical_center=torch.stack([cam_K_seq[:, 0, 2], cam_K_seq[:, 1, 2]], dim=-1).to(device),).to(device),
                # MultiViewPseudoKps2DSeqLoss(model, hoi_feats=hoi_feats_seq, window_raidus=15, alpha=1., n_views=8).to(device),
                TransflowKps3DSeqLoss(model, hoi_feats=hoi_feats_seq, window_raidus=15, alpha=10000.).to(device),
                # SmoothLoss().to(device),
            ]
            param_dicts = [
                # {"params": [loss_functions[-1].cam_pos, ], "lr": 1e-3},
                {"params": [hoi_instance.smpl_betas, hoi_instance.smpl_body_pose6d, hoi_instance.obj_rel_trans, hoi_instance.obj_rel_rot6d,
                hoi_instance.hoi_trans, hoi_instance.hoi_rot6d]},
            ]
            parameters = [hoi_instance.smpl_betas, hoi_instance.smpl_body_pose6d, hoi_instance.obj_rel_trans,
                hoi_instance.obj_rel_rot6d, hoi_instance.hoi_trans, hoi_instance.hoi_rot6d,] # loss_functions[-1].cam_pos]

            optimizer = torch.optim.Adam(param_dicts, lr=0.05, betas=(0.9, 0.999))

            iterations = 2
            steps_per_iter = 300
            interval_len = 64
            interval_step = 10

            for it in range(iterations):
                loop = tqdm(range(steps_per_iter))
                for i in loop:
                    optimizer.zero_grad()
                    total_loss = 0

                    losses_seq = {}
                    # for begin_idx in range(i % interval_len, n_seq, interval_len):
                    for begin_idx in range(0, n_seq, interval_len):
                        losses = {}
                        hoi_out = hoi_instance.forward(begin_idx, interval_len)
                        for f in loss_functions:
                            losses.update(f(hoi_out, begin_idx, interval_len))
                        loss_list = [loss_weights[k](v.mean(), it) for k, v in losses.items()]
                        total_loss += torch.stack(loss_list).sum()

                        for k, v in losses.items():
                            if k not in losses_seq:
                                losses_seq[k] = []
                            losses_seq[k].append(v.mean())
                    total_loss = total_loss / n_seq
                    # total_loss.backward()
                    # torch.nn.utils.clip_grad_norm_(parameters, 0.1)
                    # optimizer.step()

                    l_str = 'Optim. Step {}: Iter: {}, loss: {:.4f}'.format(it, i, total_loss.item())
                    for k, v in losses_seq.items():
                        # if k == 'loss_hoi_kps3d':
                        #     print(torch.stack(v).min())
                        l_str += ', {}: {:.4f}'.format(k, torch.stack(v).mean().detach().item())
                    loop.set_description(l_str)

            hoi_out = hoi_instance.forward(0, n_seq)

            recon_results[seq_name][cam_id] = {
                'img_ids': img_ids,
                'smpl_betas': hoi_out['smpl_betas'].detach().cpu().numpy(),
                'smpl_body_rotmat': hoi_out['smpl_body_rotmat'].detach().cpu().numpy(),
                'obj_rel_trans': hoi_out['obj_rel_trans'].detach().cpu().numpy(),
                'obj_rel_rotmat': hoi_out['obj_rel_rotmat'].detach().cpu().numpy(),
                'hoi_rotmat': hoi_out['hoi_rotmat'].detach().cpu().numpy(),
                'hoi_trans': hoi_out['hoi_trans'].detach().cpu().numpy(),
                'object_scale': hoi_out['object_scale'].detach().cpu().numpy(),
                'cam_Ks': cam_K_seq.detach().cpu().numpy(),
            }
        print('Sequence {} done!'.format(seq_name))
        break

    save_pickle(recon_results, os.path.join(output_dir, 'recon_results_sequence_optim_alpha_1_{}.pkl'.format(args.object)))

    smpl_male = SMPLHLayer(model_path='data/models/smplh', gender='male').to(device)
    smpl_female = SMPLHLayer(model_path='data/models/smplh', gender='female').to(device)

    gt_annotations = {}
    for file in os.listdir('data/datasets/behave_extend_datalist'):
        if file[5] != '3': # test sequences
            continue
        annotation_load = load_pickle(os.path.join('data/datasets/behave_extend_datalist', file))
        for cam_id in annotation_load:
            for item in annotation_load[cam_id]:
                gt_annotations[item['img_id']] = item

    hoi_evaluator_aligned_win10 = ReconEvaluator(align_mesh=True, smpl_only=False, window_len=10)
    hoi_evaluator_aligned = ReconEvaluator(align_mesh=True, smpl_only=False)
    smpl_evaluator_aligned = ReconEvaluator(align_mesh=True, smpl_only=True)

    evaluate_results = {}
    for seq_name in recon_results:
        evaluate_results[seq_name] = {}
        for cam_id in recon_results[seq_name]:

            recon_smpl_seq, recon_object_seq = [], []
            gt_smpl_seq, gt_object_seq = [], []

            for idx, img_id in enumerate(recon_results[seq_name][cam_id]['img_ids']):
                results_per_img = {k: v[idx] for k, v in recon_results[seq_name][cam_id].items()}
                recon_smpl, recon_object = get_recon_meshes(metadata, img_id, smpl_male, results_per_img)
                gt_smpl, gt_object = get_gt_meshes(metadata, img_id, smpl_male, smpl_female, gt_annotations)

                recon_smpl_seq.append(recon_smpl)
                recon_object_seq.append(recon_object)
                gt_smpl_seq.append(gt_smpl)
                gt_object_seq.append(gt_object)

            # smpl_error, object_error = hoi_evaluator.compute_errors([gt_smpl_seq, gt_object_seq], [recon_smpl_seq, recon_object_seq])
            hoi_smpl_error, hoi_obj_error = hoi_evaluator_aligned.compute_errors([gt_smpl_seq, gt_object_seq], [recon_smpl_seq, recon_object_seq])
            hoi_smpl_error_win10, hoi_obj_error_win10 = hoi_evaluator_aligned_win10.compute_errors([gt_smpl_seq, gt_object_seq], [recon_smpl_seq, recon_object_seq])
            smpl_aligned_error, _ = smpl_evaluator_aligned.compute_errors([gt_smpl_seq, gt_object_seq], [recon_smpl_seq, recon_object_seq])

            print(np.mean(smpl_aligned_error), np.mean(hoi_smpl_error_win10), np.mean(hoi_obj_error_win10))

            evaluate_results[seq_name][cam_id] = {
                'img_ids': recon_results[seq_name][cam_id]['img_ids'],
                'hoi_smpl_avg_error': np.mean(hoi_smpl_error),
                'hoi_obj_avg_error': np.mean(hoi_obj_error),
                'hoi_smpl_avg_error_win10': np.mean(hoi_smpl_error_win10),
                'hoi_obj_avg_error_win10': np.mean(hoi_obj_error_win10),
                'hoi_smpl_error': hoi_smpl_error,
                'hoi_obj_error': hoi_obj_error,
                'hoi_smpl_error_win10': hoi_smpl_error_win10,
                'hoi_obj_error_win10': hoi_obj_error_win10,
            }
            print('sequence: {}, cam_id: {}, hoi_smpl: {:.4f}, hoi_obj: {:.4f}'.format(seq_name, cam_id,
                evaluate_results[seq_name][cam_id]['hoi_smpl_avg_error'], evaluate_results[seq_name][cam_id]['hoi_obj_avg_error']) )

    save_json(evaluate_results, os.path.join(output_dir, 'evaluate_results_sequence_optim_alpha_1_{}.json'.format(args.object)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root_dir', default='/storage/data/huochf/BEHAVE', type=str)
    parser.add_argument('--object', default='backpack', type=str)
    parser.add_argument('--batch_size', default=32, type=int)
    args = parser.parse_args()

    set_seed(7)
    evaluate_sequence_on_behave(args)
