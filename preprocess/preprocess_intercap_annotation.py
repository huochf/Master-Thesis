import os
import sys
file_dir = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(file_dir, '..', ))
from tqdm import tqdm
import cv2
import argparse
import random
import numpy as np
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as R
from pytorch3d.transforms import axis_angle_to_matrix
from smplx import SMPLH, SMPLX, SMPLHLayer, SMPLXLayer

from transflow.datasets.behave_metadata import BEHAVEMetaData
from transflow.datasets.behave_extend_metadata import BEHAVEExtendMetaData
from transflow.datasets.intercap_metadata import InterCapMetaData
from transflow.datasets.utils import save_pickle, save_json, load_pickle
from transflow.datasets.utils import perspective_projection


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


def fit_smplx_neutral(smplx_betas, smplx_pose_rotmat, smplx):
    b = smplx_betas.shape[0]
    device = torch.device('cuda')
    smpl_neutral = SMPLXLayer(model_path='data/models/smplx', gender='neutral').to(device)

    smplx_betas_neutral = nn.Parameter(smplx_betas.to(torch.float32).clone())
    smplx_betas = smplx_betas.to(torch.float32)
    smplx_pose_rotmat = smplx_pose_rotmat[:, :22].to(torch.float32)
    global_orient = torch.eye(3, dtype=torch.float32, device=device).repeat(b, 1, 1)
    transl = torch.zeros((b, 3), dtype=torch.float32, device=device)

    optimizer = torch.optim.Adam([smplx_betas_neutral, ], lr=1e-1, betas=(0.9, 0.999))
    iterations = 2
    steps_per_iter = 100
    for it in range(iterations):
        loop = tqdm(range(steps_per_iter))
        for i in loop:
            optimizer.zero_grad()

            smpl_neutral_out = smpl_neutral(betas=smplx_betas_neutral, body_pose=smplx_pose_rotmat[:, 1:22], global_orient=global_orient, transl=transl)
            smpl_neutral_v = smpl_neutral_out.vertices
            smpl_neutral_joints = smpl_neutral_out.joints

            smpl_out = smplx(betas=smplx_betas, body_pose=smplx_pose_rotmat[:, 1:22], global_orient=global_orient, transl=transl)
            smpl_v = smpl_out.vertices
            smpl_joints = smpl_out.joints
            loss = F.l1_loss(smpl_v, smpl_neutral_v, reduction='mean') + F.l1_loss(smpl_joints, smpl_neutral_joints, reduction='mean')
            loss.backward()
            torch.nn.utils.clip_grad_norm_([smplx_betas_neutral, ], 1)
            optimizer.step()

            l_str = 'Iter: {}'.format(i)
            l_str += ', {}: {:0.4f}'.format('loss', loss.item())
            loop.set_description(l_str)
        for param_group in optimizer.param_groups:
            param_group["lr"] *= 0.1

    return smplx_betas_neutral.detach()


class InterCapAnnotationDataset():

    def __init__(self, dataset_metadata, sequence_name, all_frames, annotations):
        self.dataset_metadata = dataset_metadata
        self.sequence_name = sequence_name
        self.cam_calibration = dataset_metadata.load_cam_calibration()
        self.all_img_ids = all_frames
        if sequence_name == '04_07_0':
            self.all_img_ids = self.all_img_ids[:len(annotations)]
        print('total {} frames'.format(len(self.all_img_ids)))
        self.annotations = annotations

        assert len(self.all_img_ids) == len(self.annotations), sequence_name


    def __len__(self, ):
        return len(self.all_img_ids)


    def __getitem__(self, idx):
        img_id = self.all_img_ids[idx]

        sub_id, obj_id, seq_id, cam_id, frame_id = self.dataset_metadata.parse_img_id(img_id)
        seq_name = '_'.join([sub_id, obj_id, seq_id])
        obj_name = self.dataset_metadata.OBJECT_IDX2NAME[obj_id]

        gender = self.dataset_metadata.SUBID_GENDER[sub_id]

        person_mask = cv2.imread(self.dataset_metadata.get_person_mask_path(img_id), cv2.IMREAD_GRAYSCALE) / 255
        try:
            object_full_mask = cv2.imread(self.dataset_metadata.get_object_full_mask_path(img_id), cv2.IMREAD_GRAYSCALE) / 255
        except:
            # some anno may not exists
            h, w = person_mask.shape
            object_full_mask = np.zeros((h, w, 3))
        person_bb_xyxy = extract_bbox_from_mask(person_mask).astype(np.float32)
        object_bb_xyxy = extract_bbox_from_mask(object_full_mask).astype(np.float32)
        object_bb_xyxy *= 2
        hoi_bb_xyxy = np.concatenate([
            np.minimum(person_bb_xyxy[:2], object_bb_xyxy[:2]),
            np.maximum(person_bb_xyxy[2:], object_bb_xyxy[2:])
        ], axis=0).astype(np.float32)

        annotation = self.annotations[int(frame_id)]
        obj_axis_angle = annotation['ob_pose']
        object_rotmat = R.from_rotvec(obj_axis_angle).as_matrix()
        object_trans = annotation['ob_trans']

        calitration = self.cam_calibration[cam_id]
        cam_R = np.array(calitration['R'])
        cam_R = R.from_rotvec(cam_R).as_matrix()
        cam_T = np.array(calitration['T'])
        cx, cy = calitration['c']
        fx, fy = calitration['f']
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

        object_rotmat = np.matmul(cam_R, object_rotmat)
        object_trans = np.matmul(cam_R, object_trans) + cam_T

        annotation = self.annotations[int(frame_id)]
        smpl_params = {k: v for k, v in annotation.items() if 'ob_' not in k}

        smplx_betas = smpl_params['betas'].reshape(10, )
        smplx_global_pose = smpl_params['global_orient'].reshape(3, )
        smplx_global_rotmat = R.from_rotvec(smplx_global_pose).as_matrix()
        smplx_trans = smpl_params['transl'].reshape(3, )
        smplx_body_pose = smpl_params['body_pose'].reshape((21, 3))

        smplx_global_rotmat = np.matmul(cam_R, smplx_global_rotmat)
        smplx_global_pose = R.from_matrix(smplx_global_rotmat).as_rotvec()
        smplx_pose = np.concatenate([smplx_global_pose, smplx_body_pose.reshape(-1)], axis=0) # (66, )
        smplx_trans = np.matmul(cam_R, smplx_trans) + cam_T

        obj_keypoints_3d = self.dataset_metadata.load_object_keypoints(obj_name)
        obj_keypoints_3d = np.matmul(obj_keypoints_3d, object_rotmat.T) + object_trans.reshape(1, 3)

        max_object_kps_num = self.dataset_metadata.object_max_keypoint_num
        object_kpts_3d_padded = np.ones((max_object_kps_num, 3), dtype=np.float32)
        obj_kps_num = obj_keypoints_3d.shape[0]
        object_kpts_3d_padded[:obj_kps_num, :] = obj_keypoints_3d

        return img_id, gender, person_bb_xyxy, object_bb_xyxy, hoi_bb_xyxy, smplx_betas, smplx_trans, smplx_pose, cam_R, cam_T, object_kpts_3d_padded, object_rotmat, object_trans, K


def collect_datalist_intercap(args):
    device = torch.device('cuda')

    intercap_metadata = InterCapMetaData(args.root_dir)
    smpl_male = SMPLXLayer(model_path='data/models/smplx', gender='male').to(device)
    smpl_female = SMPLXLayer(model_path='data/models/smplx', gender='female').to(device)

    out_dir = './data/datasets/intercap_datalist'
    os.makedirs(out_dir, exist_ok=True)
    all_sequences = list(intercap_metadata.go_through_all_sequences())
    all_img_id_per_sequences = intercap_metadata.get_all_image_by_sequence()

    for sequence_name in tqdm(all_sequences):
        if os.path.exists(os.path.join(out_dir, '{}.pkl'.format(sequence_name))):
            continue

        annotation_list = {}
        all_img_id_per_cam = all_img_id_per_sequences[sequence_name]

        for cam_id, all_frames in all_img_id_per_cam.items():
            annotation_list[cam_id] = []
            if intercap_metadata.SUBID_GENDER[sequence_name.split('_')[0]] == 'male':
                smpl = smpl_male
            else:
                smpl = smpl_female

            annotations = intercap_metadata.load_annotations(sequence_name)

            dataset = InterCapAnnotationDataset(intercap_metadata, sequence_name, all_frames, annotations)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, num_workers=8, shuffle=False, drop_last=False)

            for item in tqdm(dataloader, desc='Preprocess Annotations (InterCap)'):
                img_id, gender, person_bb_xyxy, object_bb_xyxy, hoi_bb_xyxy, smplx_betas, smplx_trans, smplx_pose, cam_R, cam_T, obj_keypoints_3d, object_rotmat, object_trans, K = item

                focal_length = torch.stack([K[:, 0, 0], K[:, 1, 1]], dim=1).to(device)
                optical_center = torch.stack([K[:, 0, 2], K[:, 1, 2]], dim=1).to(device)

                b = smplx_betas.shape[0]
                smplx_betas = smplx_betas.float().to(device)
                smplx_trans = smplx_trans.float().to(device)
                smplx_pose = smplx_pose.float().to(device)
                smplx_pose_rotmat = axis_angle_to_matrix(smplx_pose.reshape(b, -1, 3))
                cam_R = cam_R.float().to(device)
                cam_T = cam_T.float().to(device)

                smplx_betas_neutral = fit_smplx_neutral(smplx_betas, smplx_pose_rotmat, smpl)
                smpl_out_org = smpl(betas=smplx_betas.to(device),
                                body_pose=smplx_pose_rotmat[:, 1:22].to(device),
                                global_orient=torch.eye(3, dtype=torch.float32, device=device).unsqueeze(0).repeat(b, 1, 1),
                                transl=0 * smplx_trans.to(device))
                J_0 = smpl_out_org.joints[:,0].detach().reshape(b, 3, 1)
                smplx_trans = smplx_trans + (torch.matmul(cam_R, J_0) - J_0).squeeze(-1)

                smpl_out = smpl(betas=smplx_betas.to(device),
                                body_pose=smplx_pose_rotmat[:, 1:22].to(device),
                                global_orient=smplx_pose_rotmat[:, :1].to(device),
                                transl=smplx_trans.to(device))
                joint_3d = smpl_out.joints.detach()

                joint_2d = perspective_projection(joint_3d, focal_length=focal_length, optical_center=optical_center)

                obj_keypoints_3d = obj_keypoints_3d.to(device)
                obj_keypoints_2d = perspective_projection(obj_keypoints_3d, focal_length=focal_length, optical_center=optical_center)

                hoi_rotmat = smplx_pose_rotmat[:, 0]
                hoi_trans = joint_3d[:, 0]

                object_rotmat = object_rotmat.to(device)
                object_trans = object_trans.to(device)
                object_rel_rotmat = torch.matmul(hoi_rotmat.transpose(2, 1), object_rotmat.float())
                object_rel_trans = torch.matmul(hoi_rotmat.transpose(2, 1), (object_trans.float() - hoi_trans).reshape(b, 3, 1)).reshape(b, 3, )

                joint_3d = joint_3d.cpu().numpy()
                joint_2d = joint_2d.cpu().numpy()
                obj_keypoints_2d = obj_keypoints_2d.cpu().numpy()
                obj_keypoints_3d = obj_keypoints_3d.cpu().numpy()
                person_bb_xyxy = person_bb_xyxy.numpy()
                object_bb_xyxy = object_bb_xyxy.numpy()
                hoi_bb_xyxy = hoi_bb_xyxy.numpy()
                smplx_betas = smplx_betas.cpu().numpy()
                smplx_betas_neutral = smplx_betas_neutral.cpu().numpy()
                smplx_trans = smplx_trans.cpu().numpy()
                smplx_pose = smplx_pose.cpu().numpy()
                smplx_pose_rotmat = smplx_pose_rotmat.cpu().numpy()
                object_rotmat = object_rotmat.cpu().numpy()
                object_trans = object_trans.cpu().numpy()
                object_rel_rotmat = object_rel_rotmat.cpu().numpy()
                object_rel_trans = object_rel_trans.cpu().numpy()
                hoi_trans = hoi_trans.cpu().numpy()
                hoi_rotmat = hoi_rotmat.cpu().numpy()
                K = K.numpy()
                for i in range(b):
                    annotation_list[cam_id].append({
                        'img_id': img_id[i],
                        'gender': gender[i],
                        'person_bb_xyxy': person_bb_xyxy[i], # (4, )
                        'object_bb_xyxy': object_bb_xyxy[i], # (4, )
                        'hoi_bb_xyxy': hoi_bb_xyxy[i], # (4, )
                        'smplx_betas': smplx_betas[i], # (10, )
                        'smplx_betas_neutral': smplx_betas_neutral[i], # (10, )
                        'smplh_theta': smplx_pose[i], # (66, )
                        'smplx_pose_rotmat': smplx_pose_rotmat[i], # (22, 3, 3)
                        'smplx_trans': smplx_trans[i], # (3, )
                        'smplx_joints_3d': joint_3d[i], # (73, 3)
                        'smplx_joints_2d': joint_2d[i], # (73, 2)
                        'obj_keypoints_3d': obj_keypoints_3d[i], # (-1, 3)
                        'obj_keypoints_2d': obj_keypoints_2d[i], # (-1, 2)
                        'object_trans': object_trans[i], # (3, )
                        'object_rotmat': object_rotmat[i], # (3, 3)
                        'object_rel_trans': object_rel_trans[i], # (3, )
                        'object_rel_rotmat': object_rel_rotmat[i], # (3, 3)
                        'hoi_trans': hoi_trans[i], # (3, )
                        'hoi_rotmat': hoi_rotmat[i], # (3, 3)
                        'cam_K': K[i], # (3, 3)
                    })
        save_pickle(annotation_list, os.path.join(out_dir, '{}.pkl'.format(sequence_name)))
        print('{} Done!'.format(sequence_name))


def generate_datalist(args):
    collect_datalist_intercap(args)


def temp(args):
    out_dir = './data/datasets/behave_extend_datalist2'
    os.makedirs(out_dir, exist_ok=True)
    metadata = BEHAVEExtendMetaData(args.root_dir, preload_annotations=False)
    for file in tqdm(os.listdir('./data/datasets/behave_extend_datalist')):
        annotations = load_pickle(os.path.join('./data/datasets/behave_extend_datalist', file))
        annotation_new = {}
        for cam_id in annotations:
            annotation_new[cam_id] = []
            cam_intrinsics_params = metadata.cam_intrinsics[int(cam_id)]
            K = np.eye(3)
            fx, fy, cx, cy = cam_intrinsics_params[:4]
            K[0, 0] = fx
            K[1, 1] = fy
            K[0, 2] = cx
            K[1, 2] = cy
            dist_coeffs = np.array(cam_intrinsics_params[4:])
            for item in annotations[cam_id]:
                img_id = item['img_id']

                smplh_joints_3d = item['smplh_joints_3d']
                obj_keypoints_3d = item['obj_keypoints_3d']

                smplh_joints_2d = cv2.projectPoints(smplh_joints_3d, np.zeros(3), np.zeros(3), K, dist_coeffs)[0].reshape(-1, 2)
                obj_keypoints_2d = cv2.projectPoints(obj_keypoints_3d, np.zeros(3), np.zeros(3), K, dist_coeffs)[0].reshape(-1, 2)

                item['smplh_joints_2d'] = smplh_joints_2d
                item['obj_keypoints_2d'] = obj_keypoints_2d
                annotation_new[cam_id].append(item)

        save_pickle(annotation_new, os.path.join(out_dir, file))
        print('{} Done!'.format(file))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', default='/storage/data/huochf/InterCap/', type=str, help='Dataset root directory.')
    parser.add_argument('--batch_size', default=128, type=int)
    args = parser.parse_args()

    # generate_datalist(args)
    temp(args)
