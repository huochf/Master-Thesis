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

from hoi_recon.datasets.behave_extend_metadata import BEHAVEExtendMetaData
from hoi_recon.datasets.intercap_metadata import InterCapMetaData
from hoi_recon.datasets.utils import save_pickle, save_json, load_pickle
from hoi_recon.datasets.utils import perspective_projection


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


def fit_smplh_male(smplh_betas, smplh_pose_rotmat):
    b = smplh_betas.shape[0]
    device = torch.device('cuda')
    smpl_male = SMPLHLayer(model_path='data/models/smplh', gender='male').to(device)
    smpl_female = SMPLHLayer(model_path='data/models/smplh', gender='female').to(device)

    smplh_betas_male = nn.Parameter(smplh_betas.to(torch.float32).clone())
    smplh_betas = smplh_betas.to(torch.float32)
    smplh_pose_rotmat = smplh_pose_rotmat.to(torch.float32)
    global_orient = torch.eye(3, dtype=torch.float32, device=device).reshape(1, 3, 3).repeat(b, 1, 1)
    transl = torch.zeros((b, 3), dtype=torch.float32, device=device)

    optimizer = torch.optim.Adam([smplh_betas_male, ], lr=1e-1, betas=(0.9, 0.999))
    iterations = 2
    steps_per_iter = 100
    for it in range(iterations):
        loop = tqdm(range(steps_per_iter))
        for i in loop:
            optimizer.zero_grad()

            smpl_female_out = smpl_female(betas=smplh_betas, body_pose=smplh_pose_rotmat[:, 1:22], global_orient=global_orient, transl=transl)
            smpl_male_out = smpl_male(betas=smplh_betas_male, body_pose=smplh_pose_rotmat[:, 1:22], global_orient=global_orient, transl=transl)

            smpl_female_v = smpl_female_out.vertices
            smpl_male_v = smpl_male_out.vertices
            smpl_female_joints = smpl_female_out.joints
            smpl_male_joints = smpl_male_out.joints
            loss = F.l1_loss(smpl_female_v, smpl_male_v, reduction='none').reshape(b, -1).mean(-1) + F.l1_loss(smpl_female_joints, smpl_male_joints, reduction='none').reshape(b, -1).mean(-1)
            loss = loss.mean()

            loss.backward()
            optimizer.step()

            l_str = 'Iter: {}'.format(i)
            l_str += ', {}: {:0.4f}'.format('loss', loss.item())
            loop.set_description(l_str)
        for param_group in optimizer.param_groups:
            param_group["lr"] *= 0.01

    return smplh_betas_male.detach()


class BEHAVEExtendAnnotationDataset():

    def __init__(self, dataset_metadata, sequence_name, all_frames, annotations):
        self.dataset_metadata = dataset_metadata
        self.all_img_ids = all_frames
        print('total {} frames'.format(len(self.all_img_ids)))
        self.annotations = annotations
        self.sequence_name = sequence_name

    def __len__(self, ):
        return len(self.all_img_ids)


    def __getitem__(self, idx):
        img_id = self.all_img_ids[idx]

        day_id, sub_id, obj_name, inter_type, frame_id, cam_id = img_id.split('_')

        gender = self.dataset_metadata.get_sub_gender(img_id)

        person_mask = cv2.imread(self.dataset_metadata.get_person_mask_path(img_id, ), cv2.IMREAD_GRAYSCALE) / 255
        object_full_mask = cv2.imread(self.dataset_metadata.get_object_full_mask_path(img_id,), cv2.IMREAD_GRAYSCALE) / 255
        person_bb_xyxy = extract_bbox_from_mask(person_mask).astype(np.float32)
        object_bb_xyxy = extract_bbox_from_mask(object_full_mask).astype(np.float32)
        object_bb_xyxy *= 2
        hoi_bb_xyxy = np.concatenate([
            np.minimum(person_bb_xyxy[:2], object_bb_xyxy[:2]),
            np.maximum(person_bb_xyxy[2:], object_bb_xyxy[2:])
        ], axis=0).astype(np.float32)

        annotation = self.annotations['t0' + frame_id]
        obj_axis_angle = annotation['ob_pose']
        object_rotmat = R.from_rotvec(obj_axis_angle).as_matrix()
        object_trans = annotation['ob_trans']

        cam_R, cam_T = self.dataset_metadata.cam_RT_matrix[day_id][int(cam_id)]
        object_trans = np.matmul(cam_R.transpose(), object_trans - cam_T)
        object_rotmat = np.matmul(cam_R.transpose(), object_rotmat)

        annotation = self.annotations['t0' + frame_id]
        smplh_params = {k: v for k, v in annotation.items() if 'ob_' not in k}

        fx, fx, cx, cy = self.dataset_metadata.cam_intrinsics[int(cam_id)][:4]
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

        smplh_trans = smplh_params['trans']
        smplh_betas = smplh_params['betas']
        smplh_pose = smplh_params['poses'].copy()
        global_pose = smplh_pose[:3]
        global_pose_rotmat = R.from_rotvec(global_pose).as_matrix()
        global_pose_rotmat = np.matmul(cam_R.transpose(), global_pose_rotmat)
        global_pose = R.from_matrix(global_pose_rotmat).as_rotvec()
        smplh_pose[:3] = global_pose

        smplh_trans = np.matmul(cam_R.transpose(), smplh_trans - cam_T)

        obj_keypoints_3d = self.dataset_metadata.load_object_keypoints(obj_name)
        obj_keypoints_3d = np.matmul(obj_keypoints_3d, object_rotmat.T) + object_trans.reshape(1, 3)

        max_object_kps_num = self.dataset_metadata.object_max_keypoint_num
        object_kpts_3d_padded = np.ones((max_object_kps_num, 3), dtype=np.float32)
        obj_kps_num = obj_keypoints_3d.shape[0]
        object_kpts_3d_padded[:obj_kps_num, :] = obj_keypoints_3d

        return img_id, gender, person_bb_xyxy, object_bb_xyxy, hoi_bb_xyxy, smplh_trans, smplh_betas, smplh_pose, object_kpts_3d_padded, object_rotmat, object_trans, K


def collect_datalist_behave_extend(args):
    device = torch.device('cuda')

    smpl_male = SMPLHLayer(model_path='data/models/smplh', gender='male').to(device)
    smpl_female = SMPLHLayer(model_path='data/models/smplh', gender='female').to(device)

    behave_metadata = BEHAVEExtendMetaData(args.root_dir, preload_annotations=False)

    out_dir = './data/datasets/behave_extend_datalist'
    os.makedirs(out_dir, exist_ok=True)
    all_sequences = list(behave_metadata.go_through_all_sequences())
    all_img_id_per_sequences = behave_metadata.get_all_image_by_sequence()
    for sequence_name in tqdm(all_sequences):
        if os.path.exists(os.path.join(out_dir, '{}.pkl'.format(sequence_name))):
            continue

        annotation_list = {}

        if behave_metadata.SUBID_GENDER[behave_metadata.parse_seq_info(sequence_name)[1]] == 'male':
            smpl = smpl_male
        else:
            smpl = smpl_female

        for cam_id in range(4):
            annotation_list[cam_id] = []
            all_frames = all_img_id_per_sequences[sequence_name][cam_id]

            try:
                annotations = behave_metadata.load_annotations(sequence_name)
            except:
                # some sequences may lack of annotations
                print('Fail to load annotation for sequence {}.'.format(sequence_name))
                continue

            dataset = BEHAVEExtendAnnotationDataset(behave_metadata, sequence_name, all_frames, annotations)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, num_workers=8, shuffle=False, drop_last=False)

            for item in tqdm(dataloader, desc='Collect data for BEHAVE-Extend'):
                (img_id, gender, person_bb_xyxy, object_bb_xyxy, hoi_bb_xyxy, 
                    smplh_trans, smplh_betas, smplh_pose, obj_keypoints_3d, object_rotmat, object_trans, K) = item

                focal_length = torch.stack([K[:, 0, 0], K[:, 1, 1]], dim=1).to(device)
                optical_center = torch.stack([K[:, 0, 2], K[:, 1, 2]], dim=1).to(device)

                b = smplh_betas.shape[0]
                smplh_trans = smplh_trans.to(device)
                smplh_betas = smplh_betas.to(device)
                smplh_pose = smplh_pose.to(device)
                smplh_pose_rotmat = axis_angle_to_matrix(smplh_pose.reshape(b, -1, 3))

                if gender[0] == 'female':
                    smplh_betas_male = fit_smplh_male(smplh_betas, smplh_pose_rotmat) # this may repeat 4 times for 4 cameras ...
                else:
                    smplh_betas_male = smplh_betas

                smpl_out = smpl(betas=smplh_betas,
                                 body_pose=smplh_pose_rotmat[:, 1:22],
                                 global_orient=smplh_pose_rotmat[:, :1],
                                 transl=smplh_trans)
                joint_3d = smpl_out.joints.detach()
                joint_2d = perspective_projection(joint_3d, focal_length=focal_length, optical_center=optical_center)

                obj_keypoints_3d = obj_keypoints_3d.to(device)
                obj_keypoints_2d = perspective_projection(obj_keypoints_3d, focal_length=focal_length, optical_center=optical_center)

                hoi_rotmat = smplh_pose_rotmat[:, 0]
                hoi_trans = joint_3d[:, 0]

                object_rotmat = object_rotmat.to(device)
                object_trans = object_trans.to(device)
                object_rel_rotmat = torch.matmul(hoi_rotmat.transpose(2, 1), object_rotmat.float())
                object_rel_trans = torch.matmul(hoi_rotmat.transpose(2, 1), (object_trans.float() - hoi_trans).reshape(b, 3, 1)).reshape(b, 3)

                joint_3d = joint_3d.cpu().numpy()
                joint_2d = joint_2d.cpu().numpy()
                obj_keypoints_2d = obj_keypoints_2d.cpu().numpy()
                obj_keypoints_3d = obj_keypoints_3d.cpu().numpy()
                person_bb_xyxy = person_bb_xyxy.numpy()
                object_bb_xyxy = object_bb_xyxy.numpy()
                hoi_bb_xyxy = hoi_bb_xyxy.numpy()
                smplh_betas = smplh_betas.cpu().numpy()
                smplh_betas_male = smplh_betas_male.cpu().numpy()
                smplh_trans = smplh_trans.cpu().numpy()
                smplh_pose = smplh_pose.cpu().numpy()
                smplh_pose_rotmat = smplh_pose_rotmat.cpu().numpy()
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
                        'smplh_betas': smplh_betas[i], # (10, )
                        'smplh_betas_male': smplh_betas_male[i], # (10, )
                        'smplh_theta': smplh_pose[i], # (156, )
                        'smplh_pose_rotmat': smplh_pose_rotmat[i], # (52, 3, 3)
                        'smplh_trans': smplh_trans[i], # (3, )
                        'smplh_joints_3d': joint_3d[i], # (73, 3)
                        'smplh_joints_2d': joint_2d[i], # (73, 2)
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
    collect_datalist_behave_extend(args)


def temp(args):
    out_dir = './data/datasets/behave_extend_datalist'
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
            dist_coeffs = 0 * np.array(cam_intrinsics_params[4:])
            for item in annotations[cam_id]:
                img_id = item['img_id']
                item['cam_K'] = K

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
    parser.add_argument('--root_dir', default='/storage/data/huochf/BEHAVE', type=str, help='Dataset root directory.')
    parser.add_argument('--batch_size', default=128, type=int)
    args = parser.parse_args()

    # generate_datalist(args)
    temp(args)
