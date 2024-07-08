import sys
sys.path.append('..')
import numpy as np
import random
import cv2
from typing import List, Dict, Tuple
from yacs.config import CfgNode
from tqdm import tqdm
import torch
import pytorch_lightning as pl

from prohmr.datasets.utils import (
    do_augmentation, 
    extreme_cropping, 
    keypoint_3d_processing, 
    generate_image_patch, 
    convert_cvimg_to_tensor, 
    smpl_param_processing,
    fliplr_keypoints,
    trans_point2d,
)

from hoi_recon.datasets.behave_extend_metadata import BEHAVEExtendMetaData
from hoi_recon.datasets.utils import load_pickle


def get_rotmat_from_angle(rot):
    rot_mat = np.eye(3)
    if not rot == 0:
        rot_rad = rot * np.pi / 180
        sn, cs = np.sin(rot_rad), np.cos(rot_rad)
        rot_mat[0, :2] = [cs, -sn]
        rot_mat[1, :2] = [sn, cs]
    return rot_mat

    
def get_example(img_path: str, center_x: float, center_y: float,
                width: float, height: float,
                keypoints_2d: np.array, keypoints_3d: np.array,
                smpl_params: Dict, has_smpl_params: Dict, hoi_trans, cam_K,
                flip_kp_permutation: List[int],
                patch_width: int, patch_height: int,
                mean: np.array, std: np.array,
                do_augment: bool, augm_config: CfgNode) -> Tuple:

    cvimg = cv2.imread(img_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if not isinstance(cvimg, np.ndarray):
        raise IOError("Fail to read %s" % img_path)
    img_height, img_width, img_channels = cvimg.shape

    img_size = np.array([img_height, img_width])

    # 2. get augmentation params
    if do_augment:
        scale, rot, do_flip, do_extreme_crop, color_scale, tx, ty = do_augmentation(augm_config)
    else:
        scale, rot, do_flip, do_extreme_crop, color_scale, tx, ty = 1.0, 0, False, False, [1.0, 1.0, 1.0], 0., 0.

    if do_extreme_crop:
        center_x, center_y, width, height = extreme_cropping(center_x, center_y, width, height, keypoints_2d)
    center_x += width * tx
    center_y += height * ty

    # Process 3D keypoints
    keypoints_3d = keypoint_3d_processing(keypoints_3d, flip_kp_permutation, rot, do_flip)

    # 3. generate image patch
    img_patch_cv, trans = generate_image_patch(cvimg,
                                               center_x, center_y,
                                               width, height,
                                               patch_width, patch_height,
                                               do_flip, scale, rot)
    width = width * scale
    height = height * scale

    image = img_patch_cv.copy()
    image = image[:, :, ::-1]
    img_patch_cv = image.copy()
    img_patch = convert_cvimg_to_tensor(image)

    smpl_params, has_smpl_params = smpl_param_processing(smpl_params, has_smpl_params, rot, do_flip)

    # apply normalization
    for n_c in range(img_channels):
        img_patch[n_c, :, :] = np.clip(img_patch[n_c, :, :] * color_scale[n_c], 0, 255)
        if mean is not None and std is not None:
            img_patch[n_c, :, :] = (img_patch[n_c, :, :] - mean[n_c]) / std[n_c]
    if do_flip:
        keypoints_2d = fliplr_keypoints(keypoints_2d, img_width, flip_kp_permutation)

    for n_jt in range(len(keypoints_2d)):
        keypoints_2d[n_jt, 0:2] = trans_point2d(keypoints_2d[n_jt, 0:2], trans)
    keypoints_2d[:, :-1] = keypoints_2d[:, :-1] / patch_width - 0.5

    rotmat = get_rotmat_from_angle(- rot)
    optical_cx, optical_cy, fx, fy = cam_K[0, 2], cam_K[1, 2], cam_K[0, 0], cam_K[1, 1]
    trans_uv = hoi_trans[:2] / hoi_trans[2:] * np.array([fx, fy]) + np.array([optical_cx, optical_cy])
    trans_uv = trans_uv - np.array([center_x, center_y])
    if do_flip:
        trans_uv[0] *= -1
    trans_uv = (rotmat[:2, :2] @ trans_uv.reshape(2, 1)).reshape(2, ) + np.array([center_x, center_y])
    trans_xy = (trans_uv - np.array([optical_cx, optical_cy])) / np.array([fx, fy]) * hoi_trans[2:]
    hoi_trans = np.concatenate([trans_xy, hoi_trans[2:]], axis=0)

    return img_patch, keypoints_2d, keypoints_3d, smpl_params, has_smpl_params, hoi_trans, center_x, center_y, width, img_size


class BEHAVEExtendSMPLDataset(pl.LightningDataModule):

    def __init__(self, cfg):
        self.cfg = cfg
        self.prepare_data_per_node = True
        self._log_hyperparams = False
        self.allow_zero_length_dataloader_with_multiple_devices = False

    def setup(self, stage):
        self.train_dataset = Dataset(self.cfg, train=True)
        self.val_dataset = Dataset(self.cfg, train=False)


    def train_dataloader(self):
        train_dataloader = torch.utils.data.DataLoader(self.train_dataset, self.cfg.TRAIN.BATCH_SIZE, shuffle=True, drop_last=True, num_workers=self.cfg.GENERAL.NUM_WORKERS)
        return train_dataloader


    def val_dataloader(self):
        val_dataloader = torch.utils.data.DataLoader(self.val_dataset, self.cfg.TRAIN.BATCH_SIZE, shuffle=True, drop_last=True, num_workers=self.cfg.GENERAL.NUM_WORKERS)
        return val_dataloader



class Dataset():

    def __init__(self, cfg, train: bool = False):

        self.metadata = BEHAVEExtendMetaData(cfg.DATASETS.CONFIG.ROOT_DIR)
        print('Loading annotations ...')
        self.annotations = self.load_annotations(cfg, self.metadata, train)
        self.img_ids = list(self.annotations.keys())
        print('Loaded {} frames.'.format(len(self.img_ids)))
        self.train = train
        self.cfg = cfg
        self.img_size = cfg.MODEL.IMAGE_SIZE
        self.mean = 255. * np.array(self.cfg.MODEL.IMAGE_MEAN)
        self.std = 255. * np.array(self.cfg.MODEL.IMAGE_STD)

        self.flip_keypoint_permutation = [0, 2, 1, 3, 5, 4, 6, 8, 7, 9, 11, 10, 12, 14, 13, 15, 17, 16, 19, 18, 21, 20]


    def load_annotations(self, cfg, metadata, train):
        sequence_names = list(metadata.go_through_all_sequences(split='train' if train else 'test'))
        all_annotations = {}
        for seq_name in tqdm(sequence_names):
            annotations = load_pickle('../data/datasets/behave_extend_datalist/{}.pkl'.format(seq_name))
            for cam_id in annotations:
                for item in annotations[cam_id]:
                    img_id = item['img_id']
                    all_annotations[img_id] = {
                        'person_bb_xyxy': item['person_bb_xyxy'],
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


    def __getitem__(self, idx: int):
        try:
            item = self.getitem(idx)
        except:
            item = self.__getitem__(np.random.randint(len(self)))

        if item['roi_size'][0] == 0:
            item = self.__getitem__(np.random.randint(len(self)))

        return item


    def getitem(self, idx: int):
        img_id = self.img_ids[idx]
        aug_config = self.cfg.DATASETS.CONFIG

        image_file = self.metadata.get_image_path(img_id)
        annotation = self.annotations[img_id]
        person_bbox = annotation['person_bb_xyxy']
        x1, y1, x2, y2 = person_bbox
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        s = 1.2 * max(x2 - x1, y2 - y1)
        s = s * (np.clip(np.random.randn(), -1.0, 1.0) * 0.3 + 1.0)

        tx = np.clip(np.random.randn(), -1.0, 1.0) * 0.1
        ty = np.clip(np.random.randn(), -1.0, 1.0) * 0.1
        cx += s * tx
        cy += s * ty

        smpl_params = {
            'global_orient': annotation['smplh_theta'][:3].astype(np.float32),
            'body_pose': np.concatenate([annotation['smplh_theta'][3:66], np.zeros(6)]).astype(np.float32), # padding with ones, SMPLH -> SMPL
            'betas': annotation['smplh_betas_male'].astype(np.float32),
        }
        has_smpl_params = {
            'global_orient': np.array(1).astype(np.float32),
            'body_pose': np.array(1).astype(np.float32),
            'betas': np.array(1).astype(np.float32),
        }
        smpl_params_is_axis_angle = {
            'global_orient': True,
            'body_pose': True,
            'betas': False,
        }
        keypoints_2d = np.concatenate([annotation['smplh_joints_2d'][:22], np.ones([22, 1])], axis=1)
        keypoints_3d = annotation['smplh_joints_3d']
        keypoints_3d = keypoints_3d - keypoints_3d[:1]
        keypoints_3d = np.concatenate([keypoints_3d[:22], np.ones([22, 1])], axis=1)

        img_patch, keypoints_2d, keypoints_3d, smpl_params, has_smpl_params, hoi_trans, cx, cy, s, img_size = get_example(image_file, 
            cx, cy, s, s, keypoints_2d, keypoints_3d, smpl_params, has_smpl_params, annotation['hoi_trans'].copy(), annotation['cam_K'],
            self.flip_keypoint_permutation, self.img_size, self.img_size, self.mean, self.std, self.train, aug_config)

        focal_length = np.array([annotation['cam_K'][0, 0], annotation['cam_K'][1, 1]]) 
        roi_center = np.array([cx, cy])
        roi_size = np.array([s, s])

        item = {}
        item['img'] = img_patch.astype(np.float32)
        item['keypoints_2d'] = keypoints_2d.astype(np.float32)
        item['keypoints_3d'] = keypoints_3d.astype(np.float32)
        item['smpl_params'] = smpl_params
        item['has_smpl_params'] = has_smpl_params
        item['smpl_params_is_axis_angle'] = smpl_params_is_axis_angle
        item['K'] = annotation['cam_K'].astype(np.float32)
        item['focal_length'] = focal_length.astype(np.float32)
        item['roi_center'] = roi_center.astype(np.float32)
        item['roi_size'] = roi_size.astype(np.float32)
        item['cam_t'] = hoi_trans.astype(np.float32)
        return item
