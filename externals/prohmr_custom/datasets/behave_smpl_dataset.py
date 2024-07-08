import sys
sys.path.append('..')
import numpy as np
import torch
import pytorch_lightning as pl

from prohmr.datasets.utils import get_example

from hoi_recon.datasets.behave_metadata import BEHAVEMetaData
from hoi_recon.datasets.utils import load_pickle

class BEHAVESMPLDataset(pl.LightningDataModule):

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

        self.metadata = BEHAVEMetaData(cfg.DATASETS.CONFIG.ROOT_DIR)
        self.annotations = load_pickle('data/datasets/behave_datalist/behave_{}_list.pkl'.format('train' if train else 'test'))
        self.img_ids = list(self.annotations.keys())
        self.train = train
        self.cfg = cfg
        self.img_size = cfg.MODEL.IMAGE_SIZE
        self.mean = 255. * np.array(self.cfg.MODEL.IMAGE_MEAN)
        self.std = 255. * np.array(self.cfg.MODEL.IMAGE_STD)

        self.flip_keypoint_permutation = [0, 2, 1, 3, 5, 4, 6, 8, 7, 9, 11, 10, 12, 14, 13, 15, 17, 16, 19, 18, 21, 20]


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

        tx = np.clip(np.random.randn(), -1.0, 1.0) * 0.3
        ty = np.clip(np.random.randn(), -1.0, 1.0) * 0.3
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

        img_patch, keypoints_2d, keypoints_3d, smpl_params, has_smpl_params, img_size = get_example(image_file, 
            cx, cy, s, s, keypoints_2d, keypoints_3d, smpl_params, has_smpl_params, 
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
        item['cam_t'] = annotation['hoi_trans'].astype(np.float32)
        return item

