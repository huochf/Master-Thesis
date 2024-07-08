import os
from tqdm import tqdm
import numpy as np
import cv2
import random
from scipy.spatial.transform import Rotation

from hoi_recon.datasets.utils import generate_image_patch, get_augmentation_params, load_pickle
from hoi_recon.datasets.behave_extend_metadata import BEHAVEExtendMetaData, OBJECT_YZ_SYM_ROTMAT, OBJECT_KPS_PERM


class BEHAVE3DKpsDataset:

    def __init__(self, root_dir, split='train'):
        self.root_dir = root_dir
        self.split = split
        self.metadata = BEHAVEExtendMetaData(root_dir)
        kps3d_root_dir = '/inspurfs/group/wangjingya/huochf/datasets_hot_data/BEHAVE_extend/'
        self.kps3d_all = self.load_kps3d(kps3d_root_dir)
        self.img_list = list(self.kps3d_all.keys())
        print('Loaded {} frames.'.format(len(self.img_list)))

        self.object_kps = {}
        self.object_flip_rotmat = {}
        for obj_name in self.metadata.OBJECT_NAME2IDX.keys():
            self.object_kps[obj_name] = self.metadata.load_object_keypoints(obj_name)
            sym_rot_rotmat = OBJECT_YZ_SYM_ROTMAT[obj_name]
            sym_rot_axis = Rotation.from_matrix(sym_rot_rotmat).as_rotvec()
            sym_rot_axis[1] *= -1
            sym_rot_axis[2] *= -1
            sym_rot_rotmat_inv = Rotation.from_rotvec(sym_rot_axis).as_matrix().transpose(1, 0)
            self.object_flip_rotmat[obj_name] = np.matmul(sym_rot_rotmat_inv, sym_rot_rotmat)

        self.hoi_bb_xyxy = self.load_boxes()

        self.hoi_img_padding_ratio = 0.2
        self.img_size = 256
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]


    def load_kps3d(self, root_dir):
        seq_names = list(self.metadata.go_through_all_sequences(split=self.split))
        hoi_kps_all = {}
        print('loadding hoi keypoints ...')
        for seq_name in tqdm(seq_names):
            for file in os.listdir(os.path.join(root_dir, 'hoi_vertices', seq_name)):
                img_id = file[:-4]
                kps_file = os.path.join(root_dir, 'hoi_vertices', seq_name, file)
                hoi_kps = load_pickle(kps_file)

                for cam_id in range(4):
                    _img_id = img_id[:-1] + str(cam_id)
                    hoi_kps_all[_img_id] = {
                        'smpl_kps': hoi_kps['smpl_kps'],
                        'smpl_kps_sym': hoi_kps['smpl_kps_sym'],
                        'object_rel_rotmat': hoi_kps['object_rel_rotmat'],
                        'object_rel_trans': hoi_kps['object_rel_trans'],
                    }
        return hoi_kps_all


    def load_boxes(self, ):
        hoi_bboxes = {}
        seq_names = list(self.metadata.go_through_all_sequences(split=self.split))
        print('loadding hoi boxes ...')
        for seq_name in tqdm(seq_names):
            annotations = load_pickle('data/datasets/behave_extend_datalist/{}.pkl'.format(seq_name))
            for cam_id, item_list in annotations.items():
                for item in item_list:
                    hoi_bboxes[item['img_id']] = item['hoi_bb_xyxy']
        return hoi_bboxes


    def __len__(self, ):
        return len(self.img_list)


    def __getitem__(self, idx):
        img_id = self.img_list[idx]
        day_id, sub_id, obj_name, inter_type, frame_id, cam_id = img_id.split('_')
        object_label = self.metadata.OBJECT_NAME2IDX[obj_name]
        seq_name = self.metadata.get_sequence_name(day_id, sub_id, obj_name, inter_type)

        hoi_kps = self.kps3d_all[img_id]

        flip = np.random.random() < 0.5 and self.split == 'train'
        if not flip:
            smpl_kps = hoi_kps['smpl_kps'].copy() # [22, 3]
            smpl_joints_org = hoi_kps['smpl_kps'].copy()
        else:
            smpl_kps = hoi_kps['smpl_kps_sym'].copy()
            smpl_joints_org = hoi_kps['smpl_kps'].copy()

        object_rel_rotmat = hoi_kps['object_rel_rotmat'].copy()
        object_rel_trans = hoi_kps['object_rel_trans'].copy()

        object_rotmat = np.eye(3)
        if flip:
            object_rotmat = np.matmul(self.object_flip_rotmat[obj_name], object_rotmat)
            rel_rot_axis = Rotation.from_matrix(object_rel_rotmat).as_rotvec()
            rel_rot_axis[1] *= -1
            rel_rot_axis[2] *= -1
            rel_rotmat_flip = Rotation.from_rotvec(rel_rot_axis).as_matrix()
            object_rotmat = np.matmul(rel_rotmat_flip, object_rotmat)
            object_rel_trans[0] *= -1
        else:
            object_rotmat = np.matmul(object_rel_rotmat, object_rotmat)

        obj_sym_rotmat, obj_sym_trans = self.metadata.get_object_sym_RT(obj_name)
        object_rel_rotmat = np.matmul(object_rotmat, obj_sym_rotmat)
        object_rel_trans = np.matmul(object_rotmat, obj_sym_trans.reshape(3, 1)).reshape(3, ) + object_rel_trans

        object_kps = self.object_kps[obj_name].copy() # [n, 3]
        object_kps = np.matmul(object_kps, object_rel_rotmat.transpose(1, 0)) + object_rel_trans.reshape(1, 3)

        hoi_kps = np.concatenate([smpl_kps, object_kps], axis=0)
        hoi_kps_padded = np.zeros((22 + self.metadata.object_max_keypoint_num, 3))
        hoi_kps_padded[:hoi_kps.shape[0]] = hoi_kps

        image_path = self.metadata.get_image_path(img_id)
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)

        hoi_bb_xyxy = self.hoi_bb_xyxy[img_id]
        box_width, box_height = hoi_bb_xyxy[2:] - hoi_bb_xyxy[:2]
        box_size = max(box_width, box_height)
        box_size = box_size * (self.hoi_img_padding_ratio + 1)
        box_center_x, box_center_y = (hoi_bb_xyxy[2:] + hoi_bb_xyxy[:2]) / 2

        if self.split == 'train':
            tx, ty, rot, scale, color_scale = get_augmentation_params()
        else:
            tx, ty, rot, scale, color_scale = 0., 0., 0., 1., [1., 1., 1.]

        box_center_x += tx * box_width
        box_center_y += ty * box_width
        out_size = self.img_size
        box_size = box_size * scale

        img_patch, img_trans = generate_image_patch(image, box_center_x, box_center_y, box_size, out_size, rot, color_scale)
        if flip:
            img_patch = img_patch[:, ::-1]
        img_patch = img_patch[:, :, ::-1].astype(np.float32) # convert to RGB
        img_patch = img_patch.transpose((2, 0, 1))
        img_patch = img_patch / 256

        for n_c in range(3):
            img_patch[n_c, :, :] = np.clip(img_patch[n_c, :, :] * color_scale[n_c], 0, 1)
            img_patch[n_c, :, :] = (img_patch[n_c, :, :] - self.mean[n_c]) / self.std[n_c]

        return img_id, img_patch, hoi_kps_padded, object_label
