import os
import numpy as np
import cv2
from tqdm import tqdm
from scipy.spatial.transform import Rotation

from hoi_recon.datasets.utils import load_pickle, get_augmentation_params, generate_image_patch
from hoi_recon.datasets.behave_extend_metadata import BEHAVEExtendMetaData, OBJECT_YZ_SYM_ROTMAT


class BEHAVE3DKpsSeqDataset:

    def __init__(self, root_dir, object_name='backpack', window_radius=15, fps=30, split='train'):

        self.root_dir = root_dir
        self.object_name = object_name
        self.window_radius = window_radius
        self.fps = fps
        self.frame_interval = 30 // fps
        self.split = split
        self.metadata = BEHAVEExtendMetaData(root_dir)
        kps_root_dir = '/inspurfs/group/wangjingya/huochf/datasets_hot_data/BEHAVE_extend/'
        self.kps3d_seq_all = self.load_frames(kps_root_dir)
        self.seq_names = list(self.kps3d_seq_all.keys())
        print('Loaded {} sequences, {} frames.'.format(len(self.kps3d_seq_all), self.total_frames))

        self.object_kps = self.metadata.load_object_keypoints(object_name)
        sym_rot_rotmat = OBJECT_YZ_SYM_ROTMAT[object_name]
        sym_rot_axis = Rotation.from_matrix(sym_rot_rotmat).as_rotvec()
        sym_rot_axis[1] *= -1
        sym_rot_axis[2] *= -1
        sym_rot_rotmat_inv = Rotation.from_rotvec(sym_rot_axis).as_matrix().transpose(1, 0)
        self.object_flip_rotmat = np.matmul(sym_rot_rotmat_inv, sym_rot_rotmat)

        self.hoi_bb_xyxy = self.load_boxes()

        self.hoi_img_padding_ratio = 0.2
        self.img_size = 256
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]


    def load_frames(self, root_dir):
        seq_names = self.metadata.go_through_all_sequences(split=self.split)
        kps3d_seq_all = {}
        self.total_frames = 0
        for seq_name in seq_names:
            object_name = seq_name.split('_')[2]
            if object_name != self.object_name:
                continue
            kps3d_seq_all[seq_name] = []
            for file in sorted(os.listdir(os.path.join(root_dir, 'hoi_vertices_smoothed', seq_name))):
                img_id = file[:-4]
                kps_file = os.path.join(root_dir, 'hoi_vertices_smoothed', seq_name, file)
                hoi_kps = load_pickle(kps_file)

                kps3d_seq_all[seq_name].append((img_id, hoi_kps))
                self.total_frames += 1

        return kps3d_seq_all


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


    def load_item_kps_seq(self, seq_name, frame_idx):
        kps_seq = self.kps3d_seq_all[seq_name]
        kps_prev = list(reversed(kps_seq[:frame_idx + 1]))[::self.frame_interval]
        kps_prev = list(reversed(kps_prev))

        if len(kps_prev) > self.window_radius:
            kps_prev = kps_prev[ - self.window_radius - 1:]
        else: # padding
            while len(kps_prev) <= self.window_radius:
                kps_prev.insert(0, kps_prev[0])

        kps_succ = kps_seq[frame_idx:][::self.frame_interval]
        if len(kps_succ) > self.window_radius:
            kps_succ = kps_succ[:self.window_radius + 1]
        else: # padding
            while len(kps_succ) <= self.window_radius:
                kps_succ.append(kps_succ[-1])

        kps_seq = kps_prev[:-1] + kps_succ
        assert len(kps_seq) == 2 * self.window_radius + 1

        return kps_seq


    def __len__(self, ):
        return self.total_frames


    def __getitem__(self, idx):
        seq_idx = np.random.randint(len(self.seq_names))
        seq_name = self.seq_names[seq_idx]
        kps_seq = self.kps3d_seq_all[seq_name]
        frame_idx = np.random.randint(len(kps_seq))
        kps_list = self.load_item_kps_seq(seq_name, frame_idx)

        obj_sym_rotmat, obj_sym_trans = self.metadata.get_object_sym_RT(self.object_name)
        flip = np.random.random() < 0.5 and self.split == 'train'
        smpl_kps_seq, object_kps_seq = [], []

        for item in kps_list:
            smpl_kps, object_kps = self.load_kps(item[1], flip, obj_sym_rotmat, obj_sym_trans)

            smpl_kps_seq.append(smpl_kps)
            object_kps_seq.append(object_kps)

        image = self.load_image(kps_seq[frame_idx][0], flip)

        smpl_kps_seq = np.array(smpl_kps_seq).astype(np.float32)
        object_kps_seq = np.array(object_kps_seq).astype(np.float32)
        hoi_kps_seq = np.concatenate([smpl_kps_seq, object_kps_seq], axis=1)
        pos_seq = np.arange(len(smpl_kps_seq)).astype(np.int64)

        return image.astype(np.float32), hoi_kps_seq, pos_seq


    def load_kps(self, item, flip, obj_sym_rotmat, obj_sym_trans):

        if flip:
            smpl_kps = item['smpl_kps_sym'].copy()
            rot = item['smpl_orient_sym'].copy()
        else:
            smpl_kps = item['smpl_kps'].copy()
            rot = item['smpl_orient'].copy()

        object_rel_rotmat = item['object_rel_rotmat'].copy()
        object_rel_trans = item['object_rel_trans'].copy()

        object_rotmat = np.eye(3)
        if flip:
            object_rotmat = np.matmul(self.object_flip_rotmat, object_rotmat)
            rel_rot_axis = Rotation.from_matrix(object_rel_rotmat).as_rotvec()
            rel_rot_axis[1] *= -1
            rel_rot_axis[2] *= -1
            rel_rotmat_flip = Rotation.from_rotvec(rel_rot_axis).as_matrix()
            object_rotmat = np.matmul(rel_rotmat_flip, object_rotmat)
            object_rel_trans[0] *= -1
        else:
            object_rotmat = np.matmul(object_rel_rotmat, object_rotmat)
        object_rel_rotmat = np.matmul(object_rotmat, obj_sym_rotmat)
        object_rel_trans = np.matmul(object_rotmat, obj_sym_trans.reshape(3, 1)).reshape(3, ) + object_rel_trans

        object_kps = self.object_kps # [n, 3]
        object_kps = np.matmul(object_kps, object_rel_rotmat.transpose(1, 0)) + object_rel_trans.reshape(1, 3)

        return smpl_kps, object_kps


    def load_image(self, img_id, flip):

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
            img_patch[n_c, :, :] = np.clip(img_patch[n_c, :, :] * color_scale[n_c], 0, 255)
            img_patch[n_c, :, :] = (img_patch[n_c, :, :] - self.mean[n_c]) / self.std[n_c]

        return img_patch


    def get_pos_embedding(self, seq_indices, pos_dim, temperature=10000, normalize=True, scale=None):
        pos = seq_indices
        if scale is None:
            scale = 2 * np.pi
        if normalize:
            eps = 1e-6
            pos = pos / (pos[-1] + eps) * scale
        dim_t = np.arange(pos_dim)
        dim_t = temperature ** (2 * (dim_t // 2) / pos_dim)
        pos = pos[:, np.newaxis] / dim_t[np.newaxis, :]

        pos = np.stack([np.sin(pos[:, 0::2]), np.cos(pos[:, 1::2])], axis=2).reshape(-1, pos_dim)
        return pos
