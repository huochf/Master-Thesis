import os
import numpy as np
import cv2
from tqdm import tqdm

from hoi_recon.datasets.behave_extend_metadata import BEHAVEExtendMetaData
from hoi_recon.datasets.utils import get_augmentation_params, generate_image_patch, load_pickle


class BEHAVEPseudoKps2DSeqDataset:

    def __init__(self, root_dir, object_name='backpack', window_radius=15, fps=30, use_gt_grouping=False, split='train'):

        self.root_dir = root_dir
        self.object_name = object_name
        self.window_radius = window_radius
        self.fps = fps
        self.frame_interval = 30 // fps
        self.split = split
        self.change_bg_ration = 0.5
        self.img_padding_ratio = 0.2
        self.img_size = 256
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        self.use_gt_grouping = use_gt_grouping
        if not use_gt_grouping:
            self.knn_grouping = load_pickle('outputs/behave_knn_grouping/knn_groups_{}_k_08_win_{:02d}_fps_{:02d}.pkl'.format(object_name, window_radius, fps))

        self.metadata = BEHAVEExtendMetaData(root_dir)

        self.item_ids, self.item_image_id, self.hoi_kps_seq_all, self.hoi_bb_xyxy = self.load_kps()
        print('loaded {} frames.'.format(len(self.item_ids)))

        bg_dir = '/storage/data/huochf/COCO2017/train2017'
        self.background_list = [os.path.join(bg_dir, f) for f in os.listdir(bg_dir)]


    def load_kps(self, ):
        datalist_dir = './data/datasets/behave_extend_datalist'
        hoi_kps_seq_all = {}
        item_ids = []
        hoi_bboxes = {}
        item_image_id = {}
        print('loading hoi kps ...')
        for file in os.listdir(datalist_dir):
            seq_name = file.split('.')[0]
            hoi_kps_seq_all[seq_name] = {}
            if seq_name not in self.metadata.dataset_splits[self.split]:
                continue
            object_name = seq_name.split('_')[2]
            if object_name != self.object_name:
                continue

            annotation_list = load_pickle(os.path.join(datalist_dir, file))
            for cam_id in annotation_list:
                hoi_kps_seq_all[seq_name][cam_id] = []
                for idx, item in enumerate(annotation_list[cam_id]):
                    hoi_kps_seq_all[seq_name][cam_id].append(self.kps3dfy(item))
                    item_id = '_'.join([seq_name, str(cam_id), str(idx)])
                    item_ids.append(item_id)
                    hoi_bboxes[item_id] = item['hoi_bb_xyxy']
                    item_image_id[item_id] = item['img_id']

        return item_ids, item_image_id, hoi_kps_seq_all, hoi_bboxes


    def kps3dfy(self, item):
        img_id = item['img_id']
        smplh_joints_2d = item['smplh_joints_2d'][:22]
        obj_keypoints_2d = item['obj_keypoints_2d'][:self.metadata.object_num_keypoints[self.object_name]]
        hoi_kps = np.concatenate([smplh_joints_2d, obj_keypoints_2d], axis=0)
        hoi_rotmat = item['hoi_rotmat']
        hoi_trans = item['hoi_trans']
        cam_K = item['cam_K']

        cx, cy, fx, fy = cam_K[0, 2], cam_K[1, 2], cam_K[0, 0], cam_K[1, 1]
        hoi_kps = (hoi_kps - np.array([cx, cy]).reshape((1, 2))) / np.array([fx, fy]).reshape((1, 2))
        z0 = 1
        n_kps = hoi_kps.shape[0]
        hoi_kps = np.concatenate([hoi_kps, np.ones((n_kps, 1)) * z0], axis=1)
        hoi_kps = np.concatenate([hoi_kps, np.zeros((1, 3))], axis=0)
        hoi_kps = hoi_kps - hoi_trans.reshape((1, 3))
        hoi_kps = hoi_kps @ hoi_rotmat # inverse matmul

        kps_directions = hoi_kps[:-1] - hoi_kps[-1:]
        kps_directions = kps_directions / (np.linalg.norm(kps_directions, axis=-1, keepdims=True) + 1e-8)
        hoi_kps[:-1] = kps_directions

        return hoi_kps


    def load_item_kps(self, item_id):
        if self.use_gt_grouping:
            seq_name = '_'.join(item_id.split('_')[:-2])
            cam_id = np.random.randint(4)
            frame_idx = int(item_id.split('_')[-1])
        else:
            groups = self.knn_grouping[item_id]
            n_groups = len(groups)
            group_idx = np.random.randint(n_groups + 1)
            if group_idx < n_groups:
                item_id = groups[group_idx][1]
            seq_name = '_'.join(item_id.split('_')[:-2])
            cam_id = int(item_id.split('_')[-2])
            frame_idx = int(item_id.split('_')[-1])

        kps_list = self.hoi_kps_seq_all[seq_name][cam_id]
        kps_prev = list(reversed(kps_list[:frame_idx + 1]))[::self.frame_interval]
        kps_prev = list(reversed(kps_prev))

        if len(kps_prev) > self.window_radius:
            kps_prev = kps_prev[- self.window_radius - 1:]
        else: # padding
            while len(kps_prev) <= self.window_radius:
                kps_prev.insert(0, kps_prev[0])

        kps_succ = kps_list[frame_idx:][::self.frame_interval]
        if len(kps_succ) > self.window_radius:
            kps_succ = kps_succ[: self.window_radius + 1]
        else: # padding
            while len(kps_succ) <= self.window_radius:
                kps_succ.append(kps_succ[-1])

        kps_seq = kps_prev[:-1] + kps_succ
        assert len(kps_seq) == 2 * self.window_radius + 1

        return np.stack(kps_seq, axis=0)


    def change_bg(self, image, img_id):
        # from CDPN (https://github.com/LZGMatrix/CDPN_ICCV2019_ZhigangLi)
        h, w, c = image.shape

        bg_num = len(self.background_list)
        idx = np.random.randint(0, bg_num - 1)
        bg_path = os.path.join(self.background_list[idx])
        bg_im = cv2.imread(bg_path, cv2.IMREAD_COLOR)
        bg_h, bg_w, bg_c = bg_im.shape
        real_hw_ratio = float(h) / float(w)
        bg_hw_ratio = float(bg_h) / float(bg_w)
        if real_hw_ratio <= bg_hw_ratio:
            crop_w = bg_w
            crop_h = int(bg_w * real_hw_ratio)
        else:
            crop_h = bg_h 
            crop_w = int(bg_h / bg_hw_ratio)
        bg_im = bg_im[:crop_h, :crop_w, :]
        bg_im = cv2.resize(bg_im, (w, h), interpolation=cv2.INTER_LINEAR)

        person_mask = cv2.imread(self.metadata.get_person_mask_path(img_id), cv2.IMREAD_GRAYSCALE) / 255
        object_full_mask_path = self.metadata.get_object_full_mask_path(img_id)
        object_full_mask = cv2.imread(object_full_mask_path, cv2.IMREAD_GRAYSCALE) / 255
        object_full_mask = cv2.resize(object_full_mask, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
        try:
            mask = (person_mask.astype(np.bool_) | object_full_mask.astype(np.bool_))
            bg_im[mask] = image[mask]
        except:
            bg_im = image

        return bg_im


    def __len__(self, ):
        return len(self.item_ids)


    def __getitem__(self, idx):
        try:
            return self.get_item(idx)
        except:
            idx = np.random.randint(len(self))
            return self.get_item(idx)
            

    def get_item(self, idx):
        item_id = self.item_ids[idx]
        img_id = self.item_image_id[item_id]
        hoi_bbox = self.hoi_bb_xyxy[item_id]

        image_path = self.metadata.get_image_path(img_id)
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)

        if (self.split == 'train') and (np.random.random() < self.change_bg_ration):
            image = self.change_bg(image, img_id)

        box_width, box_height = hoi_bbox[2:] - hoi_bbox[:2]
        box_size = max(box_width, box_height)
        box_size = box_size * (self.img_padding_ratio + 1)
        box_center_x, box_center_y = (hoi_bbox[2:] + hoi_bbox[:2]) / 2

        if self.split == 'train':
            tx, ty, rot, scale, color_scale = get_augmentation_params()
        else:
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

        hoi_kps = self.load_item_kps(item_id)
        pos_seq = np.arange(len(hoi_kps)).astype(np.int64)

        return img_patch.astype(np.float32), hoi_kps.astype(np.float32), pos_seq
