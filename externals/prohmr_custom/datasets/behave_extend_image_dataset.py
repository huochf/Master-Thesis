import sys
sys.path.append('..')
import numpy as np
import cv2
import torch
from prohmr.datasets.utils import generate_image_patch, convert_cvimg_to_tensor

from hoi_recon.datasets.utils import load_pickle


class HumanImageDataset:

    def __init__(self, cfg, seq_name, cam_id, img_ids, metadata):
        self.img_ids = img_ids
        self.metadata = metadata
        self.cfg = cfg
        self.img_size = cfg.MODEL.IMAGE_SIZE
        self.mean = 255. * np.array(self.cfg.MODEL.IMAGE_MEAN)
        self.std = 255. * np.array(self.cfg.MODEL.IMAGE_STD)

        annotations = load_pickle('data/datasets/behave_extend_datalist/{}.pkl'.format(seq_name))
        imgid_to_annos = {}
        for item in annotations[cam_id]:
            imgid_to_annos[item['img_id']] = {
                'person_bb_xyxy': item['person_bb_xyxy'],
                'cam_K': item['cam_K']
            }
        self.imgid_to_annos = imgid_to_annos


    def __len__(self, ):
        return len(self.img_ids)


    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        aug_config = self.cfg.DATASETS.CONFIG

        image_file = self.metadata.get_image_path(img_id)
        person_bbox = self.imgid_to_annos[img_id]['person_bb_xyxy']
        x1, y1, x2, y2 = person_bbox
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        s = 1.2 * max(x2 - x1, y2 - y1)

        cvimg = cv2.imread(image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        image, _ = generate_image_patch(cvimg, cx, cy, s, s, self.img_size, self.img_size, False, 1.0, 0.)
        image = image[:, :, ::-1]
        img_patch = convert_cvimg_to_tensor(image)

        for n_c in range(3):
            img_patch[n_c, :, :] = np.clip(img_patch[n_c, :, :], 0, 255)
            img_patch[n_c, :, :] = (img_patch[n_c, :, :] - self.mean[n_c]) / self.std[n_c]

        crop_bbox = np.array([cx, cy, s, s]).astype(np.float32)
        cam_K = self.imgid_to_annos[img_id]['cam_K'].astype(np.float32)
        return img_patch.astype(np.float32), crop_bbox, cam_K
