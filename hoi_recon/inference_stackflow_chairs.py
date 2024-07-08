import os
import sys
file_dir = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(file_dir, '..', ))
import numpy as np
import argparse
import json
import cv2
import random
from tqdm import tqdm

import torch
torch.multiprocessing.set_sharing_strategy('file_system')
from hoi_recon.configs.stackflow_chairs import load_config
from hoi_recon.models.stackflow_for_chairs import Model
from hoi_recon.datasets.utils import load_pickle, save_pickle, generate_image_patch, get_augmentation_params


def to_device(batch, device):
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(device)
    return batch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class ImageDataset:

    def __init__(self, root_dir, split='test'):
        self.root_dir = root_dir
        self.split = split
        self.image_padding_ratio = 0.5
        self.img_size = 256
        self.person_bbox = self.load_bboxes()
        self.img_ids = list(self.person_bbox.keys())
        self.vertex_len, self.object_meshes = self.load_meshes()

        anchor_indices = load_pickle('data/datasets/chairs_anchor_indices_n32_128.pkl')
        self.object_indices = anchor_indices['object']['sphere_1k']
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]


    def load_bboxes(self, ):
        if self.split == 'train':
            annotations = load_pickle('data/datasets/chairs_train_annotations.pkl')
        else:
            annotations = load_pickle('data/datasets/chairs_test_annotations.pkl')
        person_bbox = {img_id: item['person_bb_xyxy'] for img_id, item in annotations.items()}
        
        return person_bbox


    def load_meshes(self, ):
        object_info = np.load(os.path.join(self.root_dir, 'AHOI_Data', 'AHOI_ROOT', 'Metas', 'object_info.npy'), allow_pickle=True)
        vertex_len = object_info.item()['vertex_len']
        object_ids = object_info.item()['object_ids']
        vertex_len = {id_: len_ for id_, len_ in zip(object_ids, vertex_len)}

        object_inter_mesh_dir = 'data/datasets/chairs/object_inter_shapes'
        object_inter_meshes = {}
        for file in os.listdir(object_inter_mesh_dir):
            object_id = int(file.split('.')[0])
            object_inter_meshes[object_id] = np.load(os.path.join(object_inter_mesh_dir, file))

        return vertex_len, object_inter_meshes


    def __len__(self,):
        return len(self.img_ids)


    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        bbox = self.person_bbox[img_id]

        object_id, seq_id, cam_id, frame_id = img_id.split('_')

        image_path = os.path.join(self.root_dir, 'rimage', seq_id, cam_id, 'rgb_{}_{}_{}.jpg'.format(seq_id, cam_id, frame_id))
        if not os.path.exists(image_path):
            image_path = os.path.join(self.root_dir, 'rimage_extra', seq_id, cam_id, 'rgb_{}_{}_{}.jpg'.format(seq_id, cam_id, frame_id))
        assert os.path.exists(image_path), image_path
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)

        box_width, box_height = bbox[2:] - bbox[:2]
        box_size = max(box_width, box_height)
        box_size = box_size * (self.image_padding_ratio + 1)
        box_center_x, box_center_y = (bbox[2:] + bbox[:2]) / 2

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

        vertex_len = self.vertex_len[int(object_id)]
        object_vertices_all_parts = self.object_meshes[int(object_id)]
        part_ids = []
        for part_id in np.nonzero(vertex_len)[0]:
            part_ids.append(part_id)
        part_id = np.random.choice(part_ids)
        object_anchors = object_vertices_all_parts[part_id][self.object_indices]

        results = {}
        results['img_id'] = img_id
        results['image'] = img_patch
        results['part_labels'] = part_id
        results['object_anchors'] = object_anchors.astype(np.float32)

        return results


def train(cfg):
    device = torch.device('cuda')

    dataset_root_dir = cfg.dataset.root_dir
    dataset = ImageDataset(dataset_root_dir, split='test')
    dataloader = torch.utils.data.DataLoader(dataset, 
                                                   batch_size=cfg.train.batch_size,
                                                   num_workers=cfg.train.num_workers,
                                                   shuffle=True,
                                                   drop_last=True)

    model = Model(cfg)
    model.to(device)
    model.eval()
    model.load_checkpoint(cfg.train.resume)

    results_all = {}
    for idx, batch in enumerate(tqdm(dataloader)):
        batch = to_device(batch, device)
        outputs = model.inference(batch)

        for idx, img_id in enumerate(batch['img_id']):
            results_all[img_id] = {
                'part_label': batch['part_labels'][idx].detach().cpu().numpy(),
                'pred_betas': outputs['pred_betas'][idx].detach().cpu().numpy(),
                'pred_pose6d': outputs['pred_pose6d'][idx].detach().cpu().numpy(),
                'pred_smpl_body_pose': outputs['pred_smpl_body_pose'][idx].detach().cpu().numpy(),
                'pred_obj_rel_R': outputs['pred_obj_rel_R'][idx].detach().cpu().numpy(),
                'pred_obj_rel_T': outputs['pred_obj_rel_T'][idx].detach().cpu().numpy(),
            }
        if len(results_all) > 100:
            break


    os.makedirs(cfg.train.output_dir, exist_ok=True)
    save_pickle(results_all, os.path.join(cfg.train.output_dir, 'inference_results.pkl'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root_dir', default='/storage/data/huochf/CHAIRS', type=str)
    args = parser.parse_args()

    cfg = load_config()
    cfg.dataset.root_dir = args.dataset_root_dir
    cfg.freeze()
    set_seed(7)
    train(cfg)
