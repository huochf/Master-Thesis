import os
import pickle
import trimesh
import numpy as np
import cv2
from scipy.spatial.transform import Rotation

from hoi_recon.datasets.utils import load_pickle, generate_image_patch, get_augmentation_params


NUM_OBJECT = 92
MESH_VERTICES_NUM_MAX = 2000
OBJECT_PARTS_NUM = 7

part_name2id = {
    'chair_head': 0,
    'chair_back': 1,
    'chair_arm_left': 2,
    'chair_arm_right': 3,
    'chair_seat': 4,
    'chair_base': 5,
    'footrest': 6,
}

train_object_ids = [17, 24, 25, 26, 30, 33, 44, 46, 48, 49, 
                    59, 60, 64, 68, 75, 81, 83, 85, 87, 96, 
                    98, 99, 103, 104, 110, 111, 116, 117, 118, 121, 
                    123, 130, 131, 141, 142, 143, 147, 151, 152, 154, 
                    156, 157, 158, 162, 166, 168, 171, 173, 176, 180, 
                    181]
test_object_ids = [15, 29, 36, 43, 45, 92, 109, 149, 172]


class CHAIRSHOIDataset:

    def __init__(self, root_dir, annotations=None, is_train=True):
        self.root_dir = root_dir
        self.is_train = is_train
        self.image_padding_ratio = 0.5
        self.img_size = 256

        if annotations is None:
            self.annotations = self.load_annotations()
        else:
            self.annotations = annotations
        self.vertex_len, self.object_meshes = self.load_meshes()

        anchor_indices = load_pickle('data/datasets/chairs_anchor_indices_n32_128.pkl')
        self.object_indices = anchor_indices['object']['sphere_2k']
        self.img_ids = list(self.annotations.keys())
        print('loadded {} items.'.format(len(self.img_ids)))

        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]


    def load_annotations(self, ):
        if self.is_train:
            return load_pickle('data/datasets/chairs_train_annotations.pkl')
        else:
            return load_pickle('data/datasets/chairs_test_annotations.pkl')


    def load_meshes(self, ):
        object_info = np.load(os.path.join(self.root_dir, 'AHOI_Data', 'AHOI_ROOT', 'Metas', 'object_info.npy'), allow_pickle=True)
        vertex_len = object_info.item()['vertex_len']
        object_ids = object_info.item()['object_ids']
        vertex_len = {id_: len_ for id_, len_ in zip(object_ids, vertex_len)}

        object_inter_mesh_dir = 'data/datasets/chairs/object_inter_shapes_2k_alpha_10000_no_PCA'
        object_inter_meshes = {}
        for file in os.listdir(object_inter_mesh_dir):
            if file.split('.')[-1] != 'npy':
                continue
            object_id = int(file.split('.')[0])
            object_inter_meshes[object_id] = np.load(os.path.join(object_inter_mesh_dir, file))

        return vertex_len, object_inter_meshes


    def __len__(self, ):
        return len(self.img_ids)


    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        annotation = self.annotations[img_id]
        object_id, seq_id, cam_id, frame_id = img_id.split('_')

        image_path = os.path.join(self.root_dir, 'rimage', seq_id, cam_id, 'rgb_{}_{}_{}.jpg'.format(seq_id, cam_id, frame_id))
        if not os.path.exists(image_path):
            image_path = os.path.join(self.root_dir, 'rimage_extra', seq_id, cam_id, 'rgb_{}_{}_{}.jpg'.format(seq_id, cam_id, frame_id))
        assert os.path.exists(image_path), image_path
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)

        bbox = annotation['person_bb_xyxy']
        box_width, box_height = bbox[2:] - bbox[:2]
        box_size = max(box_width, box_height)
        box_size = box_size * (self.image_padding_ratio + 1)
        box_center_x, box_center_y = (bbox[2:] + bbox[:2]) / 2

        if self.is_train:
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

        object_vertices_all_parts = self.object_meshes[int(object_id)]
        object_anchors = object_vertices_all_parts[self.object_indices]

        results = {}
        results['img_id'] = img_id
        results['image'] = img_patch
        results['smpl_pose_rotmat'] = annotation['smplx_body_rotmat'].astype(np.float32)
        results['smpl_betas'] = annotation['smplx_betas'].astype(np.float32)
        results['object_rel_rotmat'] = annotation['object_rel_rotmat'].astype(np.float32)
        results['object_rel_trans'] = annotation['object_rel_trans'].astype(np.float32)
        results['object_anchors'] = object_anchors.astype(np.float32)
        results['smpler_betas'] = annotation['smplx_betas'].astype(np.float32) # annotation['smpler_betas'].reshape(-1).astype(np.float32)
        results['smpler_body_pose'] = annotation['smplx_body_theta'].astype(np.float32) # annotation['smpler_body_pose'][:21].reshape(-1).astype(np.float32)

        return results
