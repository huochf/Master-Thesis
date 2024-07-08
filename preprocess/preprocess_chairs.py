import os
import sys
file_dir = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(file_dir, '..', ))
from tqdm import tqdm
import cv2
import argparse
import pickle
import random
import numpy as np
from PIL import Image
import trimesh
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.transform import Rotation
# from pytorch3d.transforms import axis_angle_to_matrix
from smplx import SMPLH, SMPLX, SMPLHLayer, SMPLXLayer

from hoi_recon.datasets.chairs_hoi_template_dataset import CHAIRSDataset
from hoi_recon.datasets.utils import load_pickle, save_pickle
from hoi_recon.models.conv3d import Voxel3DEncoder
from hoi_recon.models.gconv import MeshDeformer
from hoi_recon.utils.sphere import Sphere


NUM_OBJECT = 92
OBJECT_PARTS_NUM = 7
MESH_VERTICES_NUM_MAX = 1000

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


def to_device(batch, device):
    results = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    return results


class CHAIRSDataset:

    def __init__(self, root_dir):
        self.root_dir = root_dir

        self.human_betas = np.load(os.path.join(root_dir, 'AHOI_Data', 'DATA_FOLDER', 'human_betas.npy'))
        self.human_orient = np.load(os.path.join(root_dir, 'AHOI_Data', 'DATA_FOLDER', 'human_orient.npy'))
        self.human_pose = np.load(os.path.join(root_dir, 'AHOI_Data', 'DATA_FOLDER', 'human_pose.npy'))
        self.human_transl = np.load(os.path.join(root_dir, 'AHOI_Data', 'DATA_FOLDER', 'human_transl.npy'))
        self.human_pose = np.load(os.path.join(root_dir, 'AHOI_Data', 'DATA_FOLDER', 'human_pose.npy'))

        self.object_ids = np.load(os.path.join(root_dir, 'AHOI_Data', 'DATA_FOLDER', 'object_id.npy'))
        self.object_location = np.load(os.path.join(root_dir, 'AHOI_Data', 'DATA_FOLDER', 'object_location.npy'))
        self.object_rotation = np.load(os.path.join(root_dir, 'AHOI_Data', 'DATA_FOLDER', 'object_rotation.npy'))
        self.object_info = np.load(os.path.join(root_dir, 'AHOI_Data', 'AHOI_ROOT', 'Metas', 'object_info.npy'), allow_pickle=True).item()

        self.object_verts_all = self.load_object_verts()


    def __len__(self, ):
        return self.human_betas.shape[0]


    def __getitem__(self, idx):
        human_betas = self.human_betas[idx]
        human_orient = self.human_orient[idx]
        human_pose = self.human_pose[idx]
        human_transl = self.human_transl[idx]
        human_pose = self.human_pose[idx]

        object_id = self.object_ids[idx]
        object_location = self.object_location[idx]
        object_rotation = self.object_rotation[idx]

        object_R = Rotation.from_euler('xyz', [0, np.pi, 0]).as_matrix().reshape(-1, 3, 3) @ \
                   Rotation.from_euler('xyz', object_rotation).as_matrix()
        object_T = object_location @ Rotation.from_euler('xyz', [0, np.pi, 0]).as_matrix().T

        object_ind_in_meta = np.where(self.object_info['object_ids'] == int(object_id))[0][0]
        object_v = self.object_verts_all[object_ind_in_meta] # [7, n, 3]

        outputs = {
            'object_id': object_id,
            'human_betas': human_betas.astype(np.float32),
            'human_orient': human_orient.astype(np.float32),
            'human_pose': human_pose.astype(np.float32),
            'human_transl': human_transl.astype(np.float32),
            'object_T': object_T.astype(np.float32),
            'object_R': object_R.astype(np.float32),
            'object_v': object_v.astype(np.float32),
        }

        return outputs


    def load_object_verts(self, ):
        object_verts_all = np.zeros((NUM_OBJECT, OBJECT_PARTS_NUM, MESH_VERTICES_NUM_MAX, 3))

        object_mesh_dir = os.path.join(self.root_dir, 'AHOI_Data', 'AHOI_ROOT', 'Meshes_wt')
        for object_id in os.listdir(object_mesh_dir):
            object_ind_in_meta = np.where(self.object_info['object_ids'] == int(object_id))[0][0]

            for file in os.listdir(os.path.join(object_mesh_dir, object_id)):

                if file.split('.')[-1] != 'obj':
                    continue
                part_name = file.split('.')[0]
                if part_name not in part_name2id: # other
                    continue
                part_idx = part_name2id[part_name]

                mesh = trimesh.load(os.path.join(object_mesh_dir, object_id, file), process=False)

                verts = trimesh.sample.sample_surface(mesh, MESH_VERTICES_NUM_MAX)[0]
                object_verts_all[object_ind_in_meta, part_idx, :] = verts

        return object_verts_all


def sample_smpl_object_anchors(args):

    smpl_pkl = load_pickle('data/models/smplx/SMPLX_MALE.pkl')
    radius = 0.02

    weights = smpl_pkl['weights']
    parts_indices = np.argmax(weights[:, :22], axis=1)

    smpl_anchor_indices = []
    for i in range(22):
        part_anchor_indices = np.where(parts_indices == i)[0]
        part_anchors = part_anchor_indices[np.random.choice(len(part_anchor_indices), args.smpl_anchor_num)]
        smpl_anchor_indices.append(part_anchors.tolist())

    object_anchor_indices = {}
    object_templates = {
        'sphere_1k': trimesh.load('data/models/spheres/sphere_1k.ply', process=False),
        'sphere_2k': trimesh.load('data/models/spheres/sphere_2k.ply', process=False),
        'sphere_4k': trimesh.load('data/models/spheres/sphere_4k.ply', process=False),
        'sphere_5k': trimesh.load('data/models/spheres/sphere_5k.ply', process=False),
        'sphere_8k': trimesh.load('data/models/spheres/sphere_8k.ply', process=False),
        'sphere_9k': trimesh.load('data/models/spheres/sphere_9k.ply', process=False),
    }
    for object_name, mesh in object_templates.items():
        _, face_index = trimesh.sample.sample_surface_even(mesh, count=args.object_anchor_num, radius=radius)
        while face_index.shape[0] < args.object_anchor_num:
            print('Try again.')
            _, face_index = trimesh.sample.sample_surface_even(mesh, count=args.object_anchor_num, radius=radius)

        vertex_indices = np.array(mesh.faces)[face_index][:, 0]
        object_anchor_indices[object_name] = vertex_indices.tolist()

    anchor_indices = {'smpl': smpl_anchor_indices, 'object': object_anchor_indices}
    return anchor_indices


def generate_object_inter_shape_partwise(args):
    device = torch.device('cuda')
    res = 64
    n_parts = 7
    coords_outputs_dir = 'data/datasets/chairs/object_inter_shapes_1k'
    os.makedirs(coords_outputs_dir, exist_ok=True)
    mesh_outputs_dir = 'data/datasets/chairs/object_inter_meshes_1k'
    os.makedirs(mesh_outputs_dir, exist_ok=True)

    voxel_encoder = Voxel3DEncoder(feat_dim=256, num_parts=n_parts, res=res).to(device)
    sphere = Sphere('data/models/spheres/sphere_1k.ply', radius=0.1)
    part_mesh_deformers = []
    for _ in range(n_parts):
        mesh_deformer = MeshDeformer(sphere, features_dim=256, hidden_dim=256, stages=3, layers_per_stages=4).to(device)
        part_mesh_deformers.append(mesh_deformer)

    state_dict = torch.load('outputs/chairs/chairs_object_inter_models_1k.pth')
    voxel_encoder.load_state_dict(state_dict['voxel_encoder'])
    for part_idx, mesh_deformer in enumerate(part_mesh_deformers):
        mesh_deformer.load_state_dict(state_dict['mesh_deformer_{}'.format(part_idx)])

    dataset_root_dir = '/storage/data/huochf/CHAIRS/'
    voxel_dir = os.path.join(dataset_root_dir, 'AHOI_Data', 'AHOI_ROOT', 'object_voxel_{}'.format(res))
    for file in os.listdir(voxel_dir):
        object_id = file.split('.')[0]
        voxels = np.load(os.path.join(voxel_dir, file))
        voxels = torch.from_numpy(voxels.astype(np.float32)).reshape(1, 1, res, res, res).to(device)
        object_feats = voxel_encoder(voxels)

        coords_all_parts = []
        for part_id in range(n_parts):
            part_feats = object_feats[:, part_id]
            parts_coords = part_mesh_deformers[part_id](part_feats)
            coords_all_parts.append(parts_coords[-1])
        coords_all_parts = torch.cat(coords_all_parts, dim=0) # [n_parts, n, 3]

        coords_all_parts = coords_all_parts.detach().cpu().numpy()
        np.save(os.path.join(coords_outputs_dir, '{}.npy'.format(object_id)), coords_all_parts)
        for part_idx in range(n_parts):
            part_mesh = trimesh.Trimesh(coords_all_parts[part_idx], sphere.faces.detach().cpu().numpy())
            part_mesh.export(os.path.join(mesh_outputs_dir, '{}_{}.ply'.format(object_id, part_idx)))


def generate_object_inter_shape(args):
    device = torch.device('cuda')
    res = 64
    coords_outputs_dir = 'data/datasets/chairs/object_inter_shapes_2k_alpha_10000_no_smooth'
    os.makedirs(coords_outputs_dir, exist_ok=True)
    mesh_outputs_dir = 'data/datasets/chairs/object_inter_meshes_2k_alpha_10000_no_smooth'
    os.makedirs(mesh_outputs_dir, exist_ok=True)

    voxel_encoder = Voxel3DEncoder(feat_dim=256, num_parts=1, res=res).to(device)
    sphere = Sphere('data/models/spheres/sphere_2k.ply', radius=1.)
    mesh_deformer = MeshDeformer(sphere, features_dim=256, hidden_dim=256, stages=3, layers_per_stages=4).to(device)

    state_dict = torch.load('outputs/chairs/chairs_object_inter_models_2k_alpha_10000_no_smooth.pth')
    voxel_encoder.load_state_dict(state_dict['voxel_encoder'])
    mesh_deformer.load_state_dict(state_dict['mesh_deformer'])

    dataset_root_dir = '/storage/data/huochf/CHAIRS/'
    voxel_dir = os.path.join(dataset_root_dir, 'AHOI_Data', 'AHOI_ROOT', 'object_voxel_{}'.format(res))

    with open(os.path.join(dataset_root_dir, 'AHOI_Data', 'AHOI_ROOT', 'Metas', 'object_voxel_scale.pkl'), 'rb') as f:
        voxel_scale = pickle.load(f)
    for file in tqdm(os.listdir(voxel_dir)):
        object_id = file.split('.')[0]
        object_mean = torch.from_numpy(voxel_scale[str(object_id)]['mean']).float().to(device)
        object_scale = torch.from_numpy(voxel_scale[str(object_id)]['scale']).float().to(device)
        voxels = np.load(os.path.join(voxel_dir, file))
        voxels = torch.from_numpy(voxels.astype(np.float32)).reshape(1, 1, res, res, res).to(device)
        object_feats = voxel_encoder(voxels).squeeze(1)
        object_coords = mesh_deformer(object_feats)
        object_coords = object_coords[-1]
        object_coords = object_coords * object_scale.reshape(1, 1, 3) / 2  + object_mean.reshape(1, 1, 3)

        object_coords = object_coords.detach().cpu().numpy().reshape(-1, 3)
        np.save(os.path.join(coords_outputs_dir, '{}.npy'.format(object_id)), object_coords)

        part_mesh = trimesh.Trimesh(object_coords, sphere.faces.detach().cpu().numpy())
        part_mesh.export(os.path.join(mesh_outputs_dir, '{}.ply'.format(object_id)))


def extract_person_bbox(args):

    device = torch.device('cuda')
    model = torch.hub.load('./externals/yolov5', 'custom', path='./data/yolov5s.pt', source='local', force_reload=True, _verbose=True,)

    root_dir = '/storage/data/huochf/CHAIRS/'
    img_names = np.load(os.path.join(root_dir, 'AHOI_Data', 'DATA_FOLDER', 'img_name.npy'))
    person_bboxes_all = []
    for img_name in tqdm(img_names):
        image_path = os.path.join(root_dir, 'rimage', img_name)
        if not os.path.exists(image_path):
            image_path = os.path.join(root_dir, 'rimage_extra', img_name)

        if not os.path.exists(image_path):
            person_bbox = np.zeros(4)
        else:
            results = model(image_path)
            boxes = torch.cat(results.xyxy)
            person_bboxes = boxes[boxes[:, -1] == 0]
            if len(person_bboxes) != 0:
                person_idx = torch.argmax(person_bboxes[:, -2])

                person_bbox = boxes[person_idx].detach().cpu().numpy()[:4]
            else:
                person_bbox = np.zeros(4)
        person_bboxes_all.append(person_bbox)
    person_bboxes_all = np.stack(person_bboxes_all, axis=0).astype(np.float32)
    np.save(os.path.join(root_dir, 'AHOI_Data', 'DATA_FOLDER', 'person_bboxes.npy'), person_bboxes_all)


def extract_mask(args):
    from segment_anything import sam_model_registry, SamPredictor

    device = torch.device('cuda')
    model = torch.hub.load('./externals/yolov5', 'custom', path='./data/yolov5s.pt', source='local', force_reload=True, _verbose=True,)

    sam_model = sam_model_registry['vit_h'](checkpoint='/public/home/huochf/projects/3D_HOI/hoiYouTube/03_track_hoi/weights/sam_vit_h_4b8939.pth')
    sam_predictor = SamPredictor(sam_model.to(device))

    root_dir = '/storage/data/huochf/CHAIRS/'
    for sequence_id in tqdm(sorted(os.listdir(os.path.join(root_dir, 'rimage')))):
        for cam_id in os.listdir(os.path.join(root_dir, 'rimage', sequence_id)):
            for file in os.listdir(os.path.join(root_dir, 'rimage', sequence_id, cam_id)):
                image_path = os.path.join(root_dir, 'rimage', sequence_id, cam_id, file)
                results = model(image_path)
                boxes = torch.cat(results.xyxy)
                person_bboxes = boxes[boxes[:, -1] == 0]
                if len(person_bboxes) != 0:
                    person_idx = torch.argmax(person_bboxes[:, -2])

                    person_bbox = boxes[person_idx].detach().cpu().numpy()

                    image = cv2.imread(image_path)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                    sam_predictor.set_image(image)
                    person_masks, _, _ = sam_predictor.predict(box=np.array(person_bbox[:4]), multimask_output=False)
                else:
                    person_masks = np.zeros((1, 540, 960))

                mask_path = os.path.join(root_dir, 'rmask', sequence_id, cam_id)
                os.makedirs(mask_path, exist_ok=True)
                cv2.imwrite(os.path.join(mask_path, file.replace('rgb', 'mask')), (person_masks[0] * 255).astype(np.uint8))

    for sequence_id in tqdm(sorted(os.listdir(os.path.join(root_dir, 'rimage_extra')))):
        for cam_id in os.listdir(os.path.join(root_dir, 'rimage_extra', sequence_id)):
            for file in os.listdir(os.path.join(root_dir, 'rimage_extra', sequence_id, cam_id)):
                image_path = os.path.join(root_dir, 'rimage_extra', sequence_id, cam_id, file)
                results = model(image_path)
                boxes = torch.cat(results.xyxy)
                person_bboxes = boxes[boxes[:, -1] == 0]
                if len(person_bboxes) != 0:
                    person_idx = torch.argmax(person_bboxes[:, -2])

                    person_bbox = boxes[person_idx].detach().cpu().numpy()

                    image = cv2.imread(image_path)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                    sam_predictor.set_image(image)
                    person_masks, _, _ = sam_predictor.predict(box=np.array(person_bbox[:4]), multimask_output=False)
                else:
                    person_masks = np.zeros((1, 540, 960))

                mask_path = os.path.join(root_dir, 'rmask_extra', sequence_id, cam_id)
                os.makedirs(mask_path, exist_ok=True)
                cv2.imwrite(os.path.join(mask_path, file.replace('rgb', 'mask')), (person_masks[0] * 255).astype(np.uint8))


def extract_person_mask(args):
    from detectron2.config import get_cfg
    from detectron2.engine import DefaultPredictor
    from detectron2.projects.point_rend import add_pointrend_config
    IMAGE_SIZE = 640
    POINTREND_CONFIG = '/public/home/huochf/projects/3D_HOI/StackFLOW/data/weights/pointrend_rcnn_X_101_32x8d_FPN_3x_coco.yaml'
    POINTREND_MODEL_WEIGHTS = '/public/home/huochf/projects/3D_HOI/StackFLOW/data/weights/model_final_ba17b9.pkl' # x101-FPN 3x

    def get_pointrend_predictor(min_confidence=0.9, image_format='RGB'):
        cfg = get_cfg()
        add_pointrend_config(cfg)
        cfg.merge_from_file(POINTREND_CONFIG)
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = min_confidence
        cfg.MODEL.WEIGHTS = POINTREND_MODEL_WEIGHTS
        cfg.INPUT.FORMAT = image_format
        return DefaultPredictor(cfg)

    segmenter = get_pointrend_predictor()

    root_dir = '/storage/data/huochf/CHAIRS/'
    for sequence_id in tqdm(os.listdir(os.path.join(root_dir, 'rimage'))):
        for cam_id in os.listdir(os.path.join(root_dir, 'rimage', sequence_id)):
            for file in os.listdir(os.path.join(root_dir, 'rimage', sequence_id, cam_id)):
                image_path = os.path.join(root_dir, 'rimage', sequence_id, cam_id, file)

                image = Image.open(image_path).convert('RGB')
                image = np.array(image)
                instances = segmenter(image)['instances']
                person_mask = instances.pred_masks[instances.pred_classes == 0][0]
                person_mask = person_mask.detach().cpu().numpy().astype(np.uint8) * 255

                mask_path = os.path.join(root_dir, 'rmask', sequence_id, cam_id)
                os.makedirs(mask_path, exist_ok=True)
                cv2.imwrite(os.path.join(mask_path, file.replace('rgb', 'mask')), person_mask.astype(np.uint8))

    for sequence_id in tqdm(os.listdir(os.path.join(root_dir, 'rimage_extra'))):
        for cam_id in os.listdir(os.path.join(root_dir, 'rimage_extra', sequence_id)):
            for file in os.listdir(os.path.join(root_dir, 'rimage_extra', sequence_id, cam_id)):
                image_path = os.path.join(root_dir, 'rimage_extra', sequence_id, cam_id, file)
                image = Image.open(image_path).convert('RGB')
                image = np.array(image)
                instances = segmenter(image)['instances']
                person_mask = instances.pred_masks[instances.pred_classes == 0][0]
                person_mask = person_mask.detach().cpu().numpy().astype(np.uint8) * 255

                mask_path = os.path.join(root_dir, 'rmask_extra', sequence_id, cam_id)
                os.makedirs(mask_path, exist_ok=True)
                cv2.imwrite(os.path.join(mask_path, file.replace('rgb', 'mask')), person_mask.astype(np.uint8))



def extract_smplx(args):
    import sys
    sys.path.append('externals/SMPLer-X/main/')
    sys.path.append('externals/SMPLer-X/data/')
    sys.path.append('externals/SMPLer-X/common/')
    import torch.backends.cudnn as cudnn
    import torchvision.transforms as transforms

    pretrained_model = 'smpler_x_h32'
    testset = 'EHF'
    agora_benchmark = 'agora_model'
    num_gpus = 1
    exp_name = 'inference'

    from config import cfg
    config_path = os.path.join('externals/SMPLer-X/main/config', f'config_{pretrained_model}.py')
    ckpt_path = os.path.join('externals/SMPLer/pretrained_models', f'{pretrained_model}.pth.tar')

    cfg.get_config_fromfile(config_path)
    cfg.update_test_config(testset, agora_benchmark, shapy_eval_split=None, 
                            pretrained_model_path=ckpt_path, use_cache=False)
    cfg.update_config(num_gpus, exp_name)
    cfg.encoder_config_file = 'externals/SMPLer-X/main/transformer_utils/configs/smpler_x/encoder/body_encoder_huge.py'
    cfg.pretrained_model_path = 'externals/SMPLer-X/pretrained_models/smpler_x_h32.pth.tar'
    cudnn.benchmark = True

    from utils.preprocessing import load_img, process_bbox, generate_patch_image

    class ImageDataset:

        def __init__(self, root_dir):
            self.root_dir = root_dir
            self.person_bboxes = np.load(os.path.join(root_dir, 'AHOI_Data', 'DATA_FOLDER', 'person_bboxes.npy'))
            self.img_names = np.load(os.path.join(root_dir, 'AHOI_Data', 'DATA_FOLDER', 'img_name.npy'))

            self.input_img_shape = (512, 384)
            self.transform = transforms.ToTensor()


        def __len__(self, ):
            return len(self.person_bboxes)


        def __getitem__(self, idx):
            person_bbox = self.person_bboxes[idx] # (xyxy)
            img_name = self.img_names[idx]
            image_path = os.path.join(self.root_dir, 'rimage', img_name)
            if not os.path.exists(image_path):
                image_path = os.path.join(self.root_dir, 'rimage_extra', img_name)

            if not os.path.exists(image_path):
                return img_name, torch.zeros((3, 512, 384)).float(), np.zeros(4, ).astype(np.float32), np.zeros(2, ).astype(np.float32)

            original_img = load_img(image_path)
            original_img_height, original_img_width = original_img.shape[:2]
            box_xywh = np.zeros((4))
            box_xywh[0] = person_bbox[0]
            box_xywh[1] = person_bbox[1]
            box_xywh[2] = person_bbox[2] - person_bbox[0]
            box_xywh[3] = person_bbox[3] - person_bbox[1]

            start_point = (int(box_xywh[0]), int(box_xywh[1]))
            end_point = (int(box_xywh[2]), int(box_xywh[3]))
            bbox = process_bbox(box_xywh, original_img_width, original_img_height)
            if bbox is None:
                ratio = 1.25
                w = box_xywh[2]
                h = box_xywh[3]
                c_x = box_xywh[0] + w / 2.
                c_y = box_xywh[1] + h / 2.
                aspect_ratio = self.input_img_shape[1] / self.input_img_shape[0]
                if w > aspect_ratio * h:
                    h = w / aspect_ratio
                elif w < aspect_ratio * h:
                    w = h * aspect_ratio
                bbox = np.zeros((4))
                bbox[2] = w * ratio
                bbox[3] = h * ratio
                bbox[0] = c_x - box_xywh[2] / 2.
                bbox[1] = c_y - box_xywh[3] / 2.
                bbox = bbox.astype(np.float32)

            img, img2bb_trans, bb2img_trans = generate_patch_image(original_img, bbox, 1.0, 0.0, False, self.input_img_shape)
            img = self.transform(img.astype(np.float32)) / 255

            img_orig_shape = np.array([original_img_height, original_img_width], dtype=np.float32)

            return img_name, img, bbox, img_orig_shape

    device = torch.device('cuda')
    batch_size = 8
    f = (5000, 5000)
    input_body_shape = (256, 192)
    princpt = (input_body_shape[1] / 2, input_body_shape[0] / 2)

    from base import Demoer
    demoer = Demoer()
    demoer._make_model()
    demoer.model.eval()

    smpl_params_all = []
    dataset = ImageDataset('/storage/data/huochf/CHAIRS/')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=8, shuffle=False, drop_last=False)

    for items in tqdm(dataloader):
        frame_ids, images, box_xywh, img_orig_shapes = items
        b = images.shape[0]
        inputs = {'img': images.to(device)}
        targets = {}
        meta_info = {}

        with torch.no_grad():
            output = demoer.model(inputs, targets, meta_info, 'test')

        for idx, frame_id in enumerate(frame_ids):
            smplx_params = {}
            focal = [f[0] / input_body_shape[1] * box_xywh[idx, 2], f[1] / input_body_shape[0] * box_xywh[idx, 3]]
            _princpt = [princpt[0] / input_body_shape[1] * box_xywh[idx, 2] + box_xywh[idx, 0], princpt[1] / input_body_shape[0] * box_xywh[idx, 3] + box_xywh[idx, 1]]
            smplx_params['frame_id'] = frame_id
            smplx_params['focal'] = focal
            smplx_params['princpt'] = _princpt
            smplx_params['bbox'] = box_xywh[idx].numpy()
            smplx_params['global_orient'] = output['smplx_root_pose'].reshape(b, -1, 3)[idx].detach().cpu().numpy()
            smplx_params['body_pose'] = output['smplx_body_pose'].reshape(b, -1, 3)[idx].detach().cpu().numpy()
            smplx_params['left_hand_pose'] = output['smplx_lhand_pose'].reshape(b, -1, 3)[idx].detach().cpu().numpy()
            smplx_params['right_hand_pose'] = output['smplx_rhand_pose'].reshape(b, -1, 3)[idx].detach().cpu().numpy()
            smplx_params['jaw_pose'] = output['smplx_jaw_pose'].reshape(b, -1, 3)[idx].detach().cpu().numpy()
            smplx_params['betas'] = output['smplx_shape'].reshape(b, -1, 10)[idx].detach().cpu().numpy()
            smplx_params['expression'] = output['smplx_expr'].reshape(b, -1, 10)[idx].detach().cpu().numpy()
            smplx_params['transl'] = output['cam_trans'].reshape(b, -1, 3)[idx].detach().cpu().numpy()

            smpl_params_all.append(smplx_params)

    save_pickle(smpl_params_all, os.path.join('/storage/data/huochf/CHAIRS/', 'AHOI_Data', 'DATA_FOLDER', 'smplerx.pkl'))


def prepare_annotation_partwise(args):
    output_dir = 'data/datasets/'
    root_dir = '/storage/data/huochf/CHAIRS/'
    img_names = np.load(os.path.join(root_dir, 'AHOI_Data', 'DATA_FOLDER', 'img_name.npy'))
    human_betas = np.load(os.path.join(root_dir, 'AHOI_Data', 'DATA_FOLDER', 'human_betas.npy'))
    human_pose = np.load(os.path.join(root_dir, 'AHOI_Data', 'DATA_FOLDER', 'human_pose.npy'))

    object_ids = np.load(os.path.join(root_dir, 'AHOI_Data', 'DATA_FOLDER', 'object_id.npy'))
    object_location = np.load(os.path.join(root_dir, 'AHOI_Data', 'DATA_FOLDER', 'object_location.npy'))
    object_rotation = np.load(os.path.join(root_dir, 'AHOI_Data', 'DATA_FOLDER', 'object_rotation.npy'))
    person_bboxes = np.load(os.path.join(root_dir, 'AHOI_Data', 'DATA_FOLDER', 'person_bboxes.npy'))

    smpler_params = load_pickle(os.path.join(root_dir, 'AHOI_Data', 'DATA_FOLDER', 'smplerx.pkl'))

    annotation_train_dict, annotation_test_dict = {}, {}
    for idx, img_name in enumerate(tqdm(img_names)):
        image_path = os.path.join(root_dir, 'rimage', img_name)
        if not os.path.exists(image_path):
            image_path = os.path.join(root_dir, 'rimage_extra', img_name)

        if not os.path.exists(image_path):
            continue

        _, seq_id, cam_id, frame_id = img_name.split('/')[-1].split('.')[0].split('_')
        object_id = object_ids[idx]

        img_id = '_'.join(['{:03d}'.format(object_id), seq_id, cam_id, frame_id])
        betas = human_betas[idx] # (10, )
        body_theta = human_pose[idx] # (63, )
        body_rotmat = Rotation.from_rotvec(body_theta.reshape(21, 3)).as_matrix()

        obj_rot = object_rotation[idx] # [n_parts, 3]
        obj_loc = object_location[idx] # [n_parts, 3]
        object_rel_R = Rotation.from_euler('xyz', [0, np.pi, 0]).as_matrix().reshape(-1, 3, 3) @ \
                   Rotation.from_euler('xyz', obj_rot).as_matrix()
        object_rel_T = obj_loc @ Rotation.from_euler('xyz', [0, np.pi, 0]).as_matrix().T

        _params = smpler_params[idx]

        annotations = {
            'person_bb_xyxy': person_bboxes[idx], # (4, )
            'smplx_betas': betas, # (10, )
            'smplx_body_theta': body_theta, # (66, )
            'smplx_body_rotmat': body_rotmat,
            'object_rel_rotmat': object_rel_R, # (7, 3, 3)
            'object_rel_trans': object_rel_T, # (7, 3)
            'smpler_betas': _params['betas'], # [10, ]
            'smpler_body_pose': _params['body_pose'], # [-1, 3]
        }

        if object_id in train_object_ids:
            annotation_train_dict[img_id] = annotations
        else:
            annotation_test_dict[img_id] = annotations

    save_pickle(annotation_train_dict, os.path.join(output_dir, 'chairs_train_annotations.pkl'))
    save_pickle(annotation_test_dict,  os.path.join(output_dir, 'chairs_test_annotations.pkl'))


def prepare_annotation(args):
    output_dir = 'data/datasets/'
    root_dir = '/storage/data/huochf/CHAIRS/'
    img_names = np.load(os.path.join(root_dir, 'AHOI_Data', 'DATA_FOLDER', 'img_name.npy'))
    human_betas = np.load(os.path.join(root_dir, 'AHOI_Data', 'DATA_FOLDER', 'human_betas.npy'))
    human_pose = np.load(os.path.join(root_dir, 'AHOI_Data', 'DATA_FOLDER', 'human_pose.npy'))

    object_ids = np.load(os.path.join(root_dir, 'AHOI_Data', 'DATA_FOLDER', 'object_id.npy'))
    object_location = np.load(os.path.join(root_dir, 'AHOI_Data', 'DATA_FOLDER', 'object_location.npy'))
    object_rotation = np.load(os.path.join(root_dir, 'AHOI_Data', 'DATA_FOLDER', 'object_rotation.npy'))
    person_bboxes = np.load(os.path.join(root_dir, 'AHOI_Data', 'DATA_FOLDER', 'person_bboxes.npy'))
    object_root_location = np.load(os.path.join(root_dir, 'AHOI_Data', 'DATA_FOLDER', 'object_root_location.npy'))
    object_root_rotation = np.load(os.path.join(root_dir, 'AHOI_Data', 'DATA_FOLDER', 'object_root_rotation.npy'))

    object_info = np.load(os.path.join(root_dir, 'AHOI_Data', 'AHOI_ROOT', 'Metas', 'object_info.npy'), allow_pickle=True).item()

    smpler_params = load_pickle(os.path.join(root_dir, 'AHOI_Data', 'DATA_FOLDER', 'smplerx.pkl'))

    annotation_train_dict, annotation_test_dict = {}, {}
    for idx, img_name in enumerate(tqdm(img_names)):
        image_path = os.path.join(root_dir, 'rimage', img_name)
        if not os.path.exists(image_path):
            image_path = os.path.join(root_dir, 'rimage_extra', img_name)

        if not os.path.exists(image_path):
            continue

        _, seq_id, cam_id, frame_id = img_name.split('/')[-1].split('.')[0].split('_')
        object_id = object_ids[idx]

        img_id = '_'.join(['{:03d}'.format(object_id), seq_id, cam_id, frame_id])
        betas = human_betas[idx] # (10, )
        body_theta = human_pose[idx] # (63, )
        body_rotmat = Rotation.from_rotvec(body_theta.reshape(21, 3)).as_matrix()

        object_location = object_root_location[idx]
        object_rotation = object_root_rotation[idx]
        object_R = Rotation.from_euler('xyz', [0, np.pi, 0]).as_matrix() @ \
                   Rotation.from_euler('xyz', object_rotation).as_matrix()
        object_T =  Rotation.from_euler('xyz', [0, np.pi, 0]).as_matrix() @ object_location.reshape(3, 1)
        object_T = object_T.reshape(3)

        _params = smpler_params[idx]

        annotations = {
            'person_bb_xyxy': person_bboxes[idx], # (4, )
            'smplx_betas': betas, # (10, )
            'smplx_body_theta': body_theta, # (66, )
            'smplx_body_rotmat': body_rotmat,
            'object_rel_rotmat': object_R, # (3, 3)
            'object_rel_trans': object_T, # (3)
            'smpler_betas': _params['betas'], # [10, ]
            'smpler_body_pose': _params['body_pose'], # [-1, 3]
        }

        if object_id in train_object_ids:
            annotation_train_dict[img_id] = annotations
        else:
            annotation_test_dict[img_id] = annotations

    save_pickle(annotation_train_dict, os.path.join(output_dir, 'chairs_train_annotations.pkl'))
    save_pickle(annotation_test_dict,  os.path.join(output_dir, 'chairs_test_annotations.pkl'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--smpl_anchor_num', default=32, type=int, help='the number of SMPL anchors per body part')
    parser.add_argument('--object_anchor_num', default=128, type=int, help='the number of object anchors')

    args = parser.parse_args()

    np.random.seed(7) # for reproducibility
    random.seed(7)

    # Step #1: generate anchors
    # anchor_indices = sample_smpl_object_anchors(args)
    # out_path = 'data/datasets/chairs_anchor_indices_n{}_{}.pkl'.format(args.smpl_anchor_num, args.object_anchor_num)
    # save_pickle(anchor_indices, out_path)

    # Step #2: Train the Voxel Encoders and Mesh Deformers (learn_object_inter_shape.py)
    # After training, generate the anchors for each object parts.
    generate_object_inter_shape(args)

    # Step $3 generate boxes for person
    # extract_person_bbox(args)
    # extract_mask(args)
    # extract_person_mask(args)

    # Step #4: extract smplx
    # extract_smplx(args)

    # Step #5: generate annotation lists
    # prepare_annotation(args)
