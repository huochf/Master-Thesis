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
from scipy.spatial.transform import Rotation
from hoi_recon.datasets.behave_extend_metadata import BEHAVEExtendMetaData
from hoi_recon.datasets.intercap_metadata import InterCapMetaData

import neural_renderer as nr


def render(image, v, f, K, dist_coeffs, t, R):
    device = torch.device('cuda')
    v = v @ R.transpose(1, 0) + t.reshape(1, 3)
    v = v.reshape(1, -1, 3).to(device)
    f = f.reshape(1, -1, 3).to(device)
    textures = torch.tensor([0.65098039, 0.74117647, 0.85882353], dtype=torch.float32).reshape(1, 1, 1, 1, 1, 3).repeat(1, f.shape[1], 1, 1, 1, 1).to(device)
    h, w, _ = image.shape

    K = torch.tensor(K, dtype=torch.float32).reshape(1, 3, 3).to(device)
    R = torch.eye(3, dtype=torch.float32).reshape(1, 3, 3).to(device)
    t = torch.zeros(3, dtype=torch.float32).reshape(1, 3).to(device)

    dist_coeffs = torch.tensor(dist_coeffs, dtype=torch.float32).reshape(1, -1).to(device)

    s = max(h, w)
    # renderer = nr.renderer.Renderer(image_size=s, K=K / s, R=R, t=t, dist_coeffs=dist_coeffs, orig_size=1)
    renderer = nr.renderer.Renderer(image_size=s, K=K / s, R=R, t=t, orig_size=1)
    renderer.background_color = [1, 1, 1]
    renderer.light_direction = [1.5, -0.5, 1]
    renderer.light_intensity_direction = 0.3
    renderer.light_intensity_ambient = 0.5
    rend, _, mask = renderer.render(vertices=v, faces=f, textures=textures)
    rend = rend.clip(0, 1)

    rend = rend[0].permute(1, 2, 0).detach().cpu().numpy()
    mask = mask[0].detach().cpu().numpy().reshape((s, s, 1))
    rend = (rend * 255).astype(np.uint8)
    rend = rend[:, :, ::-1]
    rend = rend[:h, :w]
    mask = mask[:h, :w]
    mask = mask * 0.4
    image = image * (1 - mask) + rend * mask

    return image


def visualize_behave_extend(args):
    metadata = BEHAVEExtendMetaData(args.root_dir)

    all_img_ids = list(metadata.go_through_all_frames(split='train'))
    random.shuffle(all_img_ids)

    object_v, object_f = metadata.load_object_mesh_templates()[args.object]
    object_v = torch.tensor(object_v).float()
    object_f = torch.tensor(object_f).float()

    out_dir = 'outputs/visualize_object_pose/{}/{}'.format(args.dataset, args.object)
    os.makedirs(out_dir, exist_ok=True)

    count = 0

    for img_id in tqdm(all_img_ids):
        day_id, sub_id, obj_name, inter_type, frame_id, cam_id = img_id.split('_')
        if obj_name != args.object:
            continue

        image_org = cv2.imread(metadata.get_image_path(img_id))
        if image_org is None:
            continue
        h, w, _ = image_org.shape

        object_render_mask_path = metadata.get_object_full_mask_path(img_id)
        object_render_mask = cv2.imread(object_render_mask_path, cv2.IMREAD_GRAYSCALE) / 255

        try:
            ys, xs = np.nonzero(object_render_mask)
            x1, x2 = xs.min(), xs.max()
            y1, y2 = ys.min(), ys.max()
            x1 *= 2
            x2 *= 2
            y1 *= 2
            y2 *= 2
        except: # invisiable object
            continue

        cam_intrinsics_params = metadata.cam_intrinsics[int(cam_id)]
        fx, fy, cx, cy = cam_intrinsics_params[:4]
        dist_coeffs = cam_intrinsics_params[4:]
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]).astype(np.float32)

        object_rotmat, object_trans = metadata.load_object_RT(img_id)
        cam_R, cam_T = metadata.cam_RT_matrix[day_id][int(cam_id)]
        object_trans = np.matmul(cam_R.transpose(), object_trans - cam_T)
        object_rotmat = np.matmul(cam_R.transpose(), object_rotmat)

        object_trans = torch.tensor(object_trans).float()
        object_rotmat = torch.tensor(object_rotmat).float()

        image = render(image_org.copy(), object_v, object_f, K, dist_coeffs, object_trans, object_rotmat)

        cx, cy = (x1 + x2) / 2, (y1  + y2) / 2
        s = int(max((y2 - y1), (x2 - x1)) * 1.0)

        crop_image = np.zeros((2 * s, 2 * s, 3))

        _x1 = int(cx - s)
        _y1 = int(cy - s)

        if _x1 < 0 and _y1 < 0:
            crop_image[-_y1 : min(h - _y1, 2 * s), -_x1 : min(w - _x1, 2 * s)] = image[0:_y1 + 2 * s, 0:_x1 + 2 * s]
        elif _x1 < 0:
            crop_image[:min(h - _y1, 2 * s), -_x1 : min(w - _x1, 2 * s)] = image[_y1:_y1 + 2 * s, 0:_x1 + 2 * s]
        elif _y1 < 0:
            crop_image[-_y1 : min(h - _y1, 2 * s), :min(w - _x1, 2 * s)] = image[0: _y1 + 2 * s, _x1:_x1 + 2 * s]
        else:
            crop_image[:min(h - _y1, 2 * s), :min(w - _x1, 2 * s)] = image[_y1:_y1 + 2 * s, _x1:_x1 + 2 * s]

        crop_image = cv2.resize(crop_image, dsize=(1024, 1024), interpolation=cv2.INTER_LINEAR)


        crop_image_org = np.zeros((2 * s, 2 * s, 3))

        _x1 = int(cx - s)
        _y1 = int(cy - s)

        if _x1 < 0 and _y1 < 0:
            crop_image_org[-_y1 : min(h - _y1, 2 * s), -_x1 : min(w - _x1, 2 * s)] = image_org[0:_y1 + 2 * s, 0:_x1 + 2 * s]
        elif _x1 < 0:
            crop_image_org[:min(h - _y1, 2 * s), -_x1 : min(w - _x1, 2 * s)] = image_org[_y1:_y1 + 2 * s, 0:_x1 + 2 * s]
        elif _y1 < 0:
            crop_image_org[-_y1 : min(h - _y1, 2 * s), :min(w - _x1, 2 * s)] = image_org[0: _y1 + 2 * s, _x1:_x1 + 2 * s]
        else:
            crop_image_org[:min(h - _y1, 2 * s), :min(w - _x1, 2 * s)] = image_org[_y1:_y1 + 2 * s, _x1:_x1 + 2 * s]

        crop_image_org = cv2.resize(crop_image_org, dsize=(1024, 1024), interpolation=cv2.INTER_LINEAR)

        crop_image = np.concatenate([crop_image, crop_image_org], axis=1)

        cv2.imwrite(os.path.join(out_dir, '{}.jpg'.format(img_id)), crop_image.astype(np.uint8))
        print('saved image {}'.format(img_id))

        count += 1
        if count > 3000:
            break


def visualize_intercap(args):
    metadata = InterCapMetaData(args.root_dir)

    all_img_ids = list(metadata.go_through_all_frames(split='train')) # vistracker split
    random.shuffle(all_img_ids)
    count = 0

    object_v, object_f = metadata.load_object_mesh_templates()[args.object]
    object_v = torch.tensor(object_v).float()
    object_f = torch.tensor(object_f).float()
    cam_calibreation = metadata.load_cam_calibration()

    out_dir = 'outputs/visualize_object_pose/{}/{}'.format(args.dataset, args.object)
    os.makedirs(out_dir, exist_ok=True)

    for img_id in tqdm(all_img_ids):
        sub_id, obj_id, seq_id, cam_id, frame_id = img_id.split('_')
        if metadata.OBJECT_IDX2NAME[obj_id] != args.object:
            continue

        image = cv2.imread(metadata.get_image_path(img_id))
        h, w, _ = image.shape

        object_render_mask_path = metadata.get_object_full_mask_path(img_id)
        object_render_mask = cv2.imread(object_render_mask_path, cv2.IMREAD_GRAYSCALE)
        if object_render_mask is None:
            continue
        try:
            ys, xs = np.nonzero(object_render_mask)
            x1, x2 = xs.min(), xs.max()
            y1, y2 = ys.min(), ys.max()
            x1 *= 2
            x2 *= 2
            y1 *= 2
            y2 *= 2
        except: # invisiable object
            continue

        calibration = cam_calibreation[cam_id]
        cam_R = np.array(calibration['R'])
        cam_R = Rotation.from_rotvec(cam_R).as_matrix()
        cam_T = np.array(calibration['T'])
        cx, cy = calibration['c']
        fx, fy = calibration['f']
        dist_coeffs = calibration['k']
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]).astype(np.float32)

        object_rotmat, object_trans = metadata.load_object_RT(img_id)

        object_rotmat = np.matmul(cam_R, object_rotmat)
        object_trans = np.matmul(cam_R, object_trans) + cam_T

        object_trans = torch.tensor(object_trans).float()
        object_rotmat = torch.tensor(object_rotmat).float()

        image = render(image, object_v, object_f, K, dist_coeffs, object_trans, object_rotmat)

        cx, cy = (x1 + x2) / 2, (y1  + y2) / 2
        s = int(max((y2 - y1), (x2 - x1)) * 1.0)

        crop_image = np.zeros((2 * s, 2 * s, 3))

        _x1 = int(cx - s)
        _y1 = int(cy - s)

        if _x1 < 0 and _y1 < 0:
            crop_image[-_y1 : min(h - _y1, 2 * s), -_x1 : min(w - _x1, 2 * s)] = image[0:_y1 + 2 * s, 0:_x1 + 2 * s]
        elif _x1 < 0:
            crop_image[:min(h - _y1, 2 * s), -_x1 : min(w - _x1, 2 * s)] = image[_y1:_y1 + 2 * s, 0:_x1 + 2 * s]
        elif _y1 < 0:
            crop_image[-_y1 : min(h - _y1, 2 * s), :min(w - _x1, 2 * s)] = image[0: _y1 + 2 * s, _x1:_x1 + 2 * s]
        else:
            crop_image[:min(h - _y1, 2 * s), :min(w - _x1, 2 * s)] = image[_y1:_y1 + 2 * s, _x1:_x1 + 2 * s]

        crop_image = cv2.resize(crop_image, dsize=(1024, 1024), interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(os.path.join(out_dir, '{}.jpg'.format(img_id)), crop_image.astype(np.uint8))
        print('saved image {}'.format(img_id))

        count += 1
        if count > 3000:
            break


def visualize_object_pose(args):
    if args.dataset == 'BEHAVE-Extend':
        visualize_behave_extend(args)
    else:
        visualize_intercap(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', default='/storage/data/huochf/BEHAVE', type=str, help='Dataset root directory.')
    parser.add_argument('--dataset', default='BEHAVE-Extend', type=str, choices=['BEHAVE-Extend', 'InterCap'], help='Process behave dataset or intercap dataset.')
    parser.add_argument('--object', default='backpack', type=str)
    args = parser.parse_args()

    visualize_object_pose(args)

