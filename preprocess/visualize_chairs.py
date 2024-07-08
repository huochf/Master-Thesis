import os
import sys
file_dir = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(file_dir, '..', ))
from tqdm import tqdm
import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from smplx import SMPLH, SMPLX, SMPLHLayer, SMPLXLayer

import neural_renderer as nr
from hoi_recon.datasets.utils import load_pickle, save_pickle


KPS_COLORS = [
    [0.,    255.,  255.],
    [0.,   255.,    170.],
    [0., 170., 255.,],
    [85., 170., 255.],
    [0.,   255.,   85.], # 4
    [0., 85., 255.],
    [170., 85., 255.],
    [0.,   255.,   0.], # 7
    [0., 0., 255.], 
    [255., 0., 255.],
    [0.,    255.,  0.], # 10
    [0., 0., 255.],
    [255., 85., 170.],
    [170., 0, 255.],
    [255., 0., 170.],
    [255., 170., 85.],
    [85., 0., 255.],
    [255., 0., 85],
    [32., 0., 255.],
    [255., 0, 32],
    [0., 0., 255.],
    [255., 0., 0.],
]


def plot_smpl_keypoints(image, keypoints):
    bone_idx = [[ 0,  1], [ 0,  2], [ 0,  3], 
                [ 1,  4], [ 2,  5], [ 3,  6], 
                [ 4,  7], [ 5,  8], [ 6,  9], 
                [ 7, 10], [ 8, 11], [ 9, 12], 
                [ 9, 13], [ 9, 14], [12, 15],
                [13, 16], [14, 17], [16, 18],
                [17, 19], [18, 20], [19, 21]]
    line_thickness = 2
    thickness = 4
    lineType = 8

    for bone in bone_idx:
        idx1, idx2 = bone
        x1, y1 = keypoints[idx1]
        x2, y2 = keypoints[idx2]
        cv2.line(image, (int(x1), int(y1)), (int(x2), int(y2)), tuple(KPS_COLORS[idx1 % len(KPS_COLORS)]), line_thickness, lineType)

    for i, points in enumerate(keypoints):
        x, y = points
        x, y = int(x), int(y)
        cv2.circle(image, (x, y), thickness, KPS_COLORS[i % len(KPS_COLORS)], thickness=-1, lineType=lineType)

    return image


def render(image, v, f, K, t, R):
    device = torch.device('cuda')
    v = v @ R.transpose(1, 0) + t.reshape(1, 3)
    v = v.reshape(1, -1, 3).to(device)
    f = f.reshape(1, -1, 3).to(device)
    textures = torch.tensor([0.65098039, 0.74117647, 0.85882353], dtype=torch.float32).reshape(1, 1, 1, 1, 1, 3).repeat(1, f.shape[1], 1, 1, 1, 1).to(device)
    h, w, _ = image.shape

    K = torch.tensor(K, dtype=torch.float32).reshape(1, 3, 3).to(device)
    R = torch.eye(3, dtype=torch.float32).reshape(1, 3, 3).to(device)
    t = torch.zeros(3, dtype=torch.float32).reshape(1, 3).to(device)

    s = max(h, w)
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


def visualize_human_object_mesh():
    root_dir = '/storage/data/huochf/CHAIRS/'
    annotations = load_pickle('./data/datasets/chairs_train_annotations.pkl')
    output_dir = 'outputs/visualize_chairs'
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device('cuda')

    smplx = SMPLX(model_path='data/models/smplx', gender='male').to(device)
    smplx_f = torch.tensor(np.array(smplx.faces).astype(np.int64)).to(device)
    img_w, img_h = 960, 540
    fov_x = 90 # degrees
    fov_y = 59
    fx = img_w / 2 / np.tan((fov_x / 2) / 180 * np.pi)
    fy = img_h / 2 / np.tan((fov_y / 2) / 180 * np.pi)
    cx, cy = img_w / 2, img_h / 2
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]).astype(np.float32)

    for img_id, anno in tqdm(annotations.items()):
        object_id, seq_id, cam_id, frame_id = img_id.split('_')
        # print(img_id)
        image_path = os.path.join(root_dir, 'rimage', seq_id, cam_id, 'rgb_{}_{}_{}.jpg'.format(seq_id, cam_id, frame_id))
        if not os.path.exists(image_path):
            image_path = image_path.replace('rimage', 'rmage_extra')
        image = cv2.imread(image_path)

        betas = anno['smplx_betas']
        thetas = anno['smplx_theta']
        hoi_trans = anno['hoi_trans']
        hoi_rotmat = anno['hoi_rotmat']
        smplx_out = smplx(betas=torch.tensor(betas).unsqueeze(0).to(device), body_pose=torch.tensor(thetas[3:]).unsqueeze(0).to(device))
        smplx_v = smplx_out.vertices.detach()[0]
        smplx_J = smplx_out.joints.detach()[0]
        smplx_v = smplx_v - smplx_J[:1]

        hoi_rotmat = Rotation.from_euler('xyz', [0, 0, np.pi]).as_matrix() @ hoi_rotmat
        hoi_trans = hoi_trans.reshape(1, 3) # @ Rotation.from_euler('xyz', [0, 0, np.pi]).as_matrix().T
        cam_R = torch.tensor(hoi_rotmat).float().to(device)
        cam_T = torch.tensor(hoi_trans).float().to(device)

        image = render(image, smplx_v, smplx_f, K, cam_T, cam_R)

        smpl_kps_2d = anno['smplx_joints_2d']
        image = plot_smpl_keypoints(image, smpl_kps_2d[:22])

        cv2.imwrite(os.path.join(output_dir, '{}.jpg'.format(img_id)), image)


if __name__ == '__main__':
    visualize_human_object_mesh()
