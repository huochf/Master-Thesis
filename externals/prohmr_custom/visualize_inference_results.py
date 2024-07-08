import os
import argparse
import cv2
import numpy as np
import torch
from tqdm import tqdm

from smplx import SMPLHLayer
import neural_renderer as nr

from transflow.datasets.utils import load_json, save_pickle, load_pickle
from transflow.datasets.behave_extend_metadata import BEHAVEExtendMetaData


def render_smpl(image, smpl_v, smpl_f, K):
    device = torch.device('cuda')
    vertices = torch.tensor(smpl_v, dtype=torch.float32).reshape(1, -1, 3).to(device)
    faces = torch.tensor(smpl_f, dtype=torch.int64).reshape(1, -1, 3).to(device)


    colors_list = [
        [251 / 255.0, 128 / 255.0, 114 / 255.0],  # red
        [0.65098039, 0.74117647, 0.85882353],  # blue
        [0.9, 0.7, 0.7],  # pink
    ]
    textures = torch.tensor(colors_list[1], dtype=torch.float32).reshape(1, 1, 1, 1, 1, 3).repeat(1, faces.shape[1], 1, 1, 1, 1).to(device)

    K = torch.tensor(K, dtype=torch.float32).reshape(1, 3, 3).to(device)
    R = torch.eye(3, dtype=torch.float32).reshape(1, 3, 3).to(device)
    t = torch.zeros(3, dtype=torch.float32).reshape(1, 3).to(device)

    h, w, _ = image.shape
    s = max(h, w)
    renderer = nr.renderer.Renderer(image_size=s, K=K, R=R, t=t, orig_size=s)
    
    renderer.background_color = [1, 1, 1]
    renderer.light_direction = [1, 0.5, 1]
    renderer.light_intensity_direction = 0.3
    renderer.light_intensity_ambient = 0.5

    rend, _, mask = renderer.render(vertices=vertices, faces=faces, textures=textures)
    rend = rend.clip(0, 1)

    rend = rend[0, :, :h, :w,].permute(1, 2, 0).detach().cpu().numpy()
    mask = mask[0, :h, :w, ].reshape(h, w, 1).detach().cpu().numpy().astype(np.bool)

    rend = (rend * 255).astype(np.uint8)
    mask = mask * 0.5
    image = image * (1 - mask) + rend * mask
    # image[mask] = rend[mask] * 255

    return image


def visualize_smpl_seq(args):
    device = torch.device('cuda')

    metadata = BEHAVEExtendMetaData(args.root_dir)
    output_dir = '/inspurfs/group/wangjingya/huochf/datasets_hot_data/BEHAVE_extend/prohmr'
    prohmr_results_load = load_pickle(os.path.join(output_dir, '{}.pkl'.format(args.seq_name)))
    smpl = SMPLHLayer('/public/home/huochf/projects/3D_HOI/hoiYouTube/data/smpl/smplh/', gender='male').to(device)

    smpl_f = torch.tensor(np.array(smpl.faces).astype(np.int64)).reshape(-1, 3).to(device)

    for cam_id in prohmr_results_load:
        smpl_params = prohmr_results_load[cam_id]
        n_seq = smpl_params['betas'].shape[0]

        video = cv2.VideoWriter(os.path.join('__debug__', '{}_{}.mp4'.format(args.seq_name, cam_id)), 
            cv2.VideoWriter_fourcc(*'mp4v'), 30, (2048, 1536))

        n_seq = 128
        for i in tqdm(range(n_seq)):
            img_id = smpl_params['img_ids'][i]
            betas = torch.tensor(smpl_params['betas'][i]).reshape(1, 10).float().to(device)
            global_orient = torch.tensor(smpl_params['global_orient_rotmat'][i]).reshape(1, 3, 3).float().to(device)
            body_pose = torch.tensor(smpl_params['body_pose_rotmat'][i]).reshape(1, 21, 3, 3).to(device)
            trans = torch.tensor(smpl_params['transl'][i]).reshape(1, 3).to(device)
            cam_K = smpl_params['cam_Ks'][i]

            smpl_out = smpl(betas=betas, global_orient=global_orient, body_pose=body_pose)
            smpl_J = smpl_out.joints[0].detach()
            smpl_v = smpl_out.vertices[0].detach()
            smpl_v = smpl_v - smpl_J[:1] + trans

            image = cv2.imread(metadata.get_image_path(img_id))
            image = render_smpl(image, smpl_v, smpl_f, cam_K)

            video.write(image.astype(np.uint8))
        video.release()
        print('save to {}.'.format(os.path.join('__debug__', '{}_{}.mp4'.format(args.seq_name, cam_id))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', default='/public/home/huochf/datasets/BEHAVE/', type=str, help='Dataset root directory.')
    parser.add_argument('--seq_name', default=None)
    args = parser.parse_args()

    visualize_smpl_seq(args)

