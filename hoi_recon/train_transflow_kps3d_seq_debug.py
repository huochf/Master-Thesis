import os
import sys
import argparse
import cv2
import numpy as np
from tqdm import tqdm
from scipy.spatial.transform import Rotation
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F

from hoi_recon.models.transflow import TransFlow
from hoi_recon.datasets.behave_extend_metadata import BEHAVEExtendMetaData
from hoi_recon.datasets.utils import save_pickle, load_pickle
from hoi_recon.utils.visualization import visualize_sparse_keypoints_3d
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

        smpl_kps_seq = np.array(smpl_kps_seq).astype(np.float32)
        object_kps_seq = np.array(object_kps_seq).astype(np.float32)
        hoi_kps_seq = np.concatenate([smpl_kps_seq, object_kps_seq], axis=1)
        pos_seq = np.arange(len(smpl_kps_seq)).astype(np.int64)

        return hoi_kps_seq, pos_seq


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


class Model(nn.Module):

    def __init__(self, flow_dim, flow_width, pos_dim, seq_len, c_dim, num_blocks_per_layers, layers, head_dim, num_heads, dropout_probability):
        super().__init__()
        self.seq_len = seq_len

        self.pose_embedding = nn.Linear(flow_dim, c_dim)
        self.transflow = TransFlow(flow_dim, flow_width, pos_dim, seq_len, c_dim, num_blocks_per_layers, layers, head_dim, num_heads, dropout_probability)


    def log_prob(self, x, pos, ref_pose):
        pose_condition = self.pose_embedding(ref_pose)
        log_prob = self.transflow.log_prob(x, pos, condition=pose_condition)
        return log_prob


    def sampling(self, n_samples, pos, ref_pose, z_std=1.):
        pose_condition = self.pose_embedding(ref_pose)
        x, log_prob = self.transflow.sampling(n_samples, pos, condition=pose_condition, z_std=z_std)
        return x, log_prob


def get_loss_weights(window_radius, alpha):
    weights = (torch.arange(window_radius * 2 + 1) - window_radius) / window_radius
    weights = alpha * weights ** 2
    weights = (weights + 1) * torch.exp(- weights)
    return weights


def sampling_and_visualization(args, model, res_pose, pos_seq, gt_hoi_kps_seq, output_dir, epoch):
    res = 256
    n_samples = 8
    z_stds = [0., 0.1, 0.2, 0.5, 0.75, 1.0]
    batch_size = pos_seq.shape[0]

    samples_all = []
    for z_std in z_stds:
        kps_seq, _ = model.sampling(n_samples, pos_seq, res_pose, z_std=z_std)
        seq_n = kps_seq.shape[2]
        kps_seq = kps_seq[:, :, :].reshape(n_samples * batch_size, seq_n, -1, 3)

        for i in range(n_samples * batch_size):
            for j in range(seq_n):
                smpl_kps = kps_seq[i, j, :22].detach().cpu().numpy()
                object_kps = kps_seq[i, j, 22:].detach().cpu().numpy()
                image = visualize_sparse_keypoints_3d(smpl_kps, object_kps, args.object, res)
                samples_all.append(image)

    samples_all = np.array(samples_all).reshape(len(z_stds), batch_size, n_samples, seq_n, res, res, 3)
    samples_all = samples_all.transpose(1, 3, 0, 4, 2, 5, 6).reshape(batch_size, seq_n, len(z_stds) * res, n_samples * res, 3)

    gt_hoi_kps_seq = gt_hoi_kps_seq.reshape(batch_size, seq_n, -1, 3)
    gt_visualizations = []
    for i in range(batch_size):
        for k in range(seq_n):
            smpl_kps = gt_hoi_kps_seq[i, k, :22].detach().cpu().numpy()
            object_kps = gt_hoi_kps_seq[i, k, 22:].detach().cpu().numpy()
            image = visualize_sparse_keypoints_3d(smpl_kps, object_kps, args.object, res)
            gt_visualizations.append(image)
    gt_visualizations = np.array(gt_visualizations).reshape(batch_size, seq_n, 1, res, res, 3)
    gt_visualizations = gt_visualizations.repeat(len(z_stds), axis=2).reshape(batch_size, seq_n, len(z_stds) * res, res, 3)

    images = gt_visualizations[:, 15:16].repeat(seq_n, axis=1)
    samples_all = np.concatenate([images, gt_visualizations, samples_all], axis=3)

    for i in range(min(batch_size, 8)):
        video = cv2.VideoWriter(os.path.join(output_dir, '{:04d}_{:03d}.mp4'.format(epoch, i)), 
            cv2.VideoWriter_fourcc(*'mp4v'), 15, (res * (n_samples + 2), len(z_stds) * res))
        for j in range(seq_n):
            video.write(samples_all[i, j].astype(np.uint8))
        video.release()


def train(args):
    device = torch.device('cuda')
    batch_size = args.batch_size
    output_dir = args.save_dir
    os.makedirs(output_dir, exist_ok=True)

    dataset_train = BEHAVE3DKpsSeqDataset(args.root_dir, args.object, args.window_radius, args.fps, split='train')
    dataset_test = BEHAVE3DKpsSeqDataset(args.root_dir, args.object, args.window_radius, args.fps, split='test')
    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, num_workers=8, shuffle=True)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=4, num_workers=2, shuffle=False)

    flow_dim = (22 + dataset_train.metadata.object_num_keypoints[args.object]) * 3
    model = Model(flow_dim=flow_dim, 
                      flow_width=args.flow_width, 
                      pos_dim=args.pos_dim, 
                      seq_len=args.window_radius * 2 + 1, 
                      c_dim=args.c_dim,
                      num_blocks_per_layers=args.num_blocks_per_layers, 
                      layers=args.layers,
                      head_dim=args.head_dim,
                      num_heads=args.num_heads,
                      dropout_probability=args.dropout_probability)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    if False and os.path.exists(os.path.join(output_dir, 'checkpoint_{}.pth'.format(args.object))):
        state_dict = torch.load(os.path.join(output_dir, 'checkpoint_{}.pth'.format(args.object)))
        model.load_state_dict(state_dict['model'])
        optimizer.load_state_dict(state_dict['optimizer'])
        begin_epoch = state_dict['epoch']
    else:
        begin_epoch = 0

    f_log = open(os.path.join(output_dir, 'logs_{}.txt'.format(args.object)), 'a')

    loss_weights = get_loss_weights(args.window_radius, args.temporal_alpha)
    loss_weights = loss_weights.reshape(1, -1).float().to(device)

    for epoch in range(begin_epoch, args.epoch):
        model.train()

        for idx, item in enumerate(dataloader_train):
            hoi_kps_seq, pos_seq = item
            hoi_kps_seq = hoi_kps_seq.float().to(device)
            pos_seq = pos_seq.long().to(device)

            batch_size, seq_n = hoi_kps_seq.shape[:2]
            hoi_kps_seq = hoi_kps_seq.reshape(batch_size, seq_n, -1)

            hoi_kps_seq = hoi_kps_seq + 0.001 * torch.randn_like(hoi_kps_seq)

            ref_pose = hoi_kps_seq[:, 15:16].repeat(1, seq_n, 1)

            condition = model.pose_embedding(ref_pose)
            z, _ = model.transflow.forward(hoi_kps_seq, pos_seq, condition)
            x_inverse, _ = model.transflow.inverse(z, pos_seq, condition)
            print(hoi_kps_seq[0], x_inverse[0])

            hoi_kps_seq[:, :14] = 0
            hoi_kps_seq[:, 16:] = 0
            _z, _ = model.transflow.forward(hoi_kps_seq, pos_seq, condition)
            print(z[0, 14], _z[0, 14])
            print(z[0, 15], _z[0, 15])
            exit(0)
            log_prob = model.log_prob(hoi_kps_seq, pos_seq, ref_pose)
            loss_nll = - (loss_weights * log_prob).mean()

            loss = loss_nll

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()

            if idx % 10 == 0:
                log_str = '[{} / {}] Loss: {:.4f}, Loss_nll: {:.4f}'.format(
                    epoch, idx, loss.item(), loss_nll.item())
                print(log_str)
                sys.stdout.flush()
                f_log.write(log_str + '\n')

        if epoch % 10 == 0:
            model.eval()
            for idx, item in enumerate(dataloader_test):
                if idx > 50:
                    break
                hoi_kps_seq, pos_seq = item
                hoi_kps_seq = hoi_kps_seq.float().to(device)
                pos_seq = pos_seq.long().to(device)

                batch_size, seq_n = hoi_kps_seq.shape[:2]
                hoi_kps_seq = hoi_kps_seq.reshape(batch_size, seq_n, -1)

                ref_pose = hoi_kps_seq[:, 15:16].repeat(1, seq_n, 1)

                log_prob = model.log_prob(hoi_kps_seq, pos_seq, ref_pose)
                loss_nll = - (loss_weights * log_prob).mean()

                loss = loss_nll

                if idx % 10 == 0:
                    _output_dir = os.path.join(output_dir, '{}_vis'.format(args.object))
                    os.makedirs(_output_dir, exist_ok=True)
                    sampling_and_visualization(args, model, ref_pose, pos_seq, hoi_kps_seq, _output_dir, epoch)
                    log_str = '[EVAL {} / {}] Loss: {:.4f}, Loss_nll: {:.4f}'.format(
                        epoch, idx, loss.item(), loss_nll.item())
                    print(log_str)
                    sys.stdout.flush()
                    
                    f_log.write(log_str + '\n')

            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }, os.path.join(output_dir, 'checkpoint_{}.pth'.format(args.object)))

    f_log.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate KPS (BEHAVE)')
    parser.add_argument('--root_dir', default='/storage/data/huochf/BEHAVE', type=str)
    parser.add_argument('--epoch', default=999999, type=int)
    parser.add_argument('--object', default='backpack', type=str)
    parser.add_argument('--window_radius', default=15, type=int)
    parser.add_argument('--fps', default=30, type=int)
    parser.add_argument('--pos_dim', default=256, type=int)
    parser.add_argument('--flow_width', default=512, type=int)
    parser.add_argument('--num_blocks_per_layers', default=2, type=int)
    parser.add_argument('--layers', default=4, type=int)
    parser.add_argument('--head_dim', default=256, type=int)
    parser.add_argument('--num_heads', default=8, type=int)
    parser.add_argument('--c_dim', default=256, type=int)
    parser.add_argument('--dropout_probability', default=0., type=float)
    parser.add_argument('--temporal_alpha', default=1., type=float)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--save_dir', default='./outputs/transflow_kps3d_seq_debug')

    args = parser.parse_args()

    train(args)
