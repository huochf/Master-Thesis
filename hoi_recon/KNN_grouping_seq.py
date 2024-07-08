import os
import pickle
import numpy as np
from scipy.spatial.transform import Rotation
import argparse
import cv2
import random
import heapq
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from hoi_recon.datasets.utils import load_json, load_pickle, save_pickle, save_json
from hoi_recon.datasets.behave_extend_metadata import BEHAVEExtendMetaData


class KNN:

    def __init__(self, root_dir, object_name, k, k_sampling, self_weight, temporal_window_radius, fps, sim_threshold=0.05, alpha=1., device=torch.device('cuda')):
        self.root_dir = root_dir
        self.object_name = object_name
        self.k = k
        self.k_sampling = k_sampling
        self.sim_threshold = sim_threshold
        self.self_weight = self_weight
        self.temporal_window_radius = temporal_window_radius
        self.frame_steps = 30 // fps
        self.alpha = alpha
        self.device = device

        self.metadata = BEHAVEExtendMetaData(root_dir)

        self.item_ids, self.hoi_kps_seq_all = self.load_kps()
        print('loaded {} frames.'.format(len(self.item_ids)))
        self._nn = self.init_nn(self.k)
        self._heaps = self.init_heap()


    def load_kps(self, ):
        datalist_dir = './data/datasets/behave_extend_datalist'
        hoi_kps_seq_all = {}
        item_ids = []
        print('loading hoi kps ...')
        for file in os.listdir(datalist_dir):
            seq_name = file.split('.')[0]
            hoi_kps_seq_all[seq_name] = {}
            if seq_name not in self.metadata.dataset_splits['train']:
                continue
            object_name = seq_name.split('_')[2]
            if object_name != self.object_name:
                continue

            annotation_list = load_pickle(os.path.join(datalist_dir, file))
            for cam_id in annotation_list:
                hoi_kps_seq_all[seq_name][cam_id] = []
                for idx, item in enumerate(annotation_list[cam_id]):
                    hoi_kps_seq_all[seq_name][cam_id].append(self.kps3dfy(item))
                    item_ids.append('_'.join([seq_name, str(cam_id), str(idx)]))

        return item_ids, hoi_kps_seq_all


    def load_item_kps(self, item_id):
        seq_name = '_'.join(item_id.split('_')[:-2])
        cam_id = int(item_id.split('_')[-2])
        frame_idx = int(item_id.split('_')[-1])
        kps_list = self.hoi_kps_seq_all[seq_name][cam_id]
        kps_prev = list(reversed(kps_list[:frame_idx + 1]))[::self.frame_steps]
        kps_prev = list(reversed(kps_prev))

        if len(kps_prev) > self.temporal_window_radius:
            kps_prev = kps_prev[- self.temporal_window_radius - 1:]
        else: # padding
            while len(kps_prev) <= self.temporal_window_radius:
                kps_prev.insert(0, kps_prev[0])

        kps_succ = kps_list[frame_idx:][::self.frame_steps]
        if len(kps_succ) > self.temporal_window_radius:
            kps_succ = kps_succ[: self.temporal_window_radius + 1]
        else: # padding
            while len(kps_succ) <= self.temporal_window_radius:
                kps_succ.append(kps_succ[-1])

        kps_seq = kps_prev[:-1] + kps_succ
        assert len(kps_seq) == 2 * self.temporal_window_radius + 1

        return np.stack(kps_seq, axis=0)


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

        return hoi_kps


    def init_nn(self, k):
        neighbors = {}

        print('initialize k nearest neighbors ...')
        for item_id in self.item_ids:
            neighbors[item_id] = {
                'neighbors': [],
                'neighbors_inverse': [],
                'new': [],
                'new_inverse': [],
            }

        for item_id in tqdm(self.item_ids):
            random_indices = np.random.choice(len(self.item_ids), k, replace=False)
            for idx in random_indices:
                neighbors[item_id]['neighbors'].append(self.item_ids[idx])
                neighbors[item_id]['new'].append(True)
                neighbors[self.item_ids[idx]]['neighbors_inverse'].append(item_id)
                neighbors[self.item_ids[idx]]['new_inverse'].append(True)

        return neighbors


    class DatasetInit:

        def __init__(self, knn):
            self.knn = knn


        def __len__(self, ):
            return len(self.knn.item_ids)


        def __getitem__(self, idx):
            item_id = self.knn.item_ids[idx]

            neighbors = self.knn._nn[item_id]['neighbors']
            kps = self.knn.load_item_kps(item_id) # [T, n, 3]
            kps_nn = np.stack([self.knn.load_item_kps(nn_id) for nn_id in neighbors], axis=0) # [m, T, n, 3]

            return item_id, neighbors, kps, kps_nn


    def init_heap(self, ):
        heaps = {}
        print('building heaps ...')
        batch_size = 128

        dataset = self.DatasetInit(self)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=1, shuffle=False, drop_last=False)

        for item in tqdm(dataloader):
            item_ids, neighbors, kps, kps_nn = item
            kps = kps.float().to(self.device) # [b, T, n, 3]
            kps_nn = kps_nn.float().to(self.device) # [b, k, T, n, 3]
            b, k, T, n, _ = kps_nn.shape

            distance_pairwise = self.cross_view_distance(kps_nn, kps_nn).mean(-1) # [b, k]
            distance_main = self.cross_view_distance(kps.unsqueeze(1), kps_nn).squeeze(1) # [b, k]
            distance = (1 - self.self_weight) * distance_main + self.self_weight * distance_pairwise

            for b_idx in range(b):
                item_id = item_ids[b_idx]
                if item_id not in heaps:
                    heaps[item_id] = []

                for nn_idx in range(k):
                    nn_id = neighbors[nn_idx][b_idx]

                    heaps[item_id].append([- distance[b_idx, nn_idx].item(), nn_id])

        for item_id in heaps.keys():
            heapq.heapify(heaps[item_id])

        return heaps


    def cross_view_distance(self, kps1, kps2):
        # kps1: [b, m1, T, n, 3], kps2: [b, m2, T, n, 3]
        b, m1, T, n, _ = kps1.shape
        m2 = kps2.shape[1]
        kps1_directions = kps1[:, :, :, :-1] - kps1[:, :, :, -1:]
        kps2_directions = kps2[:, :, :, :-1] - kps2[:, :, :, -1:]

        center_lines = kps1[:, :, :, -1].view(b, m1, 1, T, 3) - kps2[:, :, :, -1].view(b, 1, m2, T, 3)
        cross = torch.cross(kps1_directions.view(b, m1, 1, T, -1, 3), kps2_directions.view(b, 1, m2, T, -1, 3), dim=-1)
        distance = torch.abs((center_lines.view(b, m1, m2, T, 1, 3) * cross).sum(-1)) / (torch.norm(cross, dim=-1) + 1e-8)
        distance = distance.mean(-1) # [b, m1, m2, T]

        weights = self.alpha * (torch.arange(T).float().to(distance.device) - T // 2) ** 2
        weights = (weights + 1) * torch.exp(- weights)
        weights = weights.view(1, 1, 1, T)
        distance = (distance * weights).mean(-1)
        return distance


    def get_loss_weights(self, window_radius, alpha):
        weights = (torch.arange(window_radius * 2 + 1) - window_radius) / window_radius
        weights = alpha * weights ** 2
        weights = (weights + 1) * torch.exp(- weights)
        return weights


    def kps3d_distance(self, kps1, kps2):
        # kps1: [b, m1, T, n, 3], kps2: [b, m2, T, n, 3]
        b, m1, T, n, _ = kps1.shape
        m2 = kps2.shape[1]
        kps1 = kps1 / torch.norm(kps1, dim=-1, keepdim=True)
        kps2 = kps2 / torch.norm(kps2, dim=-1, keepdim=True)
        distance = torch.abs(kps1.view(b, m1, 1, T, n, 3) - kps2.view(b, 1, m2, T, n, 3)).reshape(b, m1, m2, T, -1).mean(-1)

        weights = self.get_loss_weights(self.temporal_window_radius, self.alpha)
        weights = weights.view(1, 1, 1, T).float().to(distance.device)
        distance = (distance * weights).mean(-1)
        return distance


    class DatasetStep:

        def __init__(self, knn):
            self.knn = knn
            self.comparison_list = self.get_comparison_list()


        def get_comparison_list(self, ):
            comparison_list = []
            for item_id in self.knn.item_ids:

                neighbors = self.knn._nn[item_id]['neighbors']
                new = self.knn._nn[item_id]['new']
                neighbors_inverse = self.knn._nn[item_id]['neighbors_inverse']
                new_inverse = self.knn._nn[item_id]['new_inverse']

                neighbors_all = neighbors + neighbors_inverse
                random_indices = np.random.choice(len(neighbors_all), min(self.knn.k_sampling, len(neighbors_all)), replace=False)
                neighbors_all = [neighbors_all[idx] for idx in random_indices]

                neighbors_new = []
                neighbors_new_idx, neighbors_new_inverse_idx = [], []
                for i in range(len(new)):
                    if new[i]:
                        neighbors_new.append(neighbors[i])
                        neighbors_new_idx.append(i)
                for i in range(len(new_inverse)):
                    if new_inverse[i]:
                        neighbors_new.append(neighbors_inverse[i])
                        neighbors_new_inverse_idx.append(i)

                random_indices = np.random.choice(len(neighbors_new), min(self.knn.k_sampling, len(neighbors_new)), replace=False)
                neighbors_new = [neighbors_new[idx] for idx in random_indices]
                for idx in random_indices:
                    if idx >= len(neighbors_new_idx):
                        new_inverse[neighbors_new_inverse_idx[idx - len(neighbors_new_idx)]] = False
                    else:
                        new[neighbors_new_idx[idx]] = False

                for id1 in neighbors_new:
                    for id2 in neighbors_all:
                        if id1 != id2:
                            comparison_list.append((id1, id2))

            return comparison_list


        def __len__(self, ):
            return len(self.comparison_list)


        def __getitem__(self, idx):
            id1, id2 = self.comparison_list[idx]

            kps1 = self.knn.load_item_kps(id1) # [T, n, 3]
            kps2 = self.knn.load_item_kps(id2) # [T, n, 3]
            kps1_nn = np.stack([self.knn.load_item_kps(item_id) for item_id in self.knn._nn[id1]['neighbors']], axis=0)
            kps2_nn = np.stack([self.knn.load_item_kps(item_id) for item_id in self.knn._nn[id2]['neighbors']], axis=0)

            return id1, id2, kps1, kps2, kps1_nn, kps2_nn


    def step(self, ):
        change_count = 0

        batch_size = 4096
        dataset = self.DatasetStep(self)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=1, shuffle=False, drop_last=False)

        for item in tqdm(dataloader):

            id1s, id2s, kps1s, kps2s, kps1s_nn, kps2s_nn = item
            kps1s = kps1s.float().to(self.device)
            kps2s = kps2s.float().to(self.device) # [b, T, n, 3]
            kps1s_nn = kps1s_nn.float().to(self.device)
            kps2s_nn = kps2s_nn.float().to(self.device) # [b, k, T, n, 3]

            b, k, T, n, _ = kps2s_nn.shape

            def step_item(target_ids, target_kps, ids, kps, kps_nn):
                _change_count = 0
                # insert target_kps to kps
                distance_pairwise = self.cross_view_distance(target_kps.unsqueeze(1), kps_nn).squeeze(1).mean(-1) # [b, ]
                distance_main = self.cross_view_distance(target_kps.unsqueeze(1), kps.unsqueeze(1)).reshape(-1) # [b, ]
                distance = (1 - self.self_weight) * distance_pairwise + self.self_weight * distance_main

                similarity, _ = self.kps3d_distance(target_kps.unsqueeze(1), kps2s_nn).squeeze(1).min(1) # [b, ]

                for b_idx in range(b):
                    distance_max = - self._heaps[ids[b_idx]][0][0]

                    if distance[b_idx] < distance_max and similarity[b_idx] > self.sim_threshold and target_ids[b_idx] not in self._nn[ids[b_idx]]['neighbors']:
                        _, item_id_del = heapq.heapreplace(self._heaps[ids[b_idx]], [-distance[b_idx].item(), target_ids[b_idx]])

                        nn_idx1 = self._nn[ids[b_idx]]['neighbors'].index(item_id_del)
                        self._nn[ids[b_idx]]['neighbors'][nn_idx1] = target_ids[b_idx]
                        self._nn[ids[b_idx]]['new'][nn_idx1] = True
                        nn_idx2 = self._nn[item_id_del]['neighbors_inverse'].index(ids[b_idx])
                        self._nn[item_id_del]['neighbors_inverse'].pop(nn_idx2)
                        self._nn[item_id_del]['new_inverse'].pop(nn_idx2)
                        self._nn[target_ids[b_idx]]['neighbors_inverse'].append(ids[b_idx])
                        self._nn[target_ids[b_idx]]['new_inverse'].append(True)

                        _change_count += 1
                return _change_count

            change_count += step_item(id1s, kps1s, id2s, kps2s, kps2s_nn)
            change_count += step_item(id2s, kps2s, id1s, kps1s, kps1s_nn)

        return change_count


    class DatasetUpdate:

        def __init__(self, knn):
            self.knn = knn
            self.item_ids = self.collect_item_ids()


        def collect_item_ids(self, ):
            item_ids = []
            for item_id in self.knn.item_ids:
                new = self.knn._nn[item_id]['new']
                for i in range(len(new)):
                    if new[i]:
                        item_ids.append(item_id)
                        break
            return item_ids


        def __len__(self, ):
            return len(self.item_ids)


        def __getitem__(self, idx):
            item_id = self.item_ids[idx]

            neighbors = [items[1] for items in self.knn._heaps[item_id]]
            kps = self.knn.load_item_kps(item_id) # [T, n, 3]
            kps_nn = np.stack([self.knn.load_item_kps(item_id) for item_id in neighbors], axis=0) # [m, T, n, 3]

            return item_id, neighbors, kps, kps_nn


    def update_distance(self, ):
        print('update distance ...')
        batch_size = 128
        
        dataset = self.DatasetUpdate(self)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=1, shuffle=False, drop_last=False)

        for item in tqdm(dataloader):
            item_ids, neighbors, kps, kps_nn = item
            kps = kps.float().to(self.device) # [b, T, n, 3]
            kps_nn = kps_nn.float().to(self.device) # [b, k, T, n, 3]
            b, k, _, _ = kps_nn.shape

            distance_pairwise = self.cross_view_distance(kps_nn, kps_nn).mean(-1) # [b, k]
            distance_main = self.cross_view_distance(kps.unsqueeze(1), kps_nn).squeeze(1) # [b, k]
            distance = (1 - self.self_weight) * distance_main + self.self_weight * distance_pairwise

            for b_idx in range(b):
                item_id = item_ids[b_idx]

                for nn_idx in range(k):
                    nn_id = neighbors[nn_idx][b_idx]
                    self._heaps[item_id][nn_idx][0] = - distance[b_idx][nn_idx].item()

        for item_id in self._heaps.keys():
            heapq.heapify(self._heaps[item_id])


    def save(self, path):
        save_pickle(self._heaps, path)


def main(args):
    output_dir = args.save_dir
    os.makedirs(output_dir, exist_ok=True)

    knn = KNN(root_dir=args.root_dir, 
              object_name=args.object, 
              k=args.k, 
              k_sampling=args.k, 
              self_weight=args.self_weight,
              temporal_window_radius=args.window_radius,
              fps=args.fps,
              alpha=args.alpha)
    for i in range(args.n_steps):
        n_changes = knn.step()
        print('Iter: {}, changes: {}'.format(i, n_changes))
        if args.self_weight != 1:
            knn.update_distance()

    knn.save(os.path.join(output_dir, 'knn_groups_{}_k_{:02d}_win_{:02d}_fps_{:02d}.pkl'.format(
        args.object, args.k, args.window_radius, args.fps)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='KNN Grouping on BEHAVE dataset')
    parser.add_argument('--root_dir', default='/storage/data/huochf/BEHAVE/')
    parser.add_argument('--object', default='backpack')
    parser.add_argument('--k', type=int, default=8)
    parser.add_argument('--self_weight', type=float, default=1.)
    parser.add_argument('--n_steps', type=int, default=20)
    parser.add_argument('--window_radius', type=int, default=15)
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--alpha', type=float, default=1) # the lower -> mean among time steps, the higher -> drop the temporal info.
    parser.add_argument('--save_dir', type=str, default='./outputs/behave_knn_grouping')

    args = parser.parse_args()
    main(args)
