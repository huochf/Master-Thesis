import os
import sys
file_dir = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(file_dir, '..', ))
sys.path.insert(0, os.path.join(file_dir, '/public/home/huochf/projects/3D_HOI/StackFLOW/', ))
import numpy as np
import argparse
import json
import random
from tqdm import tqdm

import torch
torch.multiprocessing.set_sharing_strategy('file_system')
from hoi_recon.configs.stackflow_config import load_config
from hoi_recon.models.stackflow import Model
# from hoi_recon.configs.stackflow_obj_RT_config import load_config
# from hoi_recon.models.stackflow_obj_RT import Model
from hoi_recon.datasets.behave_hoi_dataset import BEHAVEExtendDataset
from hoi_recon.datasets.behave_extend_metadata import BEHAVEExtendMetaData
from hoi_recon.datasets.utils import load_pickle

from hoi_recon.utils.visualization_stackflow import visualize_step


def to_device(batch, device):
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(device)
    return batch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_annotations(metadata, split):
    annotations_all = {}
    seq_names = list(metadata.go_through_all_sequences(split=split))
    print('loadding hoi boxes ...')
    for seq_name in tqdm(seq_names):
        annotations = load_pickle('data/datasets/behave_extend_datalist/{}.pkl'.format(seq_name))
        for cam_id, item_list in annotations.items():
            for item in item_list:
                annotations_all[item['img_id']] = {
                    'img_id': item['img_id'],
                    'hoi_bb_xyxy': item['hoi_bb_xyxy'],
                    'smplh_betas_male': item['smplh_betas_male'],
                    'smplh_pose_rotmat': item['smplh_pose_rotmat'],
                    'smplh_joints_3d': item['smplh_joints_3d'][:22],
                    'smplh_joints_2d': item['smplh_joints_2d'][:22],
                    'obj_keypoints_3d': item['obj_keypoints_3d'],
                    'obj_keypoints_2d': item['obj_keypoints_2d'],
                    'object_rel_trans': item['object_rel_trans'],
                    'object_rel_rotmat': item['object_rel_rotmat'],
                    'hoi_trans': item['hoi_trans'],
                    'hoi_rotmat': item['hoi_rotmat'],
                    'cam_K': item['cam_K'],
                }
    return annotations_all


def train(cfg):
    device = torch.device('cuda')
    metadata = BEHAVEExtendMetaData(cfg.dataset.root_dir)
    annotation_train_shared = load_annotations(metadata, 'train')
    annotation_test_shared = load_annotations(metadata, 'test')

    train_dataset = BEHAVEExtendDataset(cfg, annotation_train_shared, is_train=True)
    test_dataset = BEHAVEExtendDataset(cfg, annotation_test_shared, is_train=False)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, 
                                                   batch_size=cfg.train.batch_size,
                                                   num_workers=cfg.train.num_workers,
                                                   shuffle=True,
                                                   drop_last=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, 
                                                   batch_size=cfg.train.batch_size,
                                                   num_workers=2,
                                                   shuffle=True,
                                                   drop_last=True)

    model = Model(cfg)
    model.to(device)

    begin_epoch = 0
    if cfg.train.resume and os.path.exists(cfg.train.resume):
        begin_epoch = model.load_checkpoint(cfg.train.resume)

    for epoch in range(begin_epoch, cfg.train.max_epoch):
        if epoch == cfg.train.drop_lr_at:
            for param_group in model.optimizer.param_groups:
                param_group['lr'] *= 0.1
        model.train()
        if epoch > cfg.train.trans_begin_epoch:
            model.loss_weights['loss_trans'] = 0.1
        else:
            model.loss_weights['loss_trans'] = 0.
        for idx, batch in enumerate(train_dataloader):
            batch = to_device(batch, device)
            loss, all_losses = model.train_step(batch)

            if idx % cfg.train.log_interval == 0:
                loss_str = '[{}, {}], loss: {:.4f}'.format(epoch, idx, loss.item())
                for k, v in all_losses.items():
                    loss_str += ', {}: {:.4f}'.format(k, v.item())
                loss_str += ', {}: {:.5f}'.format('lr', model.optimizer.state_dict()['param_groups'][0]['lr'])
                print(loss_str)
                sys.stdout.flush()
        model.eval()

        eval_losses = {}
        for idx, batch in enumerate(test_dataloader):
            if idx > 100:
                break
            batch = to_device(batch, device)
            loss, all_losses = model.forward_train(batch) # no loss backward !!!

            if idx % 10 == 0:
                loss_str = 'EVAL: [{}, {}], loss: {:.4f}'.format(epoch, idx, loss.item())
                for k, v in all_losses.items():
                    loss_str += ', {}: {:.4f}'.format(k, v.item())
                print(loss_str)
                sys.stdout.flush()

            if idx % 10 == 0:
                pred = model.inference(batch, debug=True)
                visualize_step(cfg, test_dataset.dataset_metadata, batch, pred, epoch, idx)
            for k, v in all_losses.items():
                if k not in eval_losses:
                    eval_losses[k] = 0
                
                eval_losses[k] += v.item() / 100

        os.makedirs(cfg.train.output_dir, exist_ok=True)

        eval_losses['epoch'] = epoch
        with open(os.path.join(cfg.train.output_dir, 'logs.json'), 'a') as f:
            f.write(json.dumps(eval_losses) + "\n")
        model.save_checkpoint(epoch, os.path.join(cfg.train.output_dir, 'latest_checkpoint.pth'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root_dir', default='/storage/data/huochf/BEHAVE', type=str)
    args = parser.parse_args()

    cfg = load_config()
    cfg.dataset.root_dir = args.dataset_root_dir
    cfg.freeze()
    set_seed(7)
    train(cfg)
