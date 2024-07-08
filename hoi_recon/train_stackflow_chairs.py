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
from hoi_recon.configs.stackflow_chairs import load_config
from hoi_recon.models.stackflow_for_chairs import Model
# from hoi_recon.configs.stackflow_chairs_RT import load_config
# from hoi_recon.models.stackflow_for_chairs_RT import Model
from hoi_recon.datasets.chairs_hoi_dataset import CHAIRSHOIDataset
from hoi_recon.datasets.utils import load_pickle



def to_device(batch, device):
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(device)
    return batch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_annotations(split, ):
    if split == 'train':
        return load_pickle('data/datasets/chairs_train_annotations.pkl')
    else:
        return load_pickle('data/datasets/chairs_test_annotations.pkl')


def train(cfg):
    device = torch.device('cuda')
    annotation_train_shared = load_annotations('train')
    annotation_test_shared = load_annotations('test')

    dataset_root_dir = cfg.dataset.root_dir
    train_dataset = CHAIRSHOIDataset(dataset_root_dir, annotation_train_shared, is_train=True)
    test_dataset = CHAIRSHOIDataset(dataset_root_dir, annotation_test_shared, is_train=False)

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
    # if cfg.train.resume and os.path.exists(cfg.train.resume):
    #     begin_epoch = model.load_checkpoint(cfg.train.resume)

    for epoch in range(begin_epoch, cfg.train.max_epoch):
        if epoch == cfg.train.drop_lr_at:
            for param_group in model.optimizer.param_groups:
                param_group['lr'] *= 0.1
        model.train()
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
    parser.add_argument('--dataset_root_dir', default='/storage/data/huochf/CHAIRS', type=str)
    args = parser.parse_args()

    cfg = load_config()
    cfg.dataset.root_dir = args.dataset_root_dir
    cfg.freeze()
    set_seed(7)
    train(cfg)
