import os
import sys
import argparse

import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F

from hoi_recon.models.transflow import TransFlow
from hoi_recon.datasets.behave_extend_metadata import BEHAVEExtendMetaData
from hoi_recon.datasets.behave_pseudo_kps2d_seq_dataset import BEHAVEPseudoKps2DSeqDataset
from hoi_recon.datasets.utils import save_pickle, load_pickle


class Model(nn.Module):

    def __init__(self, flow_dim, flow_width, pos_dim, seq_len, c_dim, num_blocks_per_layers, layers, head_dim, num_heads, dropout_probability):
        super().__init__()
        self.seq_len = seq_len
        resnet = torchvision.models.resnet50(pretrained=True)
        modules = list(resnet.children())[:-1]
        modules.append(nn.Conv2d(resnet.fc.in_features, 2048, kernel_size=1))
        self.backbone = nn.Sequential(*modules)

        self.visual_embedding = nn.Linear(2048, c_dim)
        self.transflow = TransFlow(flow_dim, flow_width, pos_dim, seq_len, c_dim, num_blocks_per_layers, layers, head_dim, num_heads, dropout_probability)


    def image_embedding(self, images):
        return self.backbone(images)


    def log_prob(self, x, pos, visual_feats):
        visual_condition = self.visual_embedding(visual_feats)
        log_prob = self.transflow.log_prob(x, pos, condition=visual_condition)
        return log_prob


    def sampling(self, n_samples, pos, visual_feats, z_std=1.):
        visual_condition = self.visual_embedding(visual_feats)
        x, log_prob = self.transflow.sampling(n_samples, pos, condition=visual_condition, z_std=z_std)
        return x, log_prob


def get_loss_weights(window_radius, alpha):
    weights = (torch.arange(window_radius * 2 + 1) - window_radius) / window_radius
    weights = alpha * weights ** 2
    weights = (weights + 1) * torch.exp(- weights)
    return weights


def train(args):
    device = torch.device('cuda')
    batch_size = args.batch_size
    output_dir = args.save_dir
    os.makedirs(output_dir, exist_ok=True)

    dataset_train = BEHAVEPseudoKps2DSeqDataset(args.root_dir, args.object, args.window_radius, args.fps, use_gt_grouping=False, split='train')
    dataset_test = BEHAVEPseudoKps2DSeqDataset(args.root_dir, args.object, args.window_radius, args.fps, use_gt_grouping=True, split='test')
    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, num_workers=8, shuffle=True)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=4, num_workers=2, shuffle=False)

    flow_dim = (22 + dataset_train.metadata.object_num_keypoints[args.object] + 1) * 3
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

    if os.path.exists(os.path.join(output_dir, 'checkpoint_{}.pth'.format(args.object))):
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
            image, hoi_kps_seq, pos_seq = item
            image = image.float().to(device)
            hoi_kps_seq = hoi_kps_seq.float().to(device)
            pos_seq = pos_seq.long().to(device)

            batch_size, seq_n, _, _ = hoi_kps_seq.shape
            hoi_kps_seq = hoi_kps_seq.reshape(batch_size, seq_n, -1)

            hoi_kps_seq = hoi_kps_seq + 0.0001 * torch.randn_like(hoi_kps_seq)

            visual_feats = model.image_embedding(image)
            visual_feats = visual_feats.reshape(batch_size, 1, -1).repeat(1, seq_n, 1)

            log_prob = model.log_prob(hoi_kps_seq, pos_seq, visual_feats)
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

            if idx % 1000 == 0:
                torch.save({
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict()
                }, os.path.join(output_dir, 'checkpoint_{}.pth'.format(args.object)))

        model.eval()
        for idx, item in enumerate(dataloader_test):
            if idx > 50:
                break
            image, hoi_kps_seq, pos_seq = item
            image = image.float().to(device)
            hoi_kps_seq = hoi_kps_seq.float().to(device)
            pos_seq = pos_seq.long().to(device)

            batch_size, seq_n, _, _ = hoi_kps_seq.shape
            hoi_kps_seq = hoi_kps_seq.reshape(batch_size, seq_n, -1)

            visual_feats = model.image_embedding(image)
            visual_feats = visual_feats.reshape(batch_size, 1, -1).repeat(1, seq_n, 1)

            log_prob = model.log_prob(hoi_kps_seq, pos_seq, visual_feats)
            loss_nll = - (loss_weights * log_prob).mean()

            loss = loss_nll

            if idx % 10 == 0:
                log_str = '[EVAL {} / {}] Loss: {:.4f}, Loss_nll: {:.4f}'.format(
                    epoch, idx, loss.item(), loss_nll.item())
                print(log_str)
                sys.stdout.flush()
                
                f_log.write(log_str + '\n')

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
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--save_dir', default='./outputs/transflow_pseudo_kps2d_seq')

    args = parser.parse_args()

    train(args)
