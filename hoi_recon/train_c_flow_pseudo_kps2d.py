import os
import sys
import argparse

import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F

from hoi_recon.models.condition_flow import ConditionFlow
from hoi_recon.datasets.behave_extend_metadata import BEHAVEExtendMetaData
from hoi_recon.datasets.behave_pseudo_kps2d_dataset import BEHAVEPseudoKps2DDataset
from hoi_recon.datasets.utils import save_pickle, load_pickle


class Model(nn.Module):

    def __init__(self, flow_dim, flow_width, c_dim, num_blocks_per_layers, layers, dropout_probability):
        super().__init__()
        resnet = torchvision.models.resnet50(pretrained=True)
        modules = list(resnet.children())[:-1]
        modules.append(nn.Conv2d(resnet.fc.in_features, 2048, kernel_size=1))
        self.backbone = nn.Sequential(*modules)

        self.visual_embedding = nn.Linear(2048, c_dim)

        self.flow = ConditionFlow(dim=flow_dim, hidden_dim=flow_width, c_dim=c_dim, num_blocks_per_layer=num_blocks_per_layers, num_layers=layers, dropout_probability=dropout_probability)


    def image_embedding(self, images):
        return self.backbone(images)


    def log_prob(self, x, visual_feats):
        visual_condition = self.visual_embedding(visual_feats)
        log_prob = self.flow.log_prob(x, condition=visual_condition)
        return log_prob


    def sampling(self, n_samples, visual_feats, z_std=1.):
        visual_condition = self.visual_embedding(visual_feats)
        x, log_prob = self.flow.sampling(n_samples, condition=visual_condition, z_std=z_std)
        return x, log_prob


def train(args):
    device = torch.device('cuda')
    batch_size = args.batch_size
    output_dir = args.save_dir
    os.makedirs(output_dir, exist_ok=True)

    dataset_train = BEHAVEPseudoKps2DDataset(args.root_dir, args.object, args.n_views, use_gt_grouping=False, split='train')
    dataset_test = BEHAVEPseudoKps2DDataset(args.root_dir, args.object, args.n_views, use_gt_grouping=True, split='test')
    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, num_workers=8, shuffle=True)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=4, num_workers=2, shuffle=False)

    flow_dim = (22 + dataset_train.metadata.object_num_keypoints[args.object] + 1) * 3
    model = Model(flow_dim=flow_dim, 
                      flow_width=args.flow_width, 
                      c_dim=args.c_dim,
                      num_blocks_per_layers=args.num_blocks_per_layers, 
                      layers=args.layers,
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

    for epoch in range(begin_epoch, args.epoch):
        model.train()

        for idx, item in enumerate(dataloader_train):
            image, hoi_kps = item
            image = image.float().to(device)
            hoi_kps = hoi_kps.float().to(device)

            batch_size, n_views, _, _ = hoi_kps.shape
            hoi_kps = hoi_kps.reshape(batch_size * n_views, -1)

            hoi_kps = hoi_kps + 0.001 * torch.randn_like(hoi_kps)

            visual_feats = model.image_embedding(image)
            visual_feats = visual_feats.reshape(batch_size, 1, -1).repeat(1, n_views, 1).reshape(batch_size * n_views, -1)

            log_prob = model.log_prob(hoi_kps, visual_feats)
            loss_nll = - log_prob.mean()

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
            image, hoi_kps = item
            image = image.float().to(device)
            hoi_kps = hoi_kps.float().to(device)

            batch_size, n_views, _, _ = hoi_kps.shape
            hoi_kps = hoi_kps.reshape(batch_size * n_views, -1)

            visual_feats = model.image_embedding(image)
            visual_feats = visual_feats.reshape(batch_size, 1, -1).repeat(1, n_views, 1).reshape(batch_size * n_views, -1)

            log_prob = model.log_prob(hoi_kps, visual_feats)
            loss_nll = - log_prob.mean()

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
    parser.add_argument('--n_views', default=8, type=int)
    parser.add_argument('--object', default='backpack', type=str)
    parser.add_argument('--flow_width', default=512, type=int)
    parser.add_argument('--num_blocks_per_layers', default=2, type=int)
    parser.add_argument('--layers', default=4, type=int)
    parser.add_argument('--c_dim', default=256, type=int)
    parser.add_argument('--dropout_probability', default=0., type=float)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--save_dir', default='./outputs/cflow_pseudo_kps2d')

    args = parser.parse_args()

    train(args)
