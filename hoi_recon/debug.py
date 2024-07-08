import os
import sys
import numpy as np
import random
import trimesh
import argparse
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F

from smplx import SMPLHLayer, SMPLXLayer
from sklearn.decomposition import PCA

from hoi_recon.models.incrementalPCA import IncrementalPCA
from hoi_recon.datasets.utils import load_pickle, save_pickle
from hoi_recon.datasets.behave_metadata import BEHAVEMetaData
from hoi_recon.datasets.behave_extend_metadata import BEHAVEExtendMetaData


class HOOffsetDataList:

    def __init__(self, annotation_file, metadata, object_name):

        self.metadata = metadata
        self.smpl = SMPLHLayer(model_path='data/models/smplh', gender='male', ext='pkl')
        self.annotations = []
        for item in annotation_file:
            if item['img_id'].split('_')[2] == object_name:
                self.annotations.append({
                    'img_id': item['img_id'],
                    'smplh_betas_male': item['smplh_betas_male'],
                    'smplh_pose_rotmat': item['smplh_pose_rotmat'],
                    'object_rel_rotmat': item['object_rel_rotmat'],
                    'object_rel_trans': item['object_rel_trans'],
                })
        random.shuffle(self.annotations)
        # self.annotations = annotation_file[:3000]
        self.obj_mesh_templates = metadata.obj_mesh_templates


    def __len__(self, ):
        return len(self.annotations)


    def __getitem__(self, idx):
        annotation = self.annotations[idx]
        img_id = annotation['img_id']
        obj_name = img_id.split('_')[2]
        object_v_org = self.metadata.obj_mesh_templates[obj_name][0]
        object_v_org_padded = np.ones((2000, 3)) * 1e8
        object_v_org_padded[:object_v_org.shape[0]] = object_v_org

        person_beta = annotation['smplh_betas_male'].astype(np.float32)
        person_body_pose = annotation['smplh_pose_rotmat'].astype(np.float32)[1:22]
        person_global_orient = annotation['smplh_pose_rotmat'].astype(np.float32)[0]

        object_r = annotation['object_rel_rotmat']
        object_t = annotation['object_rel_trans']

        return person_beta, person_body_pose, person_global_orient, object_v_org_padded, object_r, object_t

        # person_global_orient = torch.eye(3, dtype=torch.float32).reshape(1, 1, 3, 3)
        # smpl_out = self.smpl(betas=person_beta, body_pose=person_body_pose, global_orient=person_global_orient, transl=0 * person_transl,)

        # smpl_v = smpl_out.vertices.detach().reshape(-1, 3).numpy()
        # J_0 = smpl_out.joints.detach()[:, 0].reshape(1, 3).numpy()
        # smpl_v = smpl_v - J_0

        # object_r = annotation['object_rel_rotmat']
        # object_t = annotation['object_rel_trans']
        # object_v = np.matmul(object_v_org_padded, object_r.T) + object_t.reshape((1, 3))

        # # return smpl_v, object_v, object_v_org_padded, object_r, object_t

        # distance = ((smpl_v.reshape(-1, 1, 3) - object_v.reshape(1, -1, 3)) ** 2).sum(axis=-1)
        # object_v_indices = np.argmin(distance, axis=1)

        # offset = object_v.reshape(-1, 3)[object_v_indices] - smpl_v.reshape(-1, 3)
        # object_anchors = object_v_org[object_v_indices]

        # ho_offset = np.concatenate([offset, object_anchors], axis=1)
        # return ho_offset.reshape(-1)



def extract_pca_models():
    device = torch.device('cuda')
    annotation_file = '/public/home/huochf/projects/3D_HOI/StackFLOW/data/datasets/behave_extend_test_list_filtered.pkl'
    annotation_file = load_pickle(annotation_file)

    dataset_root = '/storage/data/huochf/BEHAVE'
    metadata = BEHAVEExtendMetaData(dataset_root)
    object_name = 'backpack'
    dataset_train = HOOffsetDataList(annotation_file, metadata, object_name)
    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=32, num_workers=2, shuffle=True)
    annotation_file = '/public/home/huochf/projects/3D_HOI/StackFLOW/data/datasets/behave_extend_test_list_filtered.pkl'
    annotation_file = load_pickle(annotation_file)
    dataset_test = HOOffsetDataList(annotation_file, metadata, object_name)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=32, num_workers=2, shuffle=True)

    smpl = SMPLHLayer(model_path='data/models/smplh', gender='male', ext='pkl').to(device)

    object_mesh = os.path.join(dataset_root, 'objects', object_name, '{}_f1000.ply'.format(object_name))
    if not os.path.exists(object_mesh):
        object_mesh = os.path.join(dataset_root, 'objects', object_name, '{}_f2000.ply'.format(object_name))
    if not os.path.exists(object_mesh):
        object_mesh = os.path.join(dataset_root, 'objects', object_name, '{}_f2500.ply'.format(object_name))
    if not os.path.exists(object_mesh):
        object_mesh = os.path.join(dataset_root, 'objects', object_name, '{}_closed_f1000.ply'.format(object_name))
    assert os.path.exists(object_mesh)

    object_mesh = trimesh.load(object_mesh, process=False)
    object_v_center = object_mesh.vertices.mean(0)
    object_v = object_mesh.vertices # trimesh.sample.sample_surface(object_mesh, 1000)[0]
    object_v_org = object_v - object_v_center
    object_v_org = torch.from_numpy(object_v_org).reshape(1, -1, 3).float().to(device)

    smpl_anchor_indices = sample_smpl_anchors(dataset_root, 32)

    offset_dim = 32 * 64 * 3

    incrementalPCA = IncrementalPCA(dim=offset_dim, feat_dim=128, forget_factor=0.99).to(device)

    object_anchors_weights = nn.Parameter(torch.randn(64, object_v.shape[0]).float().to(device))
    smpl_anchors_weights = nn.Parameter(torch.randn(32, 6890).float().to(device))
    optimizer = torch.optim.Adam([smpl_anchors_weights, object_anchors_weights], lr=1e-3, ) # betas=(0.9, 0.99))

    for epoch in range(1000):
        for idx, item in enumerate(dataloader_train):
            beta, body_pose, _, _, obj_rel_r, obj_rel_t = item

            batch_size = beta.shape[0]
            beta = beta.float().to(device)
            body_pose = body_pose.float().to(device)
            obj_rel_r = obj_rel_r.float().to(device)
            obj_rel_t = obj_rel_t.float().to(device)

            smpl_out = smpl(betas=beta, body_pose=body_pose)
            smpl_v = smpl_out.vertices
            smpl_J = smpl_out.joints
            smpl_v = smpl_v - smpl_J[:, :1]

            # smpl_anchors = smpl_v[:, smpl_anchor_indices['smpl']]

            object_v = object_v_org @ obj_rel_r.transpose(2, 1) + obj_rel_t.reshape(-1, 1, 3)
            weights = torch.softmax(object_anchors_weights, dim=1)
            object_anchors = weights.unsqueeze(0) @ object_v

            weights = torch.softmax(smpl_anchors_weights, dim=1)
            smpl_anchors = weights.unsqueeze(0) @ smpl_v

            ho_offsets = smpl_anchors.unsqueeze(1) - object_anchors.unsqueeze(2)
            ho_offsets = ho_offsets.reshape(batch_size, -1)


            # ho_dists = ((smpl_anchors.unsqueeze(2) - object_v.unsqueeze(1)) ** 2).sum(-1)
            # _, indices = ho_dists.min(dim=-1) # [b, n_smpl]
            # object_anchors = torch.stack([v[idx] for v, idx in zip(object_v, indices)])
            # object_anchors_org = object_v_org[0][indices]

            # ho_offsets = object_anchors - smpl_anchors
            # ho_offsets = torch.cat([ho_offsets, object_anchors_org], dim=-1).reshape(batch_size, -1)

            incrementalPCA.forward(ho_offsets)
            # print(incrementalPCA.sigma.max())
            loss_sigma = incrementalPCA.sigma.sum()

            if not torch.isnan(loss_sigma):
                optimizer.zero_grad()
                loss_sigma.backward()
                optimizer.step()

            lattent_codes = incrementalPCA.transform(ho_offsets, normalized=False)
            # print(torch.norm(lattent_codes, dim=1))
            offsets_recon = incrementalPCA.inverse(lattent_codes, normalized=False)
            loss = F.l1_loss(ho_offsets, offsets_recon)

            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()

            if idx % 10 == 0:
                print('Epoch: {}, Steps: {}, loss_recon: {:.6f}, loss_sigma: {:.4f}'.format(epoch, idx, loss.item(), loss_sigma.item()))

        if epoch % 1 == 0:
            torch.save({
                'epoch': epoch,
                'incrementalPCA': incrementalPCA.state_dict(),
            }, './incremental_pca.pth')
            weights = {'smpl_anchors_weights': smpl_anchors_weights.detach().cpu().numpy(), 'object_anchors_weights': object_anchors_weights.detach().cpu().numpy()}
            save_pickle(weights, './smpl_object_weights.pkl')



def sample_smpl_anchors(root_dir, smpl_anchor_num):

    dataset_metadata = BEHAVEExtendMetaData(root_dir)
    smpl_pkl = load_pickle('data/models/smplh/SMPLH_MALE.pkl')
    radius = 0.02

    weights = smpl_pkl['weights']
    parts_indices = np.argmax(weights[:, :22], axis=1)

    smpl_anchor_indices = []
    for i in range(22):
        part_anchor_indices = np.where(parts_indices == i)[0]
        part_anchors = part_anchor_indices[np.random.choice(len(part_anchor_indices), smpl_anchor_num)]
        smpl_anchor_indices.extend(part_anchors.tolist())

    anchor_indices = {'smpl': smpl_anchor_indices}
    return anchor_indices


class OffsetEncoder(nn.Module):

    def __init__(self, in_dim, hidden_dim, out_dim, num_layers):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, hidden_dim)
        hidden_proj = []
        for i in range(num_layers):
            hidden_proj.append(nn.Linear(hidden_dim, hidden_dim))
        self.hidden_proj = nn.ModuleList(hidden_proj)
        self.out_proj = nn.Linear(hidden_dim, out_dim)


    def forward(self, x):
        h = F.relu(self.in_proj(x))
        for layer in self.hidden_proj:
            h = F.relu(layer(h))
        return self.out_proj(h)


def learn_relation_encoding():
    device = torch.device('cuda')
    annotation_file = '/public/home/huochf/projects/3D_HOI/StackFLOW/data/datasets/behave_extend_test_list.pkl'
    annotation_file = load_pickle(annotation_file)

    dataset_root = '/storage/data/huochf/BEHAVE'
    metadata = BEHAVEExtendMetaData(dataset_root)
    object_name = 'backpack'
    dataset = HOOffsetDataList(annotation_file, metadata, object_name)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, num_workers=4, shuffle=True)

    smpl = SMPLHLayer(model_path='data/models/smplh', gender='male', ext='pkl').to(device)

    object_mesh = os.path.join(dataset_root, 'objects', object_name, '{}_f1000.ply'.format(object_name))
    if not os.path.exists(object_mesh):
        object_mesh = os.path.join(dataset_root, 'objects', object_name, '{}_f2000.ply'.format(object_name))
    if not os.path.exists(object_mesh):
        object_mesh = os.path.join(dataset_root, 'objects', object_name, '{}_f2500.ply'.format(object_name))
    if not os.path.exists(object_mesh):
        object_mesh = os.path.join(dataset_root, 'objects', object_name, '{}_closed_f1000.ply'.format(object_name))
    assert os.path.exists(object_mesh)

    object_mesh = trimesh.load(object_mesh, process=False)
    object_v_center = object_mesh.vertices.mean(0)
    object_v = trimesh.sample.sample_surface(object_mesh, 3000)[0]
    object_v_org = object_v - object_v_center
    object_v_org = torch.from_numpy(object_v_org).reshape(1, -1, 3).float().to(device)

    smpl_anchor_indices = sample_smpl_anchors(dataset_root, 128)

    offset_dim = 128 * 22 * 3 * 2
    encoder = OffsetEncoder(offset_dim, 512, 512, 2).to(device)
    decoder = OffsetEncoder(512, 512, offset_dim, 2).to(device)

    optimizer = torch.optim.Adam(list(decoder.parameters()) + list(encoder.parameters()), lr=1e-3)

    state_dict = torch.load('./relation_reconder.pth')
    encoder.load_state_dict(state_dict['encoder'])
    decoder.load_state_dict(state_dict['decoder'])
    optimizer.load_state_dict(state_dict['optimizer'])

    for param_group in optimizer.param_groups:
        param_group['lr'] *= 0.1
    for epoch in range(1000):
        for idx, item in enumerate(dataloader):
            beta, body_pose, _, _, obj_rel_r, obj_rel_t = item

            batch_size = beta.shape[0]
            beta = beta.float().to(device)
            body_pose = body_pose.float().to(device)
            obj_rel_r = obj_rel_r.float().to(device)
            obj_rel_t = obj_rel_t.float().to(device)

            smpl_out = smpl(betas=beta, body_pose=body_pose)
            smpl_v = smpl_out.vertices
            smpl_J = smpl_out.joints
            smpl_v = smpl_v - smpl_J[:, :1]

            object_v = object_v_org @ obj_rel_r.transpose(2, 1) + obj_rel_t.reshape(-1, 1, 3)

            smpl_anchors = smpl_v[:, smpl_anchor_indices['smpl']]

            ho_dists = ((smpl_anchors.unsqueeze(2) - object_v.unsqueeze(1)) ** 2).sum(-1)
            _, indices = ho_dists.min(dim=-1) # [b, n_smpl]
            object_anchors = torch.stack([v[idx] for v, idx in zip(object_v, indices)])
            object_anchors_org = object_v_org[0][indices]

            ho_offsets = object_anchors - smpl_anchors
            ho_offsets = torch.cat([ho_offsets, object_anchors_org], dim=-1).reshape(batch_size, -1)

            lattent_codes = encoder(ho_offsets)
            offsets_recon = decoder(lattent_codes)

            loss = F.l1_loss(ho_offsets, offsets_recon)

            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()
            if idx % 10 == 0:
                print('Epoch: {}, Steps: {}, Loss: {:.4f}'.format(epoch, idx, loss.item()))

        if epoch % 1 == 0:
            torch.save({
                'encoder': epoch,
                'encoder': encoder.state_dict(),
                'decoder': decoder.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, './relation_reconder.pth')



def learning_object_shape():
    device = torch.device('cuda')
    annotation_file = '/public/home/huochf/projects/3D_HOI/StackFLOW/data/datasets/behave_train_list.pkl'
    annotation_file = load_pickle(annotation_file)

    metadata = BEHAVEMetaData('/public/home/huochf/datasets/BEHAVE/')
    object_name = 'backpack'
    dataset = HOOffsetDataList(annotation_file, metadata, object_name)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=128, num_workers=4, shuffle=True)

    weights = load_pickle('./weights.pkl')

    object_shape_org = nn.Parameter(torch.randn(512, 3).float().to(device))
    optimizer = torch.optim.Adam([object_shape_org, ], lr=1e-1)

    for epoch in range(1000):
        for idx, item in enumerate(dataloader):
            smpl_v, object_v, object_v_org, object_r, object_t = item

            smpl_v = smpl_v.float().to(device)
            object_v = object_v.float().to(device)
            object_v_org = object_v_org.float().to(device) # [b, -1, 3]
            object_r = object_r.float().to(device) # [b, 3, 3]
            object_t = object_t.float().to(device) # [b, 3, 1]

            batch_size = smpl_v.shape[0]

            offsets = object_v.unsqueeze(1) - smpl_v.unsqueeze(2)
            distance = (offsets ** 2).sum(-1) # [b, m, n]
            object_v_indices = torch.argmin(distance, dim=2)
            object_anchors = torch.stack([vertices[indices] for vertices, indices in zip(object_v_org, object_v_indices)], dim=0)
            object_anchors_trans = object_anchors @ object_r.transpose(2, 1) + object_t.reshape(batch_size, 1, 3)
            offsets = object_anchors_trans - smpl_v
            # offsets_gt = torch.cat([offsets, object_anchors], dim=2).reshape(batch_size, -1)
            offsets_gt = offsets.reshape(batch_size, -1)

            object_shape = object_shape_org.unsqueeze(0) @ object_r.transpose(2, 1) + object_t.reshape(batch_size, 1, 3)
            offset_pred = object_shape.reshape(batch_size, 1, 512, 3) - smpl_v.unsqueeze(2)
            distance = (offset_pred ** 2).sum(-1) # [b, m, n]
            object_v_indices = torch.argmin(distance, dim=2)
            object_anchors = object_shape_org[object_v_indices]
            object_anchors_trans = object_anchors @ object_r.transpose(2, 1) + object_t.reshape(batch_size, 1, 3)
            offsets = object_anchors_trans - smpl_v
            # offsets_pred = torch.cat([offsets, object_anchors], dim=2).reshape(batch_size, -1)
            offsets_pred = offsets.reshape(batch_size, -1)

            loss = (offsets_pred - offsets_gt).abs().mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if idx % 10 == 0:
                print('Epoch: {}, Steps: {}, Loss: {:.4f}'.format(epoch, idx, loss.item()))

        if epoch % 10 == 0:
            weights = {'object_shape': object_shape_org.detach().cpu().numpy()}
            save_pickle(weights, './object_shape.pkl')


def learn_pca():
    device = torch.device('cuda')

    smpl = SMPLHLayer(model_path='data/models/smplh', gender='male', ext='pkl').to(device)

    annotation_file = '/public/home/huochf/projects/3D_HOI/StackFLOW/data/datasets/behave_train_list.pkl'
    annotation_file = load_pickle(annotation_file)

    metadata = BEHAVEMetaData('/public/home/huochf/datasets/BEHAVE/')
    object_name = 'backpack'
    dataset_train = HOOffsetDataList(annotation_file, metadata, object_name)
    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=32, num_workers=4, shuffle=True)

    annotation_file = '/public/home/huochf/projects/3D_HOI/StackFLOW/data/datasets/behave_test_list.pkl'
    annotation_file = load_pickle(annotation_file)
    dataset_test = HOOffsetDataList(annotation_file, metadata, object_name)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=4, num_workers=4, shuffle=True)

    A = nn.Parameter(torch.randn(6890 * 3, 32).float().to(device))
    B = nn.Parameter(torch.randn(32, 6890 * 3).float().to(device))
    optimizer = torch.optim.Adam([A, B], lr=1e-2, betas=(0.9, 0.999))
    all_offsets = []
    for epoch in range(1000):
        if epoch in [10, 20, 30, 40]:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1
        for idx, item in enumerate(dataloader_train):
            person_beta, person_body_pose, person_global_orient, object_v_org_padded, object_r, object_t = item
            person_beta = person_beta.float().to(device)
            b = person_beta.shape[0]
            person_beta = person_beta.reshape(b, 10)
            person_body_pose = person_body_pose.float().to(device).reshape(b, 21, 3, 3)
            person_global_orient = person_global_orient.float().to(device).reshape(b, 1, 3, 3)
            object_v_org_padded = object_v_org_padded.float().to(device)
            object_r = object_r.float().to(device)
            object_t = object_t.float().to(device)

            smpl_out = smpl(betas=person_beta, body_pose=person_body_pose, global_orient=person_global_orient)
            smpl_v = smpl_out.vertices
            smpl_J = smpl_out.joints
            smpl_v = smpl_v - smpl_J[:, :1]

            object_v = object_v_org_padded @ object_r.transpose(2, 1) + object_t.reshape(b, 1, 3)

            distance = ((smpl_v.reshape(b, -1, 1, 3) - object_v.reshape(b, 1, -1, 3)) ** 2).sum(dim=-1)
            object_v_indices = torch.argmin(distance, dim=2)

            object_anchors = object_v_org_padded[0][object_v_indices]

            offset = object_anchors - smpl_v.reshape(b, -1, 3)

            offsets = offset.reshape(b, -1)
            all_offsets.append(offsets.detach().cpu().numpy())
        break

        #     lattent_codes = offsets @ A
        #     offsets_recon = lattent_codes @ B
        #     loss = F.l1_loss(offsets_recon, offsets)

        #     optimizer.zero_grad()
        #     loss.backward()
        #     optimizer.step()
        #     if idx % 10 == 0:
        #         print('Epoch: {}, Iter: {}, Loss: {:.4f}'.format(epoch, idx, loss.item()))

        # for idx, item in enumerate(dataloader_test):
        #     if idx > 10:
        #         break
        #     person_beta, person_body_pose, person_global_orient, object_v_org_padded, object_r, object_t = item
        #     person_beta = person_beta.float().to(device)
        #     b = person_beta.shape[0]
        #     person_beta = person_beta.reshape(b, 10)
        #     person_body_pose = person_body_pose.float().to(device).reshape(b, 21, 3, 3)
        #     person_global_orient = person_global_orient.float().to(device).reshape(b, 1, 3, 3)
        #     object_v_org_padded = object_v_org_padded.float().to(device)
        #     object_r = object_r.float().to(device)
        #     object_t = object_t.float().to(device)

        #     smpl_out = smpl(betas=person_beta, body_pose=person_body_pose, global_orient=person_global_orient)
        #     smpl_v = smpl_out.vertices
        #     smpl_J = smpl_out.joints
        #     smpl_v = smpl_v - smpl_J[:, :1]

        #     object_v = object_v_org_padded @ object_r.transpose(2, 1) + object_t.reshape(b, 1, 3)

        #     distance = ((smpl_v.reshape(b, -1, 1, 3) - object_v.reshape(b, 1, -1, 3)) ** 2).sum(axis=-1)
        #     object_v_indices = torch.argmin(distance, dim=2)

        #     object_anchors = object_v_org_padded[0][object_v_indices]

        #     offset = object_anchors - smpl_v.reshape(b, -1, 3)

        #     offsets = offset.reshape(b, -1)

        #     lattent_codes = offsets @ A
        #     offsets_recon = lattent_codes @ B
        #     loss = F.l1_loss(offsets_recon, offsets)

        #     if idx % 10 == 0:
        #         print('[EVAL] Epoch: {}, Iter: {}, Loss: {:.4f}'.format(epoch, idx, loss.item()))

    pca = PCA(n_components=64)
    all_offsets = np.concatenate(all_offsets, axis=0)
    pca.fit_transform(all_offsets)
    print(pca.explained_variance_ratio_)
    print(pca.explained_variance_)

    mean = pca.mean_.reshape(1, -1)
    components = pca.components_
    print(components.shape, mean.shape)
    latent_code = np.matmul(all_offsets[:100] - mean, components.T)
    recon_offset = np.matmul(latent_code, components) + mean
    recon_error = np.abs(all_offsets[:100] - recon_offset).mean()
    print('reconstruction error: {}'.format(recon_error))


def laplace_coord(inputs, neigbours):
    neighbor_mask = neigbours < 0
    neighbor_num = (neigbours >= 0).sum(1)
    neigbours = inputs[:, neigbours] # [b, n_kps, n_neighbors, 3]
    neigbours[:, neighbor_mask] = 0
    neighbor_sum = neigbours.sum(2) # [b, n_kps, 3]
    loss_neighbor_lap = inputs - neighbor_sum / neighbor_num[None, :, None]

    return loss_neighbor_lap


def learn_deformation():
    from hoi_recon.models.gconv import MeshDeformation
    from hoi_recon.utils.sphere import Sphere
    from hoi_recon.utils.chamfer_wrapper import ChamferDist

    device = torch.device('cuda')

    sphere = Sphere('data/models/spheres/sphere_4k.ply')
    model = MeshDeformation(sphere, features_dim=256, hidden_dim=256, stages=3, layers_per_stages=4).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.999))

    smpl = SMPLHLayer(model_path='data/models/smplh', gender='male', ext='pkl')
    chamfer_loss = ChamferDist().to(device)
    l2_loss = nn.MSELoss(reduction='mean')

    smpl_v = torch.tensor(smpl.v_template).float().unsqueeze(0).to(device)

    model.train()

    for i in range(99999999):

        features = torch.zeros((1, 256)).float().to(device)
        stages = model(features)

        coord_init = sphere.coord.to(device).unsqueeze(0)
        stages.insert(0, coord_init)

        loss_chamfer_all_stages = []
        loss_edge_regular_all_stages = []
        loss_laplace_regular_all_stages = []
        for idx in range(1, len(stages)):
            pts_recon = stages[idx]
            pts_prev = stages[idx - 1]
    
            dist1, dist2, idx1, idx2 = chamfer_loss(pts_recon, smpl_v)

            loss_chamfer = dist1.mean() + dist2.mean()
            loss_chamfer_all_stages.append(loss_chamfer)

            loss_edge_regular = l2_loss(pts_recon[:, sphere.edges[0]], pts_recon[:, sphere.edges[1]])
            loss_edge_regular_all_stages.append(loss_edge_regular)

            lap = laplace_coord(pts_recon, sphere.neigbours.to(device))
            lap_prev = laplace_coord(pts_prev, sphere.neigbours.to(device))
            laplace_loss = l2_loss(lap, lap_prev)
            loss_laplace_regular_all_stages.append(laplace_loss)

        loss = 0
        loss_weights = [0.5, 0.7, 1.0]
        for idx in range(len(loss_chamfer_all_stages)):
            loss = loss + loss_weights[idx] * loss_chamfer_all_stages[idx]
            loss = loss + loss_weights[idx] * loss_edge_regular_all_stages[idx]
            loss = loss + 100 * loss_weights[idx] * loss_laplace_regular_all_stages[idx]

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()

        if i % 10 == 0:
            log_str = '[{}] Loss: {:.4f}, Loss_chamfer: {:.4f}, Loss_edge: {:.4f}, Loss_laplace: {:.4f}'.format(
                i, loss.item(), loss_chamfer_all_stages[-1].item(), 
                loss_edge_regular_all_stages[-1].item(), 
                loss_laplace_regular_all_stages[-1].item())
            print(log_str)
            sys.stdout.flush()

        if i % 100 == 0:
            data = pts_recon.detach().cpu().numpy()
            save_pickle({'pts_stage0': stages[0].detach().cpu().numpy(),
                'pts_stage1': stages[1].detach().cpu().numpy(),
                'pts_stage2': stages[2].detach().cpu().numpy(),
                'pts_stage3': stages[3].detach().cpu().numpy(),
                }, './smpl_v_recon.pkl')


def optimize_with_pca():
    device = torch.device('cuda')

    A = nn.Parameter(torch.randn(256, 32).float().to(device))
    optimizer = torch.optim.Adam([A, ], lr=1e-4, betas=(0.9, 0.99))
    incrementalPCA = IncrementalPCA(dim=32, feat_dim=8, forget_factor=0.99).to(device)

    for epoch in range(1000000000):
        incrementalPCA.forward(A)
        # print(incrementalPCA.sigma.max())
        # loss_sigma = incrementalPCA.sigma[16:].mean()

        loss_sigma = incrementalPCA.sigma.sum()

        if not torch.isnan(loss_sigma):
            optimizer.zero_grad()
            loss_sigma.backward()
            optimizer.step()

        lattent_codes = incrementalPCA.transform(A, normalized=False)
        # print(torch.norm(lattent_codes, dim=1))
        A_recon = incrementalPCA.inverse(lattent_codes, normalized=False)
        loss = F.l1_loss(A, A_recon)

        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()

        if epoch % 10 == 0:
            print('Epoch: {}, loss_recon: {:.6f}, loss_sigma: {:.6f}'.format(epoch, loss.item(), loss_sigma.item()))


def test_flow():

    from hoi_recon.models.condition_flow import ConditionFlow
    device = torch.device('cuda')
    flow = ConditionFlow(dim=64, 
                         hidden_dim=256, 
                         c_dim=128, 
                         num_blocks_per_layer=2,
                         num_layers=4,
                         dropout_probability=0.0).to(device)
    x = torch.randn((4, 64)).float().to(device)
    condition = torch.randn((4, 128)).float().to(device)

    z, sum_logdets = flow.forward(x, condition)
    x_inverse, logdet_inverse = flow.inverse(z, condition)

    print(x, x_inverse)


if __name__ == '__main__':
    # extract_pca_models()
    # learn_relation_encoding()
    # learning_object_shape()
    # learn_pca()
    # learn_deformation()
    # optimize_with_pca()
    test_flow()
