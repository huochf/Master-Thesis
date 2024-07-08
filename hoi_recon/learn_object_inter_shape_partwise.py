import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.loss import mesh_laplacian_smoothing
from pytorch3d.loss.chamfer import chamfer_distance
from pytorch3d.loss.point_mesh_distance import point_face_distance
from pytorch3d.transforms import matrix_to_rotation_6d, matrix_to_axis_angle, axis_angle_to_matrix, rotation_6d_to_matrix
from smplx import SMPLX

from hoi_recon.models.incrementalPCA import IncrementalPCA
from hoi_recon.datasets.utils import load_pickle, save_pickle
from hoi_recon.datasets.chairs_hoi_template_dataset import CHAIRSDataset
from hoi_recon.models.conv3d import Voxel3DEncoder
from hoi_recon.models.gconv import MeshDeformer
from hoi_recon.utils.sphere import Sphere
# from hoi_recon.utils.chamfer_wrapper import ChamferDist


def to_device(batch, device):
    results = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    return results


def point_to_mesh_distance(meshes, pcls):   
    N = len(meshes)
    min_triangle_area = 5e-3

    points = pcls.points_packed() # (P, 3)
    points_first_idx = pcls.cloud_to_packed_first_idx()
    max_points = pcls.num_points_per_cloud().max().item()

    verts_packed = meshes.verts_packed()
    faces_packed = meshes.faces_packed()
    tris = verts_packed[faces_packed]
    tris_first_idx = meshes.mesh_to_faces_packed_first_idx()

    point_to_face = point_face_distance(points, points_first_idx, tris, tris_first_idx, max_points, min_triangle_area) # (P, )

    return point_to_face.reshape(N, -1)


def f_ho_correlation(distances, alpha=1., beta=1.):
    # distance: [b, n]
    distances_ = alpha * (distances ** beta)
    weights = (distances_ + 1) * torch.exp(- distances_)
    return weights


def train():

    device = torch.device('cuda')
    batch_size = 8
    part_num = 7
    smplx = SMPLX('data/models/smplx/', gender='male', use_pca=False, batch_size=batch_size).to(device)
    smplx_f = torch.tensor(np.array(smplx.faces).astype(np.int64)).to(device)

    output_dir = './outputs/chairs/'
    os.makedirs(output_dir, exist_ok=True)

    voxel_encoder = Voxel3DEncoder(feat_dim=256, num_parts=part_num, res=64).to(device)
    sphere = Sphere('data/models/spheres/sphere_1k.ply', radius=0.1)
    part_mesh_deformers = []
    for _ in range(part_num):
        mesh_deformer = MeshDeformer(sphere, features_dim=256, hidden_dim=256, stages=3, layers_per_stages=4).to(device)
        part_mesh_deformers.append(mesh_deformer)
    # chamfer_loss = ChamferDist().to(device)
    l2_loss = nn.MSELoss(reduction='none')

    parameters = list(voxel_encoder.parameters())
    for mesh_deformer in part_mesh_deformers:
        parameters.extend(list(mesh_deformer.parameters()))
    optimizer = torch.optim.Adam(parameters, lr=1e-4, betas=(0.9, 0.999))

    dataset_train = CHAIRSDataset('/storage/data/huochf/CHAIRS', res=64, split='train')
    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, num_workers=8, shuffle=True, drop_last=True)
    dataset_test = CHAIRSDataset('/storage/data/huochf/CHAIRS', res=64, split='test')
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, num_workers=8, shuffle=True, drop_last=True)

    anchor_indices = load_pickle('data/datasets/chairs_anchor_indices_n32_128.pkl')
    smpl_anchor_indices = anchor_indices['smpl']
    object_anchor_indices = anchor_indices['object']['sphere_1k']

    offset_dim = 22 * 32 * 128 * 3
    incrementalPCA_all_parts = []
    for _ in range(part_num):
        incrementalPCA = IncrementalPCA(dim=offset_dim, feat_dim=128, forget_factor=0.99).to(device)
        incrementalPCA_all_parts.append(incrementalPCA)

    if False and os.path.exists(os.path.join(output_dir, './chairs_object_inter_models.pth')):
        state_dict = torch.load(os.path.join(output_dir, './chairs_object_inter_models.pth'))
        voxel_encoder.load_state_dict(state_dict['voxel_encoder'])
        for part_idx, mesh_deformer in enumerate(part_mesh_deformers):
            mesh_deformer.load_state_dict(state_dict['mesh_deformer_{}'.format(part_idx)])
        for part_idx, incrementalPCA in enumerate(incrementalPCA_all_parts):
            incrementalPCA.load_state_dict(state_dict['incrementalPCA_{}'.format(part_idx)])
        optimizer.load_state_dict(state_dict['optimizer'])
        begin_epoch = state_dict['epoch']
    else:
        begin_epoch = 0

    epoch = 9999
    for epoch in range(begin_epoch, epoch):

        def iteration_one_epoch(dataloader, is_train=True):
            for idx, batch in enumerate(dataloader):
                batch = to_device(batch, device)
                batch_size = batch['human_betas'].shape[0]

                smplx_out = smplx(betas=batch['human_betas'], body_pose=batch['human_pose'])
                smplx_v = smplx_out.vertices.detach()
                smplx_J = smplx_out.joints.detach()
                smplx_v = smplx_v - smplx_J[:, :1]

                object_v_org = batch['object_v'] # [b, n_parts, n, 3]
                object_v_len = batch['object_v_len'].long() # [b, n_parts]
                object_T = batch['object_T'] # [b, n_parts, 3]
                object_R = batch['object_R'] # [b, n_parts, 3, 3]
                b, n_parts, _, _ = object_v_org.shape

                object_v_org = object_v_org.reshape(b * n_parts, -1, 3)
                object_v_len = object_v_len.reshape(b * n_parts, )
                object_R = object_R.reshape(b * n_parts, 3, 3)
                object_T = object_T.reshape(b * n_parts, 1, 3)

                object_v = object_v_org @ object_R.transpose(2, 1) + object_T
                smplx_v = smplx_v.unsqueeze(1).repeat(1, n_parts, 1, 1).reshape(b * n_parts, -1, 3)
                smpl_mesh = Meshes(verts=smplx_v, faces=smplx_f.unsqueeze(0).repeat(b * n_parts, 1, 1))
                smpl_pcls = Pointclouds(points=smplx_v)

                object_pcls = Pointclouds(points=object_v)

                object_voxel = batch['object_voxel'].unsqueeze(1)
                b = object_voxel.shape[0]
                object_feats = voxel_encoder(object_voxel) # [b, num_parts, dim]

                coords_all_parts = []
                for part_id in range(n_parts):
                    parts_feats = object_feats[:, part_id]
                    parts_coords = part_mesh_deformers[part_id](parts_feats)
                    parts_coords = torch.stack(parts_coords, dim=1)
                    coords_all_parts.append(parts_coords)
                coords_all_parts = torch.stack(coords_all_parts, dim=1) # [b, n_parts, n_stages, n, 3]
                _, n_parts, n_stage, _, _ = coords_all_parts.shape

                loss_mesh_laplace_all_stages = []
                loss_ho_offset_all_stages = []
                loss_edge_regular_all_stages = []
                loss_offset_recon_all_stages = []
                loss_sigma_all_stages = []
                for stage_idx in range(n_stage):
                    coords_recon = coords_all_parts[:, :, stage_idx].reshape(b * n_parts, -1, 3)

                    object_recon_meshes = Meshes(verts=coords_recon, faces=sphere.faces.unsqueeze(0).repeat(b * n_parts, 1, 1).to(device))

                    loss_edge_regular = l2_loss(coords_recon[:, sphere.edges[0]], coords_recon[:, sphere.edges[1]]).mean(-1)
                    loss_edge_regular_top_k, _ = torch.topk(loss_edge_regular, k=100, dim=1, largest=True)
                    loss_edge_regular_all_stages.append(loss_edge_regular_top_k.mean())

                    mesh_laplace = mesh_laplacian_smoothing(object_recon_meshes)
                    loss_mesh_laplace_all_stages.append(mesh_laplace)

                    object_coords = coords_recon @ object_R.transpose(2, 1) + object_T
                    for _i, v_num in enumerate(object_v_len):
                        if v_num == 0:
                            object_coords[_i] = 0

                    coords_recon_pcls = Pointclouds(points=object_coords)
                    distances, _ = chamfer_distance(smpl_pcls, object_pcls, batch_reduction=None, point_reduction=None, single_directional=True)
                    distances_recon, _ = chamfer_distance(smpl_pcls, coords_recon_pcls, batch_reduction=None, point_reduction=None, single_directional=True)

                    dist_w = f_ho_correlation(distances, alpha=1000, beta=2)
                    dist_recon_w = f_ho_correlation(distances_recon, alpha=1000, beta=2)
                    weights = dist_w + dist_recon_w - dist_w * dist_recon_w
                    loss_ho_chamfer = F.l1_loss(distances, distances_recon, reduction='none') # [b, smpl_n]
                    loss_ho_chamfer = (loss_ho_chamfer * weights).mean()
                    loss_ho_offset_all_stages.append(loss_ho_chamfer)

                    object_anchors = object_coords[:, object_anchor_indices]
                    smpl_anchors = smplx_v[:, smpl_anchor_indices].reshape(b * n_parts, -1, 3)
                    ho_offsets = object_anchors.unsqueeze(1) - smpl_anchors.unsqueeze(2)
                    ho_offsets = ho_offsets.reshape(b, n_parts, -1)

                    loss_sigma_all_parts = []
                    loss_offset_recon_all_parts = []
                    for part_id in range(n_parts):
                        _ho_offsets = ho_offsets[:, part_id]
                        incrementalPCA = incrementalPCA_all_parts[part_id]
                        try:
                            incrementalPCA.forward(_ho_offsets)
                        except:
                            print('exception raised during incremental PCA.')
                        loss_sigma = incrementalPCA.sigma.sum()
                        loss_sigma_all_parts.append(loss_sigma)

                        latent_codes = incrementalPCA.transform(ho_offsets, normalized=False)
                        offset_recon = incrementalPCA.inverse(latent_codes, normalized=False)
                        loss_offset_recon = F.l1_loss(ho_offsets, offset_recon)
                        loss_offset_recon_all_parts.append(loss_offset_recon)

                    loss_offset_recon_all_stages.append(torch.stack(loss_offset_recon_all_parts).mean())
                    loss_sigma_all_stages.append(torch.stack(loss_sigma_all_parts).mean())

                loss = 0
                loss_weights = [0.1, 0.5, 1.0]
                for stage_idx in range(len(loss_ho_offset_all_stages)):
                    loss = loss + loss_weights[stage_idx] * loss_ho_offset_all_stages[stage_idx]
                    loss = loss + loss_weights[stage_idx] * loss_edge_regular_all_stages[stage_idx]
                    loss = loss + loss_weights[stage_idx] * loss_mesh_laplace_all_stages[stage_idx]
                    loss = loss + loss_weights[stage_idx] * 0.0001 * loss_sigma_all_stages[stage_idx]

                if is_train:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                if idx % 10 == 0:
                    log_str = '[{} / {}] Loss: {:.4f}, Loss_ho_offset: {:.4f}, Loss_edge: {:.4f}, Loss_mesh_laplace: {:.4f}, Loss_sigma: {:.4f}, Loss_recon: {:.4f}'.format(
                        epoch, idx, loss.item(),
                        loss_ho_offset_all_stages[-1].item(), 
                        loss_edge_regular_all_stages[-1].item(), 
                        loss_mesh_laplace_all_stages[-1].item(),
                        loss_sigma_all_stages[-1].item(),
                        loss_offset_recon_all_stages[-1].item(),
                        )
                    if not is_train:
                        log_str = '[EVAL] ' + log_str
                    print(log_str)
                    sys.stdout.flush()

                if idx != 0 and idx % 1000 == 0:
                    weights = {'parts_coords': coords_all_parts.detach().cpu().numpy(), 
                               'sphere_faces': sphere.faces.numpy(),
                               'object_id': batch['object_id'].detach().cpu().numpy(),
                            }
                    if is_train:
                        save_pickle(weights, os.path.join(output_dir, './object_shape_train.pkl'))

                        save_dict = {
                            'epoch': epoch,
                        }
                        for part_idx, mesh_deformer in enumerate(part_mesh_deformers):
                            save_dict['mesh_deformer_{}'.format(part_idx)] = mesh_deformer.state_dict()
                        save_dict['voxel_encoder'] = voxel_encoder.state_dict()
                        for part_idx, incrementalPCA in enumerate(incrementalPCA_all_parts):
                            save_dict['incrementalPCA_{}'.format(part_idx)] = incrementalPCA.state_dict()
                        save_dict['optimizer'] = optimizer.state_dict()
                        torch.save(save_dict, os.path.join(output_dir, './chairs_object_inter_models_alpha_1000.pth'))
                    else:
                        save_pickle(weights, os.path.join(output_dir, './object_shape_test.pkl'))
                        return
        iteration_one_epoch(dataloader_train, is_train=True)
        iteration_one_epoch(dataloader_test, is_train=False)


if __name__ == '__main__':
    train()
