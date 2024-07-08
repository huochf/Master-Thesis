import torch
import torch.nn as nn
from smplx import SMPLHLayer, SMPLXLayer
import json
import pickle
from pytorch3d.transforms import matrix_to_axis_angle, matrix_to_rotation_6d, rotation_6d_to_matrix, axis_angle_to_matrix

from smplx import SMPL, SMPLH, SMPLLayer, SMPLHLayer
from hoi_recon.datasets.utils import load_J_regressor


class HOIInstanceSeq(nn.Module):

    def __init__(self, smpl, object_kps, object_v, smpl_betas=None, smpl_body_pose6d=None, obj_rel_trans=None, obj_rel_rotmat=None, hoi_trans=None, hoi_rot6d=None):
        super(HOIInstanceSeq, self).__init__()

        self.smpl = smpl

        if isinstance(smpl, SMPLH) or isinstance(smpl, SMPLHLayer):
            npose = 21
        elif isinstance(smpl, SMPL) or isinstance(smpl, SMPLLayer):
            npose = 23
        else:
            assert False

        self.register_buffer('object_v', object_v) # (1, n, 3)
        self.register_buffer('object_kps', object_kps) # (1, n, 3)

        batch_size = smpl_betas.shape[0]

        if smpl_betas is not None:
            self.smpl_betas = nn.Parameter(smpl_betas.reshape(batch_size, 10))
        else:
            self.smpl_betas = nn.Parameter(torch.zeros(batch_size, 10, dtype=torch.float32))

        if smpl_body_pose6d is not None:
            self.smpl_body_pose6d = nn.Parameter(smpl_body_pose6d.reshape(batch_size, npose, 6))
        else:
            self.smpl_body_pose6d = nn.Parameter(matrix_to_rotation_6d(torch.eye(3, dtype=torch.float32).reshape(1, 1, 3, 3).repeat(batch_size, npose, 1, 1)))

        if obj_rel_trans is not None:
            self.obj_rel_trans = nn.Parameter(obj_rel_trans.reshape(batch_size, 3))
        else:
            self.obj_rel_trans = nn.Parameter(torch.zeros(batch_size, 3, dtype=torch.float32))

        if obj_rel_rotmat is not None:
            self.obj_rel_rot6d = nn.Parameter(matrix_to_rotation_6d(obj_rel_rotmat.reshape(batch_size, 3, 3)))
        else:
            self.obj_rel_rot6d = nn.Parameter(matrix_to_rotation_6d(torch.eye(3, dtype=torch.float32).reshape(1, 3, 3).repeat(batch_size, 1, 1)))

        if hoi_trans is not None:
            self.hoi_trans = nn.Parameter(hoi_trans.reshape(batch_size, 3))
        else:
            self.hoi_trans = nn.Parameter(torch.zeros(batch_size, 3, dtype=torch.float32))

        if hoi_rot6d is not None:
            self.hoi_rot6d = nn.Parameter(hoi_rot6d.reshape(batch_size, 6))
        else:
            self.hoi_rot6d = nn.Parameter(matrix_to_rotation_6d(torch.eye(3, dtype=torch.float32).reshape(1, 3, 3).repeat(batch_size, 1, 1)))

        self.object_scale = nn.Parameter(torch.ones(1).reshape(1, 1, 1).repeat(batch_size, 1, 1).float())

        openpose_regressor = load_J_regressor('data/models/smpl/J_regressor_body25_smplh.txt')
        self.register_buffer('openpose_regressor', torch.tensor(openpose_regressor).float())


    def get_optimizer(self, fix_trans=False, fix_global_orient=False, fix_betas=True, fix_scale=True, lr=0.001):
        param_list = [self.smpl_body_pose6d, self.obj_rel_rot6d, self.obj_rel_trans]
        if not fix_trans:
            param_list.append(self.hoi_trans)
        if not fix_global_orient:
            param_list.append(self.hoi_rot6d)
        if not fix_betas:
            param_list.append(self.smpl_betas)
        if not fix_scale:
            param_list.append(self.object_scale)

        optimizer = torch.optim.Adam(param_list, lr=lr, betas=(0.9, 0.999))
        return optimizer


    def forward(self, batch_idx, batch_size):
        b = self.smpl_betas[batch_idx:batch_idx+batch_size].shape[0]
        smpl_body_rotmat = rotation_6d_to_matrix(self.smpl_body_pose6d[batch_idx:batch_idx+batch_size])
        smpl_out = self.smpl(betas=self.smpl_betas[batch_idx:batch_idx+batch_size], body_pose=smpl_body_rotmat)
        smpl_v = smpl_out.vertices
        smpl_J = smpl_out.joints # [:, :22]
        orig = smpl_J[:, 0:1]
        smpl_v_centered = smpl_v - orig
        smpl_J_centered = smpl_J - orig

        hoi_rotmat = rotation_6d_to_matrix(self.hoi_rot6d[batch_idx:batch_idx+batch_size])
        smpl_v = torch.matmul(smpl_v_centered, hoi_rotmat.permute(0, 2, 1)) + self.hoi_trans[batch_idx:batch_idx+batch_size].reshape(b, 1, 3)
        smpl_J = torch.matmul(smpl_J_centered, hoi_rotmat.permute(0, 2, 1)) + self.hoi_trans[batch_idx:batch_idx+batch_size].reshape(b, 1, 3)

        scale = self.object_scale[batch_idx:batch_idx+batch_size].reshape(-1, 1, 1)
        object_v_org = self.object_v * scale
        object_kps_org = self.object_kps * scale

        obj_rel_rotmat = rotation_6d_to_matrix(self.obj_rel_rot6d[batch_idx:batch_idx+batch_size])
        obj_rotmat = torch.matmul(hoi_rotmat, obj_rel_rotmat)
        obj_trans = torch.matmul(hoi_rotmat, self.obj_rel_trans[batch_idx:batch_idx+batch_size].reshape(b, 3, 1)).squeeze(-1) + self.hoi_trans[batch_idx:batch_idx+batch_size]
        object_v = torch.matmul(object_v_org, obj_rotmat.permute(0, 2, 1)) + obj_trans.reshape(b, 1, 3)
        object_v_centered = torch.matmul(object_v_org, obj_rel_rotmat.permute(0, 2, 1)) + self.obj_rel_trans[batch_idx:batch_idx+batch_size].reshape(b, 1, 3)
        object_kps_centered = torch.matmul(object_kps_org, obj_rel_rotmat.permute(0, 2, 1)) + self.obj_rel_trans[batch_idx:batch_idx+batch_size].reshape(b, 1, 3)
        object_kps_centered[object_kps_org == 0] = 0

        openpose_kps3d = self.openpose_regressor.unsqueeze(0) @ smpl_v

        results = {
            'smpl_betas': self.smpl_betas[batch_idx:batch_idx+batch_size],
            'smpl_body_pose6d': self.smpl_body_pose6d[batch_idx:batch_idx+batch_size],
            'smpl_body_rotmat': smpl_body_rotmat,
            'smpl_v_centered': smpl_v_centered,
            'smpl_J_centered': smpl_J_centered, # smpl_J_centered,
            'smpl_v': smpl_v,
            'smpl_J': smpl_v, # smpl_J,
            'openpose_kps3d': openpose_kps3d,
            'obj_rel_trans': self.obj_rel_trans[batch_idx:batch_idx+batch_size],
            'obj_rel_rotmat': obj_rel_rotmat,
            'obj_rel_rot6d': self.obj_rel_rot6d[batch_idx:batch_idx+batch_size],
            'obj_rotmat': obj_rotmat,
            'obj_trans': obj_trans,
            'object_v': object_v,
            'object_kps_centered': object_kps_centered,
            'object_v_centered': object_v_centered,
            'hoi_rot6d': self.hoi_rot6d[batch_idx:batch_idx+batch_size],
            'hoi_rotmat': hoi_rotmat,
            'hoi_trans': self.hoi_trans[batch_idx:batch_idx+batch_size],
            'object_scale': self.object_scale[batch_idx:batch_idx+batch_size],
        }
        return results
