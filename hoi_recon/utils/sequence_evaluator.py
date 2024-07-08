###########################################################################################
# adapted from CHORE: https://github.com/xiexh20/CHORE
# (recon/eval/chamfer_distance.py, recon/eval/pose_utils.py and recon/eval/evaluate.py)
###########################################################################################
import trimesh
import numpy as np
import torch
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors


def get_recon_meshes(dataset_metadata, img_id, smpl, recon_result):
    device = torch.device('cuda')
    betas = torch.tensor(recon_result['smpl_betas'], dtype=torch.float32).reshape(1, 10).to(device)
    body_pose_rotmat = torch.tensor(recon_result['smpl_body_rotmat'], dtype=torch.float32).reshape(1, 21, 3, 3).to(device)

    smpl_out = smpl(betas=betas,
                    body_pose=body_pose_rotmat,
                    global_orient=torch.eye(3, dtype=torch.float32, device=device).reshape(1, 3, 3),
                    transl=torch.zeros(3, dtype=torch.float32, device=device).reshape(1, 3))
    smpl_v = smpl_out.vertices
    smpl_J = smpl_out.joints
    smpl_v = smpl_v - smpl_J[:, 0:1]

    hoi_rotmat = recon_result['hoi_rotmat']
    hoi_trans = recon_result['hoi_trans']
    smpl_v = smpl_v.detach().cpu().numpy().reshape(-1, 3)
    smpl_v = np.matmul(smpl_v, hoi_rotmat.T) + hoi_trans.reshape(1, 3)
    smpl_f = smpl.faces

    obj_rel_rotmat = recon_result['obj_rel_rotmat']
    obj_rel_trans = recon_result['obj_rel_trans']
    obj_rotmat = np.matmul(hoi_rotmat, obj_rel_rotmat)
    obj_trans = np.matmul(hoi_rotmat, obj_rel_trans.reshape(3, 1)).reshape(3, ) + hoi_trans
    object_name = dataset_metadata.parse_object_name(img_id)
    object_v, object_f = dataset_metadata.obj_mesh_templates[object_name]
    object_v = np.matmul(object_v.copy(), obj_rotmat.T) + obj_trans.reshape(1, 3) 

    smpl_mesh = trimesh.Trimesh(smpl_v, smpl_f, process=False)
    object_mesh = trimesh.Trimesh(object_v, object_f, process=False)
    return smpl_mesh, object_mesh


def get_gt_meshes(metadata, img_id, smpl_male, smpl_female, gt_annotations):
    anno = gt_annotations[img_id]

    device = torch.device('cuda')
    betas = torch.tensor(anno['smplh_betas'], dtype=torch.float32).reshape(1, 10).to(device)
    body_pose_rotmat = torch.tensor(anno['smplh_pose_rotmat'][1:22], dtype=torch.float32).reshape(1, 21, 3, 3).to(device)
    global_orient = torch.tensor(anno['smplh_pose_rotmat'][:1], dtype=torch.float32).reshape(1, 3, 3).to(device)
    transl = torch.tensor(anno['smplh_trans'].astype(np.float32)).reshape(1, 3).to(device)

    if anno['gender'] == 'male':
        smpl = smpl_male
    else:
        smpl = smpl_female

    smpl_out = smpl(betas=betas,
                    body_pose=body_pose_rotmat,
                    global_orient=global_orient,
                    transl=transl)
    smpl_v = smpl_out.vertices.detach().cpu().numpy().reshape(-1, 3)
    smpl_f = smpl.faces.astype(np.int64)

    smpl_mesh = trimesh.Trimesh(smpl_v, smpl_f, process=False)

    obj_rotmat = anno['object_rotmat']
    obj_trans = anno['object_trans']
    object_name = metadata.parse_object_name(img_id)
    object_v, object_f = metadata.obj_mesh_templates[object_name]
    object_v = np.matmul(object_v.copy(), obj_rotmat.T) + obj_trans.reshape(1, 3) 
    object_mesh = trimesh.Trimesh(object_v, object_f, process=False)

    return smpl_mesh, object_mesh


class ReconEvaluator:

    def __init__(self, window_len=1, align_mesh=True, smpl_only=False):
        self.align = ProcrusteAlign(smpl_only=smpl_only, window_len=window_len)
        self.align_mesh = align_mesh
        self.sample_num = 10000


    def compute_errors(self, gt_meshes, recon_meshes):
        if self.align_mesh:
            aligned_meshes = self.align.align_meshes(gt_meshes, recon_meshes)
        else:
            aligned_meshes = recon_meshes

        gt_smpl_points = [mesh.sample(self.sample_num) for mesh in gt_meshes[0]]
        gt_object_points = [mesh.sample(self.sample_num) for mesh in gt_meshes[1]]

        aligned_smpl_points = [mesh.sample(self.sample_num) for mesh in aligned_meshes[0]]
        aligned_object_points = [mesh.sample(self.sample_num) for mesh in aligned_meshes[1]]

        smpl_chamfer_dist, object_chamfer_dist = [], []
        for i in tqdm(range(len(gt_smpl_points)), desc='calculate chamfer distance'):
            smpl_chamfer_dist.append(chamfer_distance(gt_smpl_points[i], aligned_smpl_points[i]))
            object_chamfer_dist.append(chamfer_distance(gt_object_points[i], aligned_object_points[i]))

        return smpl_chamfer_dist, object_chamfer_dist


def chamfer_distance(x, y, metric='l2', direction='bi'):
    if direction == 'y_to_x':
        x_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(x)
        min_y_to_x = x_nn.kneighbors(y)[0]
        chamfer_dist = np.mean(min_y_to_x)
    elif direction == 'x_to_y':
        y_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(y)
        min_x_to_y = y_nn.kneighbors(x)[0]
        chamfer_dist = np.mean(min_x_to_y)
    elif direction == 'bi':
        x_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(x)
        min_y_to_x = x_nn.kneighbors(y)[0]
        y_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(y)
        min_x_to_y = y_nn.kneighbors(x)[0]
        chamfer_dist = np.mean(min_y_to_x) + np.mean(min_x_to_y)
    else:
        raise ValueError("Invalid direction type. Supported types: \'y_x\', \'x_y\', \'bi\'")

    return chamfer_dist


class ProcrusteAlign:

    def __init__(self, smpl_only=False, window_len=1):
        self.smpl_only = smpl_only
        self.window_len = window_len


    def align_meshes(self, ref_meshes, recon_meshes):
        ret_smpl_meshes, ret_object_meshes = [], []

        num_meshes = len(ref_meshes[0])
        for i in tqdm(range(num_meshes), desc="Align Mehes"):
            if self.smpl_only:
                combine_ref_v = np.concatenate([m.vertices for m in ref_meshes[0][i: i + self.window_len]])
                combine_recon_v = np.concatenate([m.vertices for m in recon_meshes[0][i: i + self.window_len]])
                R, t, scale, transposed = compute_transform(combine_recon_v, combine_ref_v)
            else:
                combine_ref_smpl_v = np.concatenate([m.vertices for m in ref_meshes[0][i: i + self.window_len]])
                combine_recon_smpl_v = np.concatenate([m.vertices for m in recon_meshes[0][i: i + self.window_len]])
                combine_ref_obj_v = np.concatenate([m.vertices for m in ref_meshes[1][i: i + self.window_len]])
                combine_recon_obj_v = np.concatenate([m.vertices for m in recon_meshes[1][i: i + self.window_len]])
                combine_recon_v = np.concatenate([combine_recon_smpl_v, combine_recon_obj_v])
                combine_ref_v = np.concatenate([combine_ref_smpl_v, combine_ref_obj_v])

                R, t, scale, transposed = compute_transform(combine_recon_v, combine_ref_v)

            ret_smpl_mesh_v = (scale * R.dot(recon_meshes[0][i].vertices.T) + t).T
            ret_smpl_mesh = trimesh.Trimesh(ret_smpl_mesh_v, recon_meshes[0][i].faces)
            ret_smpl_meshes.append(ret_smpl_mesh)

            ret_object_mesh_v = (scale * R.dot(recon_meshes[1][i].vertices.T) + t).T
            ret_object_mesh = trimesh.Trimesh(ret_object_mesh_v, recon_meshes[1][i].faces)
            ret_object_meshes.append(ret_object_mesh)

        ret_meshes = [ret_smpl_meshes, ret_object_meshes]
        return ret_meshes


def compute_transform(S1, S2):
    transposed = False
    if S1.shape[0] != 3 and S1.shape[0] != 2:
        S1 = S1.T
        S2 = S2.T
        transposed = True
    assert(S2.shape[1] == S1.shape[1])

    # 1. Remove mean.
    mu1 = S1.mean(axis=1, keepdims=True)
    mu2 = S2.mean(axis=1, keepdims=True)
    X1 = S1 - mu1
    X2 = S2 - mu2

    # 2. Compute variance of X1 used for scale.
    var1 = np.sum(X1**2)

    # 3. The outer product of X1 and X2.
    K = X1.dot(X2.T)

    # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
    # singular vectors of K.
    U, s, Vh = np.linalg.svd(K)
    V = Vh.T
    # Construct Z that fixes the orientation of R to get det(R)=1.
    Z = np.eye(U.shape[0])
    Z[-1, -1] *= np.sign(np.linalg.det(U.dot(V.T)))
    # Construct R.
    R = V.dot(Z.dot(U.T))

    # 5. Recover scale.
    scale = np.trace(R.dot(K)) / var1

    # 6. Recover translation.
    t = mu2 - scale*(R.dot(mu1))

    return R, t, scale, transposed
