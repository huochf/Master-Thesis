import numpy as np
import torch
import trimesh
from sklearn.neighbors import NearestNeighbors


def get_recon_meshes(dataset_metadata, img_id, smpl, recon_result):
    device = torch.device('cuda')
    betas = torch.tensor(recon_result['betas'], dtype=torch.float32).reshape(1, 10).to(device)
    body_pose_rotmat = torch.tensor(recon_result['body_pose_rotmat'], dtype=torch.float32).reshape(1, 21, 3, 3).to(device)

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

    obj_rel_rotmat = recon_result['obj_rel_R']
    obj_rel_trans = recon_result['obj_rel_T']
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



###########################################################################################
# adapted from CHORE: https://github.com/xiexh20/CHORE
# (recon/eval/chamfer_distance.py, recon/eval/pose_utils.py and recon/eval/evaluate.py)
###########################################################################################

class ReconEvaluator:

    def __init__(self, align_mesh=True, smpl_only=False):
        self.align = ProcrusteAlign(smpl_only=smpl_only)
        self.align_mesh = align_mesh
        self.sample_num = 10000


    def compute_errors(self, gt_meshes, recon_meshes):
        if self.align_mesh:
            aligned_meshes = self.align.align_meshes(gt_meshes, recon_meshes)
        else:
            aligned_meshes = recon_meshes
        gt_points = [mesh.sample(self.sample_num) for mesh in gt_meshes]
        aligned_points = [mesh.sample(self.sample_num) for mesh in aligned_meshes]
        
        chamfer_dist = [chamfer_distance(gt_p, aligned_p) for gt_p, aligned_p in zip(gt_points, aligned_points)]
        return chamfer_dist


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

    def __init__(self, smpl_only=False):
        self.smpl_only = smpl_only


    def align_meshes(self, ref_meshes, recon_meshes):
        ref_v, recon_v = [], []
        v_lens = []
        R, recon_v, scale, t = self.get_transform(recon_meshes, recon_v, ref_meshes, ref_v, v_lens)

        recon_hat = (scale * R.dot(recon_v.T) + t).T
        ret_meshes = []
        last_idx = 0
        offset = 0
        for m in recon_meshes:
            newm = trimesh.Trimesh(recon_hat[offset:offset + len(m.vertices)], m.faces, process=False)
            ret_meshes.append(newm)
            offset += len(m.vertices)
        return ret_meshes


    def get_transform(self, recon_meshes, recon_v, ref_meshes, ref_v, v_lens):
        offset = 0
        recon_v, ref_v = self.comb_meshes(offset, recon_meshes, recon_v, ref_meshes, ref_v, v_lens)
        if ref_v.shape == recon_v.shape and not self.smpl_only:
            R, t, scale, transposed = compute_transform(recon_v, ref_v)
            return R, recon_v, scale, t
        else:
            smpl_recon_v = recon_meshes[0].vertices
            smpl_ref_v = ref_meshes[0].vertices
            R, t, scale, transposed = compute_transform(smpl_recon_v, smpl_ref_v)
            return R, recon_v, scale, t


    def comb_meshes(self, offset, recon_meshes, recon_v, ref_meshes, ref_v, v_lens):
        for fm, rm in zip(ref_meshes, recon_meshes):
            ref_v.append(fm.vertices)
            recon_v.append(rm.vertices)
            offset += fm.vertices.shape[0]
            v_lens.append(offset)
        ref_v = np.concatenate(ref_v, 0)
        recon_v = np.concatenate(recon_v, 0)
        return recon_v, ref_v


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
