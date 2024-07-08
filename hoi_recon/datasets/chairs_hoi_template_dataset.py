import os
import pickle
import trimesh
import numpy as np
from scipy.spatial.transform import Rotation


NUM_OBJECT = 92
MESH_VERTICES_NUM_MAX = 4000
OBJECT_PARTS_NUM = 7

part_name2id = {
    'chair_head': 0,
    'chair_back': 1,
    'chair_arm_left': 2,
    'chair_arm_right': 3,
    'chair_seat': 4,
    'chair_base': 5,
    'footrest': 6,
}


train_object_ids = [17, 24, 25, 26, 30, 33, 44, 46, 48, 49, 
                    59, 60, 64, 68, 75, 81, 83, 85, 87, 96, 
                    98, 99, 103, 104, 110, 111, 116, 117, 118, 121, 
                    123, 130, 131, 141, 142, 143, 147, 151, 152, 154, 
                    156, 157, 158, 162, 166, 168, 171, 173, 176, 180, 
                    181]
test_object_ids = [15, 29, 36, 43, 45, 92, 109, 149, 172]


class CHAIRSDataset:

    def __init__(self, root_dir, res=64, split='train'):
        self.root_dir = root_dir

        self.human_betas = np.load(os.path.join(root_dir, 'AHOI_Data', 'DATA_FOLDER', 'human_betas.npy'))
        self.human_orient = np.load(os.path.join(root_dir, 'AHOI_Data', 'DATA_FOLDER', 'human_orient.npy'))
        self.human_pose = np.load(os.path.join(root_dir, 'AHOI_Data', 'DATA_FOLDER', 'human_pose.npy'))
        self.human_transl = np.load(os.path.join(root_dir, 'AHOI_Data', 'DATA_FOLDER', 'human_transl.npy'))
        self.human_pose = np.load(os.path.join(root_dir, 'AHOI_Data', 'DATA_FOLDER', 'human_pose.npy'))

        self.object_ids = np.load(os.path.join(root_dir, 'AHOI_Data', 'DATA_FOLDER', 'object_id.npy'))
        self.object_location = np.load(os.path.join(root_dir, 'AHOI_Data', 'DATA_FOLDER', 'object_location.npy'))
        self.object_rotation = np.load(os.path.join(root_dir, 'AHOI_Data', 'DATA_FOLDER', 'object_rotation.npy'))
        self.object_root_location = np.load(os.path.join(root_dir, 'AHOI_Data', 'DATA_FOLDER', 'object_root_location.npy'))
        self.object_root_rotation = np.load(os.path.join(root_dir, 'AHOI_Data', 'DATA_FOLDER', 'object_root_rotation.npy'))
        self.object_info = np.load(os.path.join(root_dir, 'AHOI_Data', 'AHOI_ROOT', 'Metas', 'object_info.npy'), allow_pickle=True).item()

        with open(os.path.join(root_dir, 'AHOI_Data', 'AHOI_ROOT', 'Metas', 'object_meta.pkl'), 'rb') as f:
            self.object_meda = pickle.load(f) 

        self.indices = []
        if split == 'train':
            dataset_object_ids = train_object_ids
        else:
            dataset_object_ids = test_object_ids

        for idx, object_id in enumerate(self.object_ids):
            if object_id in dataset_object_ids:
                self.indices.append(idx)

        self.indices = np.arange(self.human_betas.shape[0])
        # self.indices.extend(np.where(self.object_ids == 147)[0].tolist())
        # print(self.indices)

        self.object_voxels, self.voxel_scale = self.load_voxels(res=res)
        self.object_verts_all = self.load_object_meshes()


    def load_voxels(self, res=64):
        voxel_all = {}
        voxel_dir = os.path.join(self.root_dir, 'AHOI_Data', 'AHOI_ROOT', 'object_voxel_{}'.format(res))
        for file in os.listdir(voxel_dir):
            object_id = file.split('.')[0]
            voxel_all[object_id] = np.load(os.path.join(voxel_dir, file))

        with open(os.path.join(self.root_dir, 'AHOI_Data', 'AHOI_ROOT', 'Metas', 'object_voxel_scale.pkl'), 'rb') as f:
            voxel_scale = pickle.load(f)
        return voxel_all, voxel_scale


    def load_object_meshes(self, ):
        object_verts_all = np.zeros((NUM_OBJECT, MESH_VERTICES_NUM_MAX, 3))

        object_mesh_dir = os.path.join(self.root_dir, 'AHOI_Data', 'AHOI_ROOT', 'Meshes_wt')
        for object_id in os.listdir(object_mesh_dir):
            object_ind_in_meta = np.where(self.object_info['object_ids'] == int(object_id))[0][0]

            object_v = []
            object_f = []

            vertice_count = 0
            for file in os.listdir(os.path.join(object_mesh_dir, object_id)):

                if file.split('.')[-1] != 'obj':
                    continue
                part_name = file.split('.')[0]
                if part_name not in self.object_meda[str(object_id)]['init_shift']:
                    continue
                if part_name not in part_name2id: # other
                    continue
                part_idx = part_name2id[part_name]

                object_v_len = self.object_info['vertex_len'][object_ind_in_meta][part_idx]
                if object_v_len == 0:
                    continue

                mesh = trimesh.load(os.path.join(object_mesh_dir, object_id, file), process=False)
                part_v = mesh.vertices
                part_f = mesh.faces

                trans_init = self.object_meda[str(object_id)]['init_shift'][part_name]
                part_v = part_v + trans_init.reshape(1, 3)
                object_v.append(part_v)
                object_f.append(part_f + vertice_count)
                vertice_count += part_v.shape[0]

            object_v = np.concatenate(object_v)
            object_f = np.concatenate(object_f)
            object_mesh = trimesh.Trimesh(object_v, object_f)
            verts = trimesh.sample.sample_surface(object_mesh, MESH_VERTICES_NUM_MAX)[0]
            object_verts_all[object_ind_in_meta, :] = verts

        return object_verts_all


    def __len__(self, ):
        return len(self.indices)


    def __getitem__(self, idx):
        idx = self.indices[idx]
        human_betas = self.human_betas[idx]
        human_orient = self.human_orient[idx]
        human_pose = self.human_pose[idx]
        human_transl = self.human_transl[idx]
        human_pose = self.human_pose[idx]

        object_id = self.object_ids[idx]
        object_location = self.object_root_location[idx]
        object_rotation = self.object_root_rotation[idx]

        object_R = Rotation.from_euler('xyz', [0, np.pi, 0]).as_matrix() @ \
                   Rotation.from_euler('xyz', object_rotation).as_matrix()
        object_T =  Rotation.from_euler('xyz', [0, np.pi, 0]).as_matrix() @ object_location.reshape(3, 1)
        object_T = object_T.reshape(1, 3)

        object_ind_in_meta = np.where(self.object_info['object_ids'] == int(object_id))[0][0]
        object_v = self.object_verts_all[object_ind_in_meta] # [n, 3]

        # assert not np.isnan(object_R).any()
        # assert not np.isnan(object_T).any()

        object_voxel = self.object_voxels[str(object_id)]
        voxel_mean = self.voxel_scale[str(object_id)]['mean']
        voxel_scale = self.voxel_scale[str(object_id)]['scale']

        outputs = {
            'object_id': object_id,
            'human_betas': human_betas.astype(np.float32),
            'human_orient': human_orient.astype(np.float32),
            'human_pose': human_pose.astype(np.float32),
            'human_transl': human_transl.astype(np.float32),
            'object_T': object_T.astype(np.float32),
            'object_R': object_R.astype(np.float32),
            'object_v': object_v.astype(np.float32),

            'object_voxel': object_voxel.astype(np.float32),
            'voxel_mean': voxel_mean.astype(np.float32),
            'voxel_scale': np.array(voxel_scale).astype(np.float32),
        }

        return outputs
