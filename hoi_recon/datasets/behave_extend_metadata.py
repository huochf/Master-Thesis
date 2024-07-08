import os
from tqdm import tqdm
import trimesh
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R

from hoi_recon.datasets.utils import load_json, load_pickle


class BEHAVEExtendMetaData():

    def __init__(self, root_dir, preload_annotations=False, ):
        self.root_dir = root_dir
        self.IMG_HEIGHT = 1536
        self.IMG_WIDTH = 2048

        self.OBJECT_NAME2IDX = {
            'backpack': 0,
            'basketball': 1,
            'boxlarge': 2,
            'boxlong': 3,
            'boxmedium': 4,
            'boxsmall': 5,
            'boxtiny': 6,
            'chairblack': 7,
            'chairwood': 8,
            'keyboard': 9,
            'monitor': 10,
            'plasticcontainer': 11,
            'stool': 12,
            'suitcase': 13,
            'tablesmall': 14,
            'tablesquare': 15,
            'toolbox': 16,
            'trashbin': 17,
            'yogaball': 18,
            'yogamat': 19,
        }
        self.OBJECT_IDX2NAME = {v: k for k, v in self.OBJECT_NAME2IDX.items()}
        self.OBJECT_COORDINATE_NORM = {
            'basketball': [0.12236282829610301, 0.13042416009636515, 0.13102541633815146], 
            'tablesmall': [0.29788815447386074, 0.4007032965176957, 0.27832651484960635], 
            'toolbox': [0.09294969423491206, 0.13222591918103288, 0.1673733647629312], 
            'plasticcontainer': [0.3358690018561631, 0.22639700585163278, 0.25677905980620397], 
            'boxlarge': [0.26936167564988484, 0.30247685136558083, 0.28287613106284965], 
            'trashbin': [0.15079510307636423, 0.1896149735794419, 0.1603298959826267], 
            'stool': [0.23256295110606504, 0.27003098172428946, 0.2205380318634632], 
            'boxsmall': [0.16008218470133753, 0.14842027491577947, 0.22244380626078553], 
            'monitor': [0.13533075340542058, 0.2979130832046061, 0.29667681821373404], 
            'keyboard': [0.10469683236514524, 0.16815164813278008, 0.2406570611341632], 
            'boxmedium': [0.14518500688264757, 0.2078229492491641, 0.25048296294494005], 
            'chairblack': [0.303116101212625, 0.4511368757997035, 0.2987161170926357], 
            'chairwood': [0.32013054983251643, 0.48153881571638113, 0.37033998297393567], 
            'suitcase': [0.16022201445086703, 0.2550602338788379, 0.2613365624202387], 
            'boxlong': [0.39511341702499553, 0.1720738671379548, 0.1971366981998387], 
            'boxtiny': [0.11570125012439958, 0.060232502239181196, 0.1634993526289597], 
            'yogaball': [0.27815387740465014, 0.26961738674524627, 0.3164645608250861], 
            'backpack': [0.2202841718619516, 0.2839561989281594, 0.19267741049215822], 
            'yogamat': [0.524749682746465, 0.2720144866073263, 0.12567161343996003], 
            'tablesquare': [0.4920387357121939, 0.48840298724966774, 0.48018395294091076]
        }
        self.SUBID_GENDER = {
            '1': 'male',
            '2': 'male',
            '3': 'male',
            '4': 'male',
            '5': 'male',
            '6': 'female',
            '7': 'female',
            '8': 'female',
        }

        self.obj_mesh_templates = self.load_object_mesh_templates()
        self.obj_keypoints_dict = self.load_object_keypoints_dict()

        self.dataset_splits = load_json(os.path.join(root_dir, 'split.json'))
        self.cam_intrinsics = self.load_cam_intrinsics()
        self.cam_RT_matrix = self.load_cam_RT_matrix()
        self.annotations = {}
        if preload_annotations:
            all_sequences = list(self.go_through_all_sequences(split='all'))
            for sequence_name in tqdm(all_sequences, desc='loading annotations'):
                try:
                    self.annotations[sequence_name] = self.load_annotations(sequence_name)
                except:
                    continue

        self.object_max_keypoint_num = 16
        self.object_num_keypoints = {
            'backpack': 8,
            'basketball': 1,
            'boxlarge': 8,
            'boxlong': 8,
            'boxmedium': 8,
            'boxsmall': 8,
            'boxtiny': 8,
            'chairblack': 16,
            'chairwood': 10,
            'keyboard': 4,
            'monitor': 8,
            'plasticcontainer': 8,
            'stool': 6,
            'suitcase': 8,
            'tablesmall': 10,
            'tablesquare': 8,
            'toolbox': 8,
            'trashbin': 2,
            'yogaball': 1,
            'yogamat': 2,
        }
        self.object_max_vertices_num = 1700
        self.object_num_vertices = {
            "backpack": 548,
            "basketball": 508,
            "boxlarge": 524,
            "boxlong": 526,
            "boxmedium": 509,
            "boxsmall": 517,
            "boxtiny": 506,
            "chairblack": 1609,
            "chairwood": 1609,
            "keyboard": 502,
            "monitor": 537,
            "plasticcontainer": 562,
            "stool": 532,
            "suitcase": 520,
            "tablesmall": 507,
            "tablesquare": 1046,
            "toolbox": 499,
            "trashbin": 547,
            "yogaball": 534,
            "yogamat": 525,
        }
        self.all_valid_frames = None
        self.hoi_trans_avg = np.array([-0.01450529,  0.32624382,  2.3658798 ]) # depth: [1.0, 3.6]


    def load_cam_intrinsics(self, ):
        cam_intrinsics_params = []
        for i in range(4):
            params = load_json(os.path.join(self.root_dir, 'calibs', 'intrinsics', str(i), 'calibration.json'))
            cx, cy = params['color']['cx'], params['color']['cy']
            fx, fy = params['color']['fx'], params['color']['fy']
            dist_coeffs = params['color']['opencv'][4:]
            # cam_intrinsics_params.append([cx, cy, fx, fy])
            cam_intrinsics_params.append(params['color']['opencv'])
        return cam_intrinsics_params


    def load_cam_RT_matrix(self, ):
        cam_RT = {}
        for day_id in range(7):
            cam_RT[str(day_id + 1)] = []
            for cam_id in range(4):
                params = load_json(os.path.join(self.root_dir, 'calibs', 'Date0{}'.format(day_id + 1), 'config', str(cam_id), 'config.json'))
                cam_RT[str(day_id + 1)].append([np.array(params['rotation']).reshape((3, 3)), np.array(params['translation'])])
        return cam_RT


    def load_object_mesh_templates(self, ):
        templates = {}
        for object_name in self.OBJECT_NAME2IDX.keys():
            object_mesh = os.path.join(self.root_dir, 'objects', object_name, '{}_f1000.ply'.format(object_name))
            if not os.path.exists(object_mesh):
                object_mesh = os.path.join(self.root_dir, 'objects', object_name, '{}_f2000.ply'.format(object_name))
            if not os.path.exists(object_mesh):
                object_mesh = os.path.join(self.root_dir, 'objects', object_name, '{}_f2500.ply'.format(object_name))
            if not os.path.exists(object_mesh):
                object_mesh = os.path.join(self.root_dir, 'objects', object_name, '{}_closed_f1000.ply'.format(object_name))
            assert os.path.exists(object_mesh)

            object_mesh = trimesh.load(object_mesh, process=False)
            object_vertices = np.array(object_mesh.vertices).astype(np.float32)
            object_faces = np.array(object_mesh.faces).astype(np.int64)
            object_vertices = object_vertices - object_vertices.mean(axis=0)

            templates[object_name] = (object_vertices, object_faces)

        return templates


    def load_object_trimesh(self, ):
        templates = {}
        for object_name in self.OBJECT_NAME2IDX.keys():
            object_mesh = os.path.join(self.root_dir, 'objects', object_name, '{}_f1000.ply'.format(object_name))
            if not os.path.exists(object_mesh):
                object_mesh = os.path.join(self.root_dir, 'objects', object_name, '{}_f2000.ply'.format(object_name))
            if not os.path.exists(object_mesh):
                object_mesh = os.path.join(self.root_dir, 'objects', object_name, '{}_f2500.ply'.format(object_name))
            if not os.path.exists(object_mesh):
                object_mesh = os.path.join(self.root_dir, 'objects', object_name, '{}_closed_f1000.ply'.format(object_name))
            assert os.path.exists(object_mesh)

            object_mesh = trimesh.load(object_mesh, process=False)
            object_vertices = np.array(object_mesh.vertices).astype(np.float32)
            object_mesh.vertices = object_vertices - object_vertices.mean(axis=0)

            templates[object_name] = object_mesh

        return templates


    def parse_object_name(self, img_id):
        day_id, sub_id, obj_name, inter_type, frame_id, cam_id = img_id.split('_')
        return obj_name


    def load_object_keypoints_dict(self, ):
        keypoints_dict = {}
        for object_name in self.OBJECT_NAME2IDX.keys():
            try:
                keypoints_path = os.path.join('./data/datasets/behave_obj_keypoints/{}_keypoints.json'.format(object_name))
                keypoints = load_json(keypoints_path)
            except:
                keypoints_path = os.path.join('../data/datasets/behave_obj_keypoints/{}_keypoints.json'.format(object_name))
                keypoints = load_json(keypoints_path)
            keypoints_dict[object_name] = keypoints
        return keypoints_dict


    def load_object_keypoints(self, obj_name):
        object_vertices, _ = self.obj_mesh_templates[obj_name]
        keypoints_dict = self.obj_keypoints_dict[obj_name]
        keypoints = []
        for k, v in keypoints_dict.items():
            keypoints.append(object_vertices[v].mean(0))
        keypoints = np.array(keypoints)
        return keypoints


    def parse_seq_info(self, sequence_name):
        try:
            day_name, sub_name, obj_name, inter_type = sequence_name.split('_')
        except:
            day_name, sub_name, obj_name = sequence_name.split('_')
            inter_type = 'none'
        day_id, sub_id = day_name[5:], sub_name[4:]

        return day_id, sub_id, obj_name, inter_type


    def go_through_all_frames(self, split='all'):
        if self.all_valid_frames is None:
            self.all_valid_frames = load_pickle(os.path.join(self.root_dir, 'behave_extend_valid_frames.pkl'))
        all_valid_frames = self.all_valid_frames

        if split == 'train':
            sequences = self.dataset_splits['train']
            all_valid_frames = [img_id for img_id in self.all_valid_frames if self.in_train_set(img_id)]
        if split == 'test':
            sequences = self.dataset_splits['train']
            all_valid_frames = [img_id for img_id in self.all_valid_frames if not self.in_train_set(img_id)]

        for img_id in all_valid_frames:
            yield img_id


    def go_through_all_sequences(self, split='all'):
        
        if split == 'test':
            all_sequences = self.dataset_splits['test']
        elif split == 'train':
            all_sequences = self.dataset_splits['train']
        elif split == 'all':
            all_sequences = self.dataset_splits['train'] + self.dataset_splits['test']

        for name in all_sequences:
            yield name


    def go_through_sequence(self, sequence_name):
        if self.all_valid_frames is None:
            self.all_valid_frames = load_pickle(os.path.join(self.root_dir, 'behave_extend_valid_frames.pkl'))

        target_frames = []
        for img_id in self.all_valid_frames:
            day_id, sub_id, obj_name, inter_type, frame_id, cam_id = img_id.split('_')
            seq = self.get_sequence_name(day_id, sub_id, obj_name, inter_type)
            if seq == sequence_name:
                target_frames.append(img_id)
        target_frames = sorted(target_frames)

        for img_id in target_frames:
            yield img_id


    def get_all_image_by_sequence(self, split='all'):
        if self.all_valid_frames is None:
            self.all_valid_frames = load_pickle(os.path.join(self.root_dir, 'behave_extend_valid_frames.pkl'))

        all_valid_frames = {item: True for item in self.all_valid_frames} # list to dict, for faster lookup

        all_sequences = list(self.go_through_all_sequences(split))
        img_ids_by_seq = {}
        for sequence_name in all_sequences:
            img_ids_by_seq[sequence_name] = {}

            day_id, sub_id, obj_name, inter_type = self.parse_seq_info(sequence_name)
            for cam_id in range(4):
                img_ids = []
                frame_list = sorted(os.listdir(os.path.join(self.root_dir, 'raw_images', sequence_name)))

                for frame_name in frame_list:
                    if frame_name == 'info.json':
                        continue
                    frame_id = frame_name[2:]
                    img_id =  '_'.join([day_id, sub_id, obj_name, inter_type, frame_id, str(cam_id)])

                    if img_id not in all_valid_frames:
                        continue

                    img_ids.append(img_id)
                img_ids_by_seq[sequence_name][cam_id] = img_ids
        return img_ids_by_seq


    def get_img_id(self, day_id, sub_id, obj_name, inter_type, frame_id, cam_id):
        return '_'.join([day_id, sub_id, obj_name, inter_type, frame_id, cam_id])


    def parse_img_id(self, img_id):
        return img_id.split('_')


    def get_image_path(self, img_id):
        day_id, sub_id, obj_name, inter_type, frame_id, cam_id = img_id.split('_')
        sequence_name = self.get_sequence_name(day_id, sub_id, obj_name, inter_type)
        img_path = os.path.join(self.root_dir, 'raw_images', sequence_name, 't0{}'.format(frame_id), 'k{}.color.jpg'.format(cam_id))
        return img_path


    def get_object_coor_path(self, img_id):
        image_path = self.get_image_path(img_id)
        coor_path = image_path.replace('raw_images', '_object_coor_maps_extend').replace('color.jpg', 'obj_coor.pkl')
        return coor_path


    def get_object_full_mask_path(self, img_id,):
        image_path = self.get_image_path(img_id)
        mask_path = image_path.replace('raw_images', '_object_coor_maps_extend').replace('color', 'mask_full')

        return mask_path


    def get_object_sam_mask_path(self, img_id):
        day_id, sub_id, obj_name, inter_type, frame_id, cam_id = img_id.split('_')
        sam_path = os.path.join(self.root_dir, 'sam', obj_name, '{}.jpg'.format(img_id))
        assert os.path.exists(sam_path)
        return sam_path


    def get_person_mask_path(self, img_id, for_aug=False):
        img_path = self.get_image_path(img_id)
        mask_path = img_path.replace('raw_images', 'person_mask').replace('color', 'person_mask')

        return mask_path


    def get_sequence_name(self, day_id, sub_id, obj_name, inter_type):
        if inter_type == 'none':
            sequence_name = 'Date0{}_Sub0{}_{}'.format(day_id, sub_id, obj_name)
        else:
            sequence_name = 'Date0{}_Sub0{}_{}_{}'.format(day_id, sub_id, obj_name, inter_type)
        return sequence_name


    def load_annotations(self, sequence_name):
        object_anno_file = os.path.join(self.root_dir, 'behave-30fps-params-v1', sequence_name, 'object_fit_all.npz')
        smpl_anno_file = os.path.join(self.root_dir, 'behave-30fps-params-v1', sequence_name, 'smpl_fit_all.npz')
        object_fit = np.load(object_anno_file)
        smpl_fit = np.load(smpl_anno_file)

        annotations = {}
        for idx, frame_name in enumerate(smpl_fit['frame_times']):
            annotations[frame_name] = {
                'poses': smpl_fit['poses'][idx],
                'betas': smpl_fit['betas'][idx],
                'trans': smpl_fit['trans'][idx],
            }
        for idx, frame_name in enumerate(object_fit['frame_times']):
            if frame_name not in annotations:
                annotations[frame_name] = {}
            annotations[frame_name]['ob_pose'] = object_fit['angles'][idx]
            annotations[frame_name]['ob_trans'] = object_fit['trans'][idx]

        return annotations


    def load_object_RT(self, img_id):
        day_id, sub_id, obj_name, inter_type, frame_id, cam_id = self.parse_img_id(img_id)

        sequence_name = self.get_sequence_name(day_id, sub_id, obj_name, inter_type)
        if sequence_name not in self.annotations:
            print('loading annotations for {}'.format(sequence_name))
            self.annotations[sequence_name] = self.load_annotations(sequence_name)

        annotation = self.annotations[sequence_name]['t0' + frame_id]
        obj_axis_angle = annotation['ob_pose']
        obj_rotmat = R.from_rotvec(obj_axis_angle).as_matrix()
        obj_trans = annotation['ob_trans']

        return obj_rotmat, obj_trans


    def load_smpl_params(self, img_id):
        day_id, sub_id, obj_name, inter_type, frame_id, cam_id = self.parse_img_id(img_id)
        sequence_name = self.get_sequence_name(day_id, sub_id, obj_name, inter_type)
        if sequence_name not in self.annotations:
            print('loading annotations for {}'.format(sequence_name))
            self.annotations[sequence_name] = self.load_annotations(sequence_name)

        annotation = self.annotations[sequence_name]['t0' + frame_id]
        smpl_params = {k: v for k, v in annotation.items() if 'ob_' not in k}
        return smpl_params


    def get_sub_gender(self, img_id):
        day_id, sub_id, obj_name, inter_type, frame_id, cam_id = self.parse_img_id(img_id)
        return self.SUBID_GENDER[sub_id]


    def get_pred_coor_map_path(self, img_id):
        image_path = self.get_image_path(img_id)
        coor_path = image_path.replace('raw_images', '_epro_pnp_extend').replace('color.jpg', 'obj_coor.pkl')
        return coor_path


    def get_pred_dino_coor_map_path(self, img_id):
        day_id, sub_id, obj_name, inter_type, frame_id, cam_id = img_id.split('_')
        seq_name = self.get_sequence_name(day_id, sub_id, obj_name, inter_type)
        coor_path = '/inspurfs/group/wangjingya/huochf/datasets_hot_data/BEHAVE_extend/dino_corr/{}/{}/{}.pkl'.format(seq_name, cam_id, img_id)
        return coor_path


    def get_openpose_path(self, img_id):
        image_path = self.get_image_path(img_id)
        openpose_path = image_path.replace('raw_images', 'openpose_extend').replace('color.jpg', 'color_keypoints.json')
        return openpose_path


    def get_vitpose_path(self, img_id):
        vitpose_dir = '/inspurfs/group/wangjingya/huochf/datasets_hot_data/BEHAVE_extend/vitpose_split/'
        day_id, sub_id, obj_name, inter_type, frame_id, cam_id = img_id.split('_')
        sequence_name = self.get_sequence_name(day_id, sub_id, obj_name, inter_type)
        path = os.path.join(vitpose_dir, sequence_name, 't0{}'.format(frame_id), 'k{}.vitpose.json'.format(cam_id))
        return path


    def get_obj_visible_ratio(self, img_id):
        object_full_mask = cv2.imread(self.get_object_full_mask_path(img_id), cv2.IMREAD_GRAYSCALE)
        object_full_mask = cv2.resize(object_full_mask, (0, 0),  fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
        person_mask = cv2.imread(self.get_person_mask_path(img_id), cv2.IMREAD_GRAYSCALE)
        try:
            object_occlusion_mask = object_full_mask.astype(np.bool_) & person_mask.astype(np.bool_)
            visible_ratio = 1 - np.sum(object_occlusion_mask != 0) / np.sum(object_full_mask != 0)
        except: # mask may not exists
            print('Exception occurs during loading masks.')
            visible_ratio = 0.
        return visible_ratio


    def in_train_set(self, img_id):
        day_id, sub_id, obj_name, inter_type, frame_id, cam_id = self.parse_img_id(img_id)
        sequence_name = self.get_sequence_name(day_id, sub_id, obj_name, inter_type)
        return sequence_name not in self.dataset_splits['test']


    def get_object_sym_RT(self, object_name):
        if object_name == 'backpack':
            return np.eye(3), np.zeros(3)
        elif object_name == 'basketball':
            return R.random().as_matrix(), np.zeros(3)
        elif object_name == 'boxlarge':
            rotmat = np.eye(4)
            if np.random.random() < 0.5:
                rotmat = np.matmul(BOXLARGE_SYMRT[0], rotmat)
            if np.random.random() < 0.5:
                rotmat = np.matmul(BOXLARGE_SYMRT[1], rotmat)
            prob = np.random.random()
            if prob < 0.25:
                rotmat = np.matmul(BOXLARGE_SYMRT[2], rotmat)
            elif prob < 0.5:
                rotmat = np.matmul(BOXLARGE_SYMRT[3], rotmat)
            elif prob < 0.75:
                rotmat = np.matmul(BOXLARGE_SYMRT[4], rotmat)
            return rotmat[:3, :3], rotmat[:3, 3]
        elif object_name == 'boxlong':
            rotmat = np.eye(4)
            if np.random.random() < 0.5:
                rotmat = np.matmul(BOXLONG_SYMRT[0], rotmat)
            if np.random.random() < 0.5:
                rotmat = np.matmul(BOXLONG_SYMRT[1], rotmat)
            prob = np.random.random()
            if prob < 0.25:
                rotmat = np.matmul(BOXLONG_SYMRT[2], rotmat)
            elif prob < 0.5:
                rotmat = np.matmul(BOXLONG_SYMRT[3], rotmat)
            elif prob < 0.75:
                rotmat = np.matmul(BOXLONG_SYMRT[4], rotmat)
            return rotmat[:3, :3], rotmat[:3, 3]
        elif object_name == 'boxmedium':
            rotmat = np.eye(4)
            for rot_idx in range(BOXMEDIUM_SYMRT.shape[0]):
                if np.random.random() < 0.5:
                    rotmat = np.matmul(BOXMEDIUM_SYMRT[rot_idx], rotmat)
            return rotmat[:3, :3], rotmat[:3, 3]
        elif object_name == 'boxsmall':
            rotmat = np.eye(4)
            for rot_idx in range(BOXSMALL_SYMRT.shape[0]):
                if np.random.random() < 0.5:
                    rotmat = np.matmul(BOXSMALL_SYMRT[rot_idx], rotmat)
            return rotmat[:3, :3], rotmat[:3, 3]
        elif object_name == 'boxtiny':
            rotmat = np.eye(4)
            for rot_idx in range(BOXTINY_SYMRT.shape[0]):
                if np.random.random() < 0.5:
                    rotmat = np.matmul(BOXTINY_SYMRT[rot_idx], rotmat)
            return rotmat[:3, :3], rotmat[:3, 3]
        elif object_name == 'chairblack':
            return np.eye(3), np.zeros(3)
        elif object_name == 'chairwood':
            return np.eye(3), np.zeros(3)
        elif object_name == 'keyboard':
            return np.eye(3), np.zeros(3)
        elif object_name == 'monitor':
            return np.eye(3), np.zeros(3)
        elif object_name == 'plasticcontainer':
            rotmat = np.eye(4)
            if np.random.random() < 0.5:
                rotmat = np.matmul(PLASTICCONTAINER_SYMRT[0], rotmat)
            return rotmat[:3, :3], rotmat[:3, 3]
        elif object_name == 'stool':
            rotmat = np.eye(4)
            prob = np.random.random()
            if prob < 0.3:
                rotmat = np.matmul(STOOL_SYMRT[0], rotmat)
            elif prob < 0.6:
                rotmat = np.matmul(STOOL_SYMRT[1], rotmat)
            return rotmat[:3, :3], rotmat[:3, 3]
        elif object_name == 'suitcase':
            return np.eye(3), np.zeros(3)
        elif object_name == 'tablesmall':
            return np.eye(3), np.zeros(3)
        elif object_name == 'tablesquare':
            rotmat = np.eye(4)
            prob = np.random.random()
            if prob < 0.25:
                rotmat = np.matmul(TABLESQUARE_SYMRT[0], rotmat)
            elif prob < 0.5:
                rotmat = np.matmul(TABLESQUARE_SYMRT[1], rotmat)
            elif prob < 0.75:
                rotmat = np.matmul(TABLESQUARE_SYMRT[2], rotmat)
            return rotmat[:3, :3], rotmat[:3, 3]
        elif object_name == 'toolbox':
            rotmat = np.eye(4)
            if np.random.random() < 0.5:
                rotmat = np.matmul(TOOLBOX_SYMRT[0], rotmat)
            return rotmat[:3, :3], rotmat[:3, 3]
        elif object_name == 'trashbin':
            axis = np.array([0.0, 1.5, 0.1])
            rotmat = R.from_rotvec(np.random.random() * np.pi * 2 * axis / np.linalg.norm(axis)).as_matrix()
            trans = np.array([-0.00329292, 0.02212786, 0.01992773])
            trans = np.matmul(rotmat, - trans.reshape(3, 1)).reshape(3, ) + trans
            return rotmat, trans
        elif object_name == 'yogaball':
            return R.random().as_matrix(), np.zeros(3)
        elif object_name == 'yogamat':
            axis = np.array([0.89785126, -0.35126231, -0.0286829])
            rotmat = R.from_rotvec(np.random.random() * np.pi * 2 * axis / np.linalg.norm(axis)).as_matrix()
            trans = np.array([0.04008849, 0.00022276, -0.00475774])
            trans = np.matmul(rotmat, - trans.reshape(3, 1)).reshape(3, ) + trans
            return rotmat, trans
        else:
            assert False


BOXLARGE_SYMRT = np.array([
    [[-0.01308818, -0.06726133, -0.99764954, 0.03098779],
    [-0.06726133, -0.99541592,  0.06799315, 0.10374567],
    [-0.99764954,  0.06799315,  0.00850409, 0.02365976],
    [0, 0, 0, 1]],

    [[0.03950405,  0.07567146,  0.99634997, -0.02854362],
    [0.07567146, -0.99449144,  0.07253003, 0.10337199],
    [0.99634997,  0.07253003, -0.04501261, 0.02192894],
    [0, 0, 0, 1]],

    [[1.34868395e-05, -6.96769689e-02, -9.97569607e-01, 0.03109438],
    [7.70020262e-02, 9.94607831e-01, -6.94690575e-02, 0.00198178],
    [9.97030936e-01, -7.68139440e-02, 5.37868189e-03, 0.02850613],
    [0, 0, 0, 1]],

    [[-9.99973026e-01,  7.32505731e-03, -5.38670286e-04, 0.00251986],
    [7.32505731e-03,  9.89215663e-01, -1.46283002e-01, 0.00436692],
    [-5.38670286e-04, -1.46283002e-01, -9.89242636e-01, 0.05950929],
    [0, 0, 0, 1]],

    [[1.34868395e-05,  7.70020262e-02,  9.97030936e-01, -0.02857452],
    [-6.96769689e-02,  9.94607831e-01, -7.68139440e-02, 0.00238513],
    [-9.97569607e-01, -6.94690575e-02,  5.37868189e-03, 0.03100316],
    [0, 0, 0, 1]],
])

BOXLONG_SYMRT = np.array([
    [[-0.87911672,  0.47594085,  0.02518135, -0.02954251],
    [0.47594085,  0.87387116,  0.09914384, 0.00797974],
    [0.02518135,  0.09914384, -0.99475444, -0.00900233],
    [0, 0, 0, 1]],

    [[-0.78635143, -0.03144197, -0.61697879, -0.01054051],
    [-0.03144197, -0.99537279, 0.09079879, 0.07930199],
    [-0.61697879, 0.09079879, 0.78172422, -0.00769131],
    [0, 0, 0, 1]],

    [[0.82620347,  0.15766625,  0.54085967, -0.00583958],
    [-0.55340964,  0.0473893,   0.83156, 0.03684994],
    [0.10547798, -0.98635471,  0.12640723, 0.03749199],
    [0, 0, 0, 1]],

    [[0.65240694, -0.39574339,  0.64633765, 0.01542363],
    [-0.39574339, -0.90522139, -0.15479471, 0.07300476],
    [0.64633765, -0.15479471, -0.74718554, 0.00526819],
    [0, 0, 0, 1]],

    [[0.82620347, -0.55340964,  0.10547798, 0.02126322],
    [0.15766625,  0.0473893,  -0.98635471, 0.03615482],
    [0.54085967,  0.83156,     0.12640723, -0.0322238],
    [0, 0, 0, 1]],
])

BOXMEDIUM_SYMRT = np.array([
    [[-0.98627166,  0.00183188, -0.16512076, 0.01977026],
    [0.00183188, -0.99975556, -0.02203333, 0.14529288],
    [-0.16512076, -0.02203333,  0.98602721, 0.00325563],
    [0, 0, 0, 1]],

    [[-0.99574492,  0.09210627,  0.00291379, 0.01370516],
    [0.09210627,  0.99374962,  0.06307244, -0.00033661],
    [0.00291379,  0.06307244, -0.9980047, -0.00937349],
    [0, 0, 0, 1]],

    [[0.98342639, -0.08683736,  0.15915968, 0.00686074],
    [-0.08683736, -0.99619813, -0.00696825, 0.14597624],
    [0.15915968, -0.00696825, -0.98722826, -0.0058531],
    [0, 0, 0, 1]],
])

BOXSMALL_SYMRT = np.array([
    [[-0.96453679,  0.06425706, -0.25600744, -0.02620367],
    [0.06425706, -0.88357032, -0.46386907, 0.09371875],
    [-0.25600744, -0.46386907,  0.84810712, 0.01989327],
    [0, 0, 0, 1]],

    [[-0.99520563, -0.09479641, -0.02407053, -0.02507379],
    [-0.09479641,  0.87435745,  0.47593349, -0.00690347],
    [-0.02407053,  0.47593349, -0.87915182, 0.02219357],
    [0, 0, 0, 1]],

    [[9.62337874e-01,  7.04781484e-03,  2.71764872e-01, -0.00712959],
    [7.04781484e-03, -9.99974687e-01, 9.76054390e-04, 0.08722881],
    [2.71764872e-01,  9.76054390e-04, -9.62363186e-01, 0.04921862],
    [0, 0, 0, 1]],
])

BOXTINY_SYMRT = np.array([
    [[-0.87436897, -0.01548803, 0.48501446, 0.00249667],
    [-0.01548803, -0.99809061, -0.0597935, 0.03865318],
    [ 0.48501446, -0.0597935, 0.87245958, 0.00058762],
    [0, 0, 0, 1]],

    [[-0.96583472, 0.25915495, -0.00141525, -0.00337465],
    [0.25915495, 0.9657761,  -0.01073512, 0.00043197],
    [-0.00141525, -0.01073512, -0.99994138, -0.00236562],
    [0, 0, 0, 1]],

    [[0.85836827, -0.20065837, -0.47216536, 0.00339908],
    [-0.20065837, -0.9783338,  0.05098232, 0.03856828],
    [-0.47216536,  0.05098232, -0.88003447, -0.00301231],
    [0, 0, 0, 1]],
])

PLASTICCONTAINER_SYMRT = np.array([
    [[-0.45609467, 0.87246605, 0.17544411, -0.0099864],
    [0.87246605, 0.39950277, 0.28142587, 0.00379416],
    [0.17544411, 0.28142587, -0.9434081, 0.01209146],
    [0, 0, 0, 1]],
])

STOOL_SYMRT = np.array([
    [[-0.49828222, 0.03609013, -0.86626343, -0.00025576],
    [0.065359,   0.99785389, 0.00397737, -0.00015834],
    [0.86454788, -0.05463626, -0.49957167, -0.00987547],
    [0, 0, 0, 1]],

    [[-0.45609467, 0.87246605, 0.17544411, -0.0099864],
    [0.87246605, 0.39950277, 0.28142587, 0.00379416],
    [0.17544411, 0.28142587, -0.9434081, 0.01209146],
    [0, 0, 0, 1]],
])

TABLESQUARE_SYMRT = np.array([
    [[0.89677757, -0.44008605,  0.0459811, -0.00829433],
    [-0.02196353,  0.05951582,  0.99798571, -0.01319974],
    [-0.44193619, -0.8959811,  0.04370661, -0.02216766],
    [0, 0, 0, 1]],

    [[0.79355514, -0.46204958, -0.39595508, -0.01094276],
    [-0.46204958, -0.88096836, 0.1020046, -0.03592617],
    [-0.39595508, 0.1020046, -0.91258678, -0.00764425],
    [0, 0, 0, 1]],

    [[0.89677757, -0.02196353, -0.44193619, -0.00264844],
    [-0.44008605, 0.05951582, -0.8959811, -0.02272643],
    [0.0459811,  0.99798571, 0.04370661, 0.01452341],
    [0, 0, 0, 1]],
])

TOOLBOX_SYMRT = np.array([
    [[-0.98277884, 0.18438843, 0.01210996, 0.01798474],
    [0.18438843, 0.97426309, 0.12966238, -0.00039969],
    [0.01210996, 0.12966238, -0.99148425, -0.01948967],
    [0, 0, 0, 1]],
])

# left multiply this matrix to make the mesh symatric about yz-axis plane
OBJECT_YZ_SYM_ROTMAT = {
    'backpack': np.array([
        [0.99493741, -0.08391616, -0.05529579],
        [ 0.09352531, 0.97451671,  0.20388769],
        [ 0.0367772, -0.20802705, 0.97743142]]),
    'basketball': np.array([
        [1, 0, 0], [0, 1, 0], [0, 0, 1]]),
    'boxlarge': np.array([
        [1, 0, 0], [0, 1, 0], [0, 0, 1]]),
    'boxlong': np.array([
        [0.36937853, -0.01193658, -0.92920235],
        [-0.21933134, -0.97278712, -0.0746926],
        [-0.90302451, 0.23139304, -0.36194475]]),
    'boxmedium': np.array([
        [0.99492398, -0.05183743, 0.08625055],
        [0.05221025, 0.99863397, -0.00207088],
        [-0.08602538, 0.00656353, 0.99627133]]),
    'boxsmall': np.array([
        [0.98785859, 0.03663071, 0.15097549],
        [-0.07304361, 0.96720517, 0.24326692],
        [-0.13711323, -0.25134111, 0.95813757]]),
    'boxtiny': np.array([
        [0.9603944, -0.12474431, -0.24916149],
        [-0.12807739, -0.99176006, 0.00285608],
        [-0.2474647,  0.02916899, -0.96845774]]),
    'chairblack': np.array([
        [1, 0, 0], [0, 1, 0], [0, 0, 1]]),
    'chairwood': np.array([
        [-0.16024307, -0.26172444, 0.95174706],
        [0.03192787, 0.96232808, 0.27000976],
        [-0.98656108, 0.07365445, -0.1458501]]),
    'keyboard': np.array([
        [-0.35796803, -0.05959167, 0.93183031],
        [0.2023049, -0.97920624, 0.01509519],
        [0.91155451, 0.19391743, 0.36258022]]),
    'monitor': np.array([
        [-0.24141867, 0.02770189, 0.97002558],
        [0.95463538, 0.18634219, 0.23226683],
        [-0.17432246, 0.98209429, -0.07143169]]),
    'plasticcontainer': np.array([
        [0.85327014, -0.47727699, 0.21008747],
        [0.25519701, 0.0308503, -0.96639678],
        [0.45475769, 0.87821121, 0.14812331]]),
    'stool': np.array([
        [1, 0, 0], [0, 1, 0], [0, 0, 1]]),
    'suitcase': np.array([
        [0.15643447, 0.12379017, -0.97990012],
        [0.,          0.9921147,   0.12533323],
        [0.98768834, -0.01960644,  0.15520093]]),
    'tablesmall': np.array([
        [0.,          0.12533323, -0.9921147],
        [0.,          0.9921147,   0.12533323],
        [1.,          0.,          0.]]),
    'tablesquare': np.array([
        [0.01705112, -0.53029861, 0.84763945],
        [0.28554881,  0.81503765, 0.50415822],
        [-0.95821247,  0.23344597,  0.16532344]]),
    'toolbox': np.array([
        [-0.2710324,  -0.05068319,  0.96123496],
        [0.04764554, 0.99668234,  0.06598649],
        [-0.96139032, 0.06368304, -0.26771838]]),
    'trashbin': np.array([
        [1, 0, 0], [0, 1, 0], [0, 0, 1]]),
    'yogaball': np.array([
        [1, 0, 0], [0, 1, 0], [0, 0, 1]]),
    'yogamat': np.array([
        [0.37164627, 0.9271288, -0.04807533],
        [-0.03767215, -0.03668112, -0.9986167],
        [-0.92760975, 0.37294327, 0.02129453]]),
}

OBJECT_FLIP_INDICES = {
    'backpack': [1, 0, 3, 2, 7, 6, 5, 4],
    'boxlarge': [1, 0, 3, 2, 7, 6, 5, 4],
    'boxlong': [1, 0, 3, 2, 7, 6, 5, 4],
    'boxmedium': [1, 0, 3, 2, 7, 6, 5, 4],
    'boxsmall': [1, 0, 3, 2, 7, 6, 5, 4],
    'boxtiny': [1, 0, 3, 2, 7, 6, 5, 4],
    'chairblack': [1, 0, 9, 8, 7, 6, 5, 4, 3, 2, 13, 14, 15, 10, 11, 12],
    'chairwood': [1, 0, 5, 4, 3, 2, 9, 8, 7, 6],
    'monitor': [1, 0, 3, 2, 4, 5, 7, 6],
    'plasticcontainer': [1, 0, 3, 2, 5, 4, 7, 6],
    'stool': [1, 0, 2, 4, 3, 5],
    'suitcase': [1, 0, 3, 2, 7, 6, 5, 4],
    'tablesmall': [1, 0, 3, 2, 4, 5, 7, 6, 9, 8],
    'tablesquare': [1, 0, 3, 2, 5, 4, 7, 6],
    'toolbox': [1, 0, 3, 2, 7, 6, 5, 4],
    'trashbin': [0, 1],
    'yogaball': [0, ],
    'yogamat': [0, 1],
}
OBJECT_KPS_PERM = {
    'backpack': [[0, 1, 2, 3, 4, 5, 6, 7]],
    'boxlarge': [[0, 1, 2, 3, 4, 5, 6, 7], [6, 5, 4, 7, 2, 1, 0, 3], [2, 3, 0, 1, 6, 7, 4, 5], [4, 7, 6, 5, 0, 3, 2, 1], [1, 6, 7, 2, 3, 0, 5, 4], [5, 0, 3, 4, 7, 6, 1, 2]],
    'boxlong': [[0, 1, 2, 3, 4, 5, 6, 7], [2, 3, 0, 1, 6, 7, 4, 5], [6, 5, 4, 7, 2, 1, 0, 3], [4, 7, 6, 5, 0, 3, 2, 1], [1, 2, 3, 0, 5, 6, 7, 4], [3, 0, 1, 2, 7, 4, 5, 6]],
    'boxmedium': [[0, 1, 2, 3, 4, 5, 6, 7], [2, 3, 0, 1, 6, 7, 4, 5], [6, 5, 4, 7, 2, 1, 0, 3], [4, 7, 6, 5, 0, 3, 2, 1], ],
    'boxsmall': [[0, 1, 2, 3, 4, 5, 6, 7], [2, 3, 0, 1, 6, 7, 4, 5], [6, 5, 4, 7, 2, 1, 0, 3], [4, 7, 6, 5, 0, 3, 2, 1]],
    'boxtiny': [[0, 1, 2, 3, 4, 5, 6, 7], [2, 3, 0, 1, 6, 7, 4, 5], [6, 5, 4, 7, 2, 1, 0, 3], [4, 7, 6, 5, 0, 3, 2, 1]],
    'chairblack': [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]],
    'chairwood': [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]],
    'monitor': [[0, 1, 2, 3, 4, 5, 6, 7]],
    'plasticcontainer': [[0, 1, 2, 3, 4, 5, 6, 7], [2, 3, 0, 1, 6, 7, 4, 5]],
    'stool': [[0, 1, 2, 3, 4, 5], [1, 2, 0, 4, 5, 3], [2, 0, 1, 5, 3, 4]],
    'suitcase': [[0, 1, 2, 3, 4, 5, 6, 7]],
    'tablesmall': [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9,]],
    'tablesquare': [[0, 1, 2, 3, 4, 5, 6, 7], [1, 2, 3, 0, 5, 6, 7, 4], [2, 3, 0, 1, 6, 7, 4, 5], [3, 0, 1, 2, 7, 4, 5, 6]],
    'toolbox': [[0, 1, 2, 3, 4, 5, 6, 7]],
    'trashbin': [[0, 1]],
    'yogaball': [[0, ]],
    'yogamat': [[0, 1], [1, 0]],
}
