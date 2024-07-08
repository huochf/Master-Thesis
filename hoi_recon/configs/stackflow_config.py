from yacs.config import CfgNode as CN

_C = CN(new_allowed=True)


_C.dataset = CN(new_allowed=True)
_C.dataset.name = 'BEHAVE-Extended'
_C.dataset.root_dir = '/storage/data/huochf/BEHAVE/'
_C.dataset.aug_ratio = 0.5
_C.dataset.init_cam_translation = [0, 0.75, 2]
_C.dataset.num_object = 20
_C.dataset.img_size = 256
_C.dataset.hoi_img_padding_ratio = 0.2
_C.dataset.aug_trans_factor = 0.1
_C.dataset.aug_rot_factor = 30
_C.dataset.aug_scale_factor = 0.3
_C.dataset.aug_color_scale = 0.2
_C.dataset.mean = [0.485, 0.456, 0.406]
_C.dataset.std = [0.229, 0.224, 0.225]
_C.dataset.change_bg_ratio = 0.5
_C.dataset.bg_dir = '/inspurfs/group/wangjingya/huochf/LineMOD/bg_images/VOC2012/JPEGImages/'

_C.model = CN(new_allowed=True)
_C.model.backbone = 'resnet'
_C.model.visual_feature_dim = 2048
_C.model.cam_head_dim = 2048
_C.model.smplh_dir = 'data/models/smplh'
_C.model.smplx_dir = 'data/models/smplx'
_C.model.smplh_regressor = 'data/models/smplh/J_regressor_body25_smplh.txt'
_C.model.smplx_regressor = 'data/models/smplx/J_regressor_body25_smplx.txt'


_C.model.patch_size = 32
_C.model.dim = 1024
_C.model.depth = 6
_C.model.heads = 16
_C.model.mlp_dim = 2048
_C.model.dropout = 0.1
_C.model.emb_dropout = 0.1

_C.model.smplflow = CN(new_allowed=True)
_C.model.smplflow.hidden_dim = 1024
_C.model.smplflow.num_layers = 4
_C.model.smplflow.num_blocks_per_layer = 2

_C.model.offsetflow = CN(new_allowed=True)
_C.model.offsetflow.hidden_dim = 1024
_C.model.offsetflow.num_layers = 4
_C.model.offsetflow.num_blocks_per_layer = 2
_C.model.smpl_anchor_num = 32 # 16 # 8 # 32
_C.model.object_anchor_num = 64 # 32 # 16 # 64
_C.model.pca_dim = 64

_C.model.offset = CN(new_allowed=True)
_C.model.offset.latent_dim = 64

_C.train = CN(new_allowed=True)
_C.train.num_samples = 2
_C.train.batch_size = 32
_C.train.num_workers = 8
_C.train.lr = 1e-4
_C.train.weight_decay = 1e-4
_C.train.resume = 'outputs/stackflow_n32_d64/latest_checkpoint.pth'
_C.train.max_epoch = 15
_C.train.drop_lr_at = 7
_C.train.log_interval = 10
_C.train.output_dir = 'outputs/stackflow_n32_d64'
_C.train.trans_begin_epoch = 7

_C.eval = CN(new_allowed=True)
_C.eval.batch_size = 32
_C.eval.num_workers = 8
_C.eval.checkpoint = ''
_C.eval.output_dir = 'outputs/stackflow_n32_d64'

_C.eval.post_optim = False
_C.eval.optim_lr = 4e-2
_C.eval.optim_iters = 2
_C.eval.optim_steps = 300
_C.eval.disagreement_scale = 0.4


def load_config():
    cfg = _C.clone()
    return cfg
