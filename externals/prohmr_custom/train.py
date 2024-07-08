# adapted from prohmr.train.train_prohmr.py
import os
import argparse

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from prohmr.configs import get_config, dataset_config
from prohmr.datasets import ProHMRDataModule

from prohmr_custom.model import ProHMR
#from prohmr_custom.datasets.behave_smpl_dataset import BEHAVESMPLDataset
from prohmr_custom.datasets.behave_extend_smpl_dataset import BEHAVEExtendSMPLDataset
from prohmr_custom.datasets.intercap_smpl_dataset import InterCapSMPLDataset

parser = argparse.ArgumentParser(description="ProHMR training code")
parser.add_argument('--output_dir', type=str, help='Directory to save logs and checkpoints')
parser.add_argument('--checkpoint', type=str, help='Directory to save logs and checkpoints')
parser.add_argument('--cfg_file', default='prohmr_custom/configs/prohmr_behave.yaml', type=str, help='Directory to save logs and checkpoints')
args = parser.parse_args()

cfg = get_config(args.cfg_file)

model = ProHMR.load_from_checkpoint(args.checkpoint, strict=False, cfg=cfg)

if cfg.DATASETS.CONFIG.NAME == 'BEHAVE':
    data_module = BEHAVESMPLDataset(cfg)
elif cfg.DATASETS.CONFIG.NAME == 'BEHAVE-Extend':
    data_module = BEHAVEExtendSMPLDataset(cfg)
elif cfg.DATASETS.CONFIG.NAME == 'InterCap':
    data_module = InterCapSMPLDataset(cfg)

logger = TensorBoardLogger(os.path.join(args.output_dir, 'tensorboard'), name='', version='', default_hp_metric=False)

checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath=os.path.join(args.output_dir, 'checkpoints_cam_t'), every_n_train_steps=cfg.GENERAL.CHECKPOINT_STEPS)
trainer = pl.Trainer(default_root_dir=args.output_dir,
                     logger=logger,
                     limit_val_batches=1,
                     num_sanity_val_steps=0,
                     log_every_n_steps=cfg.GENERAL.LOG_STEPS,
                     val_check_interval=cfg.GENERAL.VAL_STEPS,
                     precision=16,
                     max_steps=cfg.GENERAL.TOTAL_STEPS,
                     callbacks=[checkpoint_callback])
trainer.fit(model, datamodule=data_module)
