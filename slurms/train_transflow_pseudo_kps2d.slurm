#!/bin/bash
#SBATCH -J cflow_behave
#SBATCH -p ShangHAI
#SBATCH -A wangjingya-ShangHAI
# SBATCH --cpus-per-task=10
#SBATCH --ntasks-per-node=10
#SBATCH --mem-per-cpu=4890MB
#SBATCH -N 1
#SBATCH --gres=gpu:1
# #SBATCH --nodelist=ai_gpu13
#SBATCH -t 05-00:00:00
#SBATCH --output=/inspurfs/group/wangjingya/huochf/Thesis/logs/train_transflow_pseudo_kps2d_seq5_.log
#SBATCH --error=/inspurfs/group/wangjingya/huochf/Thesis/logs/train_transflow_pseudo_kps2d_seq5_.log

# export CUDA_HOME=/public/software/CUDA/cuda-10.2/

# source activate
# conda activate bigdet

cd /inspurfs/group/wangjingya/huochf/Thesis/
# 1
# python -m hoi_recon.train_transflow_pseudo_kps2d_seq --epoch 30 --object backpack --dropout_probability 0.5 --batch_size 16 --save_dir ./outputs/transflow_gt_kps2d_seq
# python -m hoi_recon.train_transflow_pseudo_kps2d_seq --epoch 30 --object boxlarge --dropout_probability 0.5 --batch_size 16 --save_dir ./outputs/transflow_gt_kps2d_seq
# python -m hoi_recon.train_transflow_pseudo_kps2d_seq --epoch 30 --object boxlong --dropout_probability 0.5 --batch_size 16 --save_dir ./outputs/transflow_gt_kps2d_seq

# 2
# python -m hoi_recon.train_transflow_pseudo_kps2d_seq --epoch 20 --object boxmedium --dropout_probability 0.5 --batch_size 16 --save_dir ./outputs/transflow_gt_kps2d_seq
# python -m hoi_recon.train_transflow_pseudo_kps2d_seq --epoch 20 --object boxsmall --dropout_probability 0.5 --batch_size 16 --save_dir ./outputs/transflow_gt_kps2d_seq
# python -m hoi_recon.train_transflow_pseudo_kps2d_seq --epoch 20 --object boxtiny --dropout_probability 0.5 --batch_size 16 --save_dir ./outputs/transflow_gt_kps2d_seq

# 3
# python -m hoi_recon.train_transflow_pseudo_kps2d_seq --epoch 20 --object chairblack --dropout_probability 0.5 --batch_size 16 --save_dir ./outputs/transflow_pseudo_kps2d_seq
# python -m hoi_recon.train_transflow_pseudo_kps2d_seq --epoch 20 --object chairwood --dropout_probability 0.5 --batch_size 16 --save_dir ./outputs/transflow_pseudo_kps2d_seq
# python -m hoi_recon.train_transflow_pseudo_kps2d_seq --epoch 20 --object monitor --dropout_probability 0.5 --batch_size 16 --save_dir ./outputs/transflow_gt_kps2d_seq

# 4
# python -m hoi_recon.train_transflow_pseudo_kps2d_seq --epoch 30 --object plasticcontainer --dropout_probability 0.5 --batch_size 16 --save_dir ./outputs/transflow_gt_kps2d_seq
# python -m hoi_recon.train_transflow_pseudo_kps2d_seq --epoch 30 --object stool --dropout_probability 0.5 --batch_size 16 --save_dir ./outputs/transflow_gt_kps2d_seq
# python -m hoi_recon.train_transflow_pseudo_kps2d_seq --epoch 30 --object suitcase --dropout_probability 0.5 --batch_size 16 --save_dir ./outputs/transflow_gt_kps2d_seq

# # 5
# python -m hoi_recon.train_transflow_pseudo_kps2d_seq --epoch 30 --object tablesmall --dropout_probability 0.5 --batch_size 16 --save_dir ./outputs/transflow_pseudo_kps2d_seq
python -m hoi_recon.train_transflow_pseudo_kps2d_seq --epoch 30 --object tablesquare --dropout_probability 0.5 --batch_size 16 --save_dir ./outputs/transflow_pseudo_kps2d_seq
# python -m hoi_recon.train_transflow_pseudo_kps2d_seq --epoch 30 --object toolbox --dropout_probability 0.5 --batch_size 16 --save_dir ./outputs/transflow_pseudo_kps2d_seq

# # 6
# python -m hoi_recon.train_transflow_pseudo_kps2d_seq --epoch 30 --object trashbin --dropout_probability 0.5 --batch_size 16 --save_dir ./outputs/transflow_pseudo_kps2d_seq
# python -m hoi_recon.train_transflow_pseudo_kps2d_seq --epoch 30 --object yogaball --dropout_probability 0.5 --batch_size 16 --save_dir ./outputs/transflow_pseudo_kps2d_seq
# python -m hoi_recon.train_transflow_pseudo_kps2d_seq --epoch 30 --object yogamat --dropout_probability 0.5 --batch_size 16 --save_dir ./outputs/transflow_pseudo_kps2d_seq
# NVIDIAGeForceRTX2080Ti NVIDIATITANV
