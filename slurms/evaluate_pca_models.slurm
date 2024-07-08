#!/bin/bash
#SBATCH -J pca0
#SBATCH -p critical
#SBATCH -A wangjingya-critical
# SBATCH --cpus-per-task=16
#SBATCH --ntasks-per-node=16
#SBATCH --mem-per-cpu=5937MB
#SBATCH -N 1
#SBATCH --gres=gpu:1
# #SBATCH --nodelist=ai_gpu13
#SBATCH -t 05-00:00:00
#SBATCH --output=/inspurfs/group/wangjingya/huochf/Thesis/logs/evaluate_pca_models0.log
#SBATCH --error=/inspurfs/group/wangjingya/huochf/Thesis/logs/evaluate_pca_models0.log

# export CUDA_HOME=/public/software/CUDA/cuda-10.2/

# source activate
# conda activate bigdet

cd /inspurfs/group/wangjingya/huochf/Thesis/
python -m hoi_recon.evaluate_pca_model --root_dir /storage/data/huochf/BEHAVE --smpl_anchor_num 4 --object_anchor_num 8 --pca_dim 128
python -m hoi_recon.evaluate_pca_model --root_dir /storage/data/huochf/BEHAVE --smpl_anchor_num 4 --object_anchor_num 8 --pca_dim 64
python -m hoi_recon.evaluate_pca_model --root_dir /storage/data/huochf/BEHAVE --smpl_anchor_num 4 --object_anchor_num 8 --pca_dim 32

# NVIDIAGeForceRTX2080Ti NVIDIATITANV
