#!/bin/bash
#SBATCH -J no_PCA
#SBATCH -p ShangHAI
#SBATCH -A wangjingya-ShangHAI
# SBATCH --cpus-per-task=16
#SBATCH --ntasks-per-node=8
#SBATCH --mem-per-cpu=4937MB
#SBATCH -N 1
#SBATCH --gres=gpu:1
# #SBATCH --nodelist=ai_gpu11
#SBATCH -t 02-00:00:00
#SBATCH --output=/inspurfs/group/wangjingya/huochf/Thesis/logs/train_stackflow_chairs_2k_alpha_10000_no_PCA.log
#SBATCH --error=/inspurfs/group/wangjingya/huochf/Thesis/logs/train_stackflow_chairs_2k_alpha_10000_no_PCA.log

# export CUDA_HOME=/public/software/CUDA/cuda-10.2/

# source activate
# conda activate bigdet

cd /inspurfs/group/wangjingya/huochf/Thesis/
python -m hoi_recon.train_stackflow_chairs

# NVIDIAGeForceRTX2080Ti NVIDIATITANV
