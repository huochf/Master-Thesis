#!/bin/bash
#SBATCH -J d64_stackflow
#SBATCH -p ShangHAI
#SBATCH -A wangjingya-ShangHAI
# SBATCH --cpus-per-task=16
#SBATCH --ntasks-per-node=16
#SBATCH --mem-per-cpu=5937MB
#SBATCH -N 1
#SBATCH --gres=gpu:1
# #SBATCH --nodelist=ai_gpu13
#SBATCH -t 05-00:00:00
#SBATCH --output=/inspurfs/group/wangjingya/huochf/Thesis/logs/train_stackflow_d64.log
#SBATCH --error=/inspurfs/group/wangjingya/huochf/Thesis/logs/train_stackflow_d64.log

# export CUDA_HOME=/public/software/CUDA/cuda-10.2/

# source activate
# conda activate bigdet

cd /inspurfs/group/wangjingya/huochf/Thesis/
python -m hoi_recon.train_stackflow

# NVIDIAGeForceRTX2080Ti NVIDIATITANV
