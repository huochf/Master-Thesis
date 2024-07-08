#!/bin/bash
#SBATCH -J no_PCA
#SBATCH -p critical
#SBATCH -A wangjingya-critical
# SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=16
#SBATCH --mem-per-cpu=2937MB
#SBATCH -N 1
#SBATCH --gres=gpu:1
# SBATCH --nodelist=ai_gpu11
#SBATCH -t 05-00:00:00
#SBATCH --output=/inspurfs/group/wangjingya/huochf/Thesis/logs/learn_object_inter_shape_2k_alpha_10000_no_PCA.log
#SBATCH --error=/inspurfs/group/wangjingya/huochf/Thesis/logs/learn_object_inter_shape_2k_alpha_10000_no_PCA.log

# export CUDA_HOME=/public/software/CUDA/cuda-10.2/

# source activate
# conda activate bigdet

cd /inspurfs/group/wangjingya/huochf/Thesis/
python -m hoi_recon.learn_object_inter_shape

# NVIDIAGeForceRTX2080Ti NVIDIATITANV NVIDIAA40
