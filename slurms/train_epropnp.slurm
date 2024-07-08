#!/bin/bash
#SBATCH -J epro_behave
#SBATCH -p ShangHAI
#SBATCH -A wangjingya-ShangHAI
# SBATCH --cpus-per-task=10
#SBATCH --ntasks-per-node=10
#SBATCH -N 1
#SBATCH --gres=gpu:1
# #SBATCH --nodelist=ai_gpu13
#SBATCH -t 05-00:00:00
#SBATCH --output=/public/home/huochf/projects/3D_HOI/Thesis/logs/train_propnp_behave.log
#SBATCH --error=/public/home/huochf/projects/3D_HOI/Thesis/logs/train_propnp_behave.log

# export CUDA_HOME=/public/software/CUDA/cuda-10.2/

# source activate
# conda activate bigdet

cd /public/home/huochf/projects/3D_HOI/Thesis/externals
python ./EPro-PnP/train_epro_pnp.py --behave_extend --root_dir /storage/data/huochf/BEHAVE

# NVIDIAGeForceRTX2080Ti NVIDIATITANV
