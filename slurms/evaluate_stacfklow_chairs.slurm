#!/bin/bash
#SBATCH -J no_smooth
#SBATCH -p ShangHAI
#SBATCH -A wangjingya-ShangHAI
# SBATCH --cpus-per-task=10
#SBATCH --ntasks-per-node=10
#SBATCH --mem-per-cpu=2890MB
#SBATCH -N 1
# #SBATCH --nodelist=sist-a40-02
#SBATCH -t 01-00:00:00
#SBATCH --output=/inspurfs/group/wangjingya/huochf/Thesis/logs/evaluate_stackflow_chairs_2k_alpha_1000_no_smooth.log
#SBATCH --error=/inspurfs/group/wangjingya/huochf/Thesis/logs/evaluate_stackflow_chairs_2k_alpha_1000_no_smooth.log

# export CUDA_HOME=/public/software/CUDA/cuda-10.2/

# source activate
# conda activate bigdet

cd /inspurfs/group/wangjingya/huochf/Thesis/

python -m hoi_recon.evaluate_stackflow_on_chairs