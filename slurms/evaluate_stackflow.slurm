#!/bin/bash
#SBATCH -J d64
#SBATCH -p critical
#SBATCH -A wangjingya-critical
# SBATCH --cpus-per-task=10
#SBATCH --ntasks-per-node=10
#SBATCH --mem-per-cpu=4890MB
#SBATCH -N 1
#SBATCH --gres=gpu:1
# #SBATCH --nodelist=sist-a40-02
#SBATCH -t 05-00:00:00
#SBATCH --output=/inspurfs/group/wangjingya/huochf/Thesis/logs/evaluate_stackflow_optim_d64.log
#SBATCH --error=/inspurfs/group/wangjingya/huochf/Thesis/logs/evaluate_stackflow_optim_d64.log

# export CUDA_HOME=/public/software/CUDA/cuda-10.2/

# source activate
# conda activate bigdet

cd /inspurfs/group/wangjingya/huochf/Thesis/

python -m hoi_recon.evaluate_stackflow 