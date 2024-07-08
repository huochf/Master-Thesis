#!/bin/bash
#SBATCH -J prohmr
#SBATCH -p ShangHAI
#SBATCH -A wangjingya-ShangHAI
# SBATCH --cpus-per-task=10
#SBATCH --ntasks-per-node=16
#SBATCH --mem-per-cpu=4890MB
#SBATCH -N 1
#SBATCH --gres=gpu:1
# #SBATCH --nodelist=ai_gpu13
#SBATCH -t 05-00:00:00
#SBATCH --output=/public/home/huochf/projects/3D_HOI/Thesis/logs/train_prohmr_behave.log
#SBATCH --error=/public/home/huochf/projects/3D_HOI/Thesis/logs/train_prohmr_behave.log

# export CUDA_HOME=/public/software/CUDA/cuda-10.2/

# source activate
# conda activate bigdet

cd /public/home/huochf/projects/3D_HOI/Thesis/externals
# python -m prohmr_custom.train --output_dir outputs/prohmr/behave_extend/ --checkpoint ./prohmr_custom/ProHMR/data/checkpoint.pt --cfg_file ./prohmr_custom/configs/prohmr_behave_extend.yaml
# python -m prohmr_custom.train --output_dir ../outputs/prohmr/behave_extend/ --checkpoint ./prohmr_custom/ProHMR/data/checkpoint.pt --cfg_file ./prohmr_custom/configs/prohmr_behave_extend.yaml
python -m prohmr_custom.train --output_dir ../outputs/prohmr/behave_extend/ --checkpoint ../outputs/prohmr/behave_extend/checkpoints/epoch=6-step=130000.ckpt --cfg_file ./prohmr_custom/configs/prohmr_behave_extend.yaml
# python -m prohmr_custom.train --output_dir ../outputs/prohmr/behave_extend/ --checkpoint ../outputs/prohmr/behave_extend/checkpoints/epoch=7-step=140000.ckpt --cfg_file ./prohmr_custom/configs/prohmr_behave_extend.yaml

# python -m prohmr_custom.train --output_dir ../outputs/prohmr/intercap/ --checkpoint ./prohmr_custom/ProHMR/data/checkpoint.pt --cfg_file ./prohmr_custom/configs/prohmr_intercap.yaml

# NVIDIAGeForceRTX2080Ti NVIDIATITANV
