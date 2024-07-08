#!/bin/bash
#SBATCH -J knn3
#SBATCH -p critical
#SBATCH -A wangjingya-critical
# SBATCH --cpus-per-task=10
#SBATCH --ntasks-per-node=10
#SBATCH -N 1
#SBATCH --gres=gpu:1
# #SBATCH --nodelist=ai_gpu13
#SBATCH -t 05-00:00:00
#SBATCH --output=/inspurfs/group/wangjingya/huochf/Thesis/logs/KNN_grouping3.log
#SBATCH --error=/inspurfs/group/wangjingya/huochf/Thesis/logs/KNN_grouping3.log

# export CUDA_HOME=/public/software/CUDA/cuda-10.2/

# source activate
# conda activate bigdet

cd /inspurfs/group/wangjingya/huochf/Thesis/
# 1
# python -m hoi_recon.KNN_grouping --object backpack
# python -m hoi_recon.KNN_grouping --object boxlarge
# python -m hoi_recon.KNN_grouping --object boxlong

# 2
# python -m hoi_recon.KNN_grouping --object boxmedium
# python -m hoi_recon.KNN_grouping --object boxsmall
# python -m hoi_recon.KNN_grouping --object boxtiny

# 3
python -m hoi_recon.KNN_grouping --object chairblack
python -m hoi_recon.KNN_grouping --object chairwood
python -m hoi_recon.KNN_grouping --object monitor

# 4
# python -m hoi_recon.KNN_grouping --object plasticcontainer
# python -m hoi_recon.KNN_grouping --object stool
# python -m hoi_recon.KNN_grouping --object suitcase

# 5
# python -m hoi_recon.KNN_grouping --object tablesmall
# python -m hoi_recon.KNN_grouping --object tablesquare
# python -m hoi_recon.KNN_grouping --object toolbox

# 6
# python -m hoi_recon.KNN_grouping --object trashbin
# python -m hoi_recon.KNN_grouping --object yogaball
# python -m hoi_recon.KNN_grouping --object yogamat
# NVIDIAGeForceRTX2080Ti NVIDIATITANV
