#!/bin/bash
#SBATCH -J eval_transflow
#SBATCH -p ShangHAI
#SBATCH -A wangjingya-ShangHAI
# SBATCH --cpus-per-task=16
#SBATCH --ntasks-per-node=16
#SBATCH --mem-per-cpu=5937MB
#SBATCH -N 1
#SBATCH --gres=gpu:1
# #SBATCH --nodelist=ai_gpu13
#SBATCH -t 05-00:00:00
#SBATCH --output=/inspurfs/group/wangjingya/huochf/Thesis/logs/evaluate_transflow_kps3d.log
#SBATCH --error=/inspurfs/group/wangjingya/huochf/Thesis/logs/evaluate_transflow_kps3d.log

# export CUDA_HOME=/public/software/CUDA/cuda-10.2/

# source activate
# conda activate bigdet

cd /inspurfs/group/wangjingya/huochf/Thesis/
python -m hoi_recon.evaluate_c_flow_kps3d --object backpack
python -m hoi_recon.evaluate_c_flow_kps3d --object boxlarge
python -m hoi_recon.evaluate_c_flow_kps3d --object boxlong

python -m hoi_recon.evaluate_c_flow_kps3d --object boxmedium
python -m hoi_recon.evaluate_c_flow_kps3d --object boxsmall
python -m hoi_recon.evaluate_c_flow_kps3d --object boxtiny

python -m hoi_recon.evaluate_c_flow_kps3d --object chairblack
python -m hoi_recon.evaluate_c_flow_kps3d --object chairwood
python -m hoi_recon.evaluate_c_flow_kps3d --object monitor

python -m hoi_recon.evaluate_c_flow_kps3d --object plasticcontainer
python -m hoi_recon.evaluate_c_flow_kps3d --object stool
python -m hoi_recon.evaluate_c_flow_kps3d --object suitcase

python -m hoi_recon.evaluate_c_flow_kps3d --object tablesmall
python -m hoi_recon.evaluate_c_flow_kps3d --object tablesquare
python -m hoi_recon.evaluate_c_flow_kps3d --object toolbox

python -m hoi_recon.evaluate_c_flow_kps3d --object trashbin
python -m hoi_recon.evaluate_c_flow_kps3d --object yogaball
python -m hoi_recon.evaluate_c_flow_kps3d --object yogamat
# NVIDIAGeForceRTX2080Ti NVIDIATITANV
