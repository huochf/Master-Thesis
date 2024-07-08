#!/bin/bash
#SBATCH -J 1kps3
#SBATCH -p ShangHAI
#SBATCH -A wangjingya-ShangHAI
# SBATCH --cpus-per-task=10
#SBATCH --ntasks-per-node=16
#SBATCH -N 1
#SBATCH --gres=gpu:1
# #SBATCH --nodelist=ai_gpu13
#SBATCH -t 05-00:00:00
#SBATCH --output=/inspurfs/group/wangjingya/huochf/Thesis/logs/train_cflow_kps3d_behave1.log
#SBATCH --error=/inspurfs/group/wangjingya/huochf/Thesis/logs/train_cflow_kps3d_behave1.log

# export CUDA_HOME=/public/software/CUDA/cuda-10.2/

# source activate
# conda activate bigdet

cd /inspurfs/group/wangjingya/huochf/Thesis/

python -m hoi_recon.train_c_flow_kps3d --epoch 30 --object backpack --dropout_probability 0.5 --batch_size 16 --save_dir outputs/cflow_kps3d
python -m hoi_recon.train_c_flow_kps3d --epoch 30 --object boxlarge --dropout_probability 0.5 --batch_size 16 --save_dir outputs/cflow_kps3d
python -m hoi_recon.train_c_flow_kps3d --epoch 30 --object boxlong --dropout_probability 0.5 --batch_size 16 --save_dir outputs/cflow_kps3d

python -m hoi_recon.train_c_flow_kps3d --epoch 30 --object boxmedium --dropout_probability 0.5 --batch_size 16 --save_dir outputs/cflow_kps3d
python -m hoi_recon.train_c_flow_kps3d --epoch 30 --object boxsmall --dropout_probability 0.5 --batch_size 16 --save_dir outputs/cflow_kps3d
python -m hoi_recon.train_c_flow_kps3d --epoch 30 --object boxtiny --dropout_probability 0.5 --batch_size 16 --save_dir outputs/cflow_kps3d

# python -m hoi_recon.train_c_flow_kps3d --epoch 30 --object chairblack --dropout_probability 0.5 --batch_size 16 --save_dir outputs/cflow_kps3d
# python -m hoi_recon.train_c_flow_kps3d --epoch 30 --object chairwood --dropout_probability 0.5 --batch_size 16 --save_dir outputs/cflow_kps3d
# python -m hoi_recon.train_c_flow_kps3d --epoch 30 --object monitor --dropout_probability 0.5 --batch_size 16 --save_dir outputs/cflow_kps3d

# python -m hoi_recon.train_c_flow_kps3d --epoch 30 --object plasticcontainer --dropout_probability 0.5 --batch_size 16 --save_dir outputs/cflow_kps3d
# python -m hoi_recon.train_c_flow_kps3d --epoch 30 --object stool --dropout_probability 0.5 --batch_size 16 --save_dir outputs/cflow_kps3d
# python -m hoi_recon.train_c_flow_kps3d --epoch 30 --object suitcase --dropout_probability 0.5 --batch_size 16 --save_dir outputs/cflow_kps3d

# python -m hoi_recon.train_c_flow_kps3d --epoch 30 --object tablesmall --dropout_probability 0.5 --batch_size 16 --save_dir outputs/cflow_kps3d
# python -m hoi_recon.train_c_flow_kps3d --epoch 30 --object tablesquare --dropout_probability 0.5 --batch_size 16 --save_dir outputs/cflow_kps3d
# python -m hoi_recon.train_c_flow_kps3d --epoch 30 --object toolbox --dropout_probability 0.5 --batch_size 16 --save_dir outputs/cflow_kps3d

# python -m hoi_recon.train_c_flow_kps3d --epoch 30 --object trashbin --dropout_probability 0.5 --batch_size 16 --save_dir outputs/cflow_kps3d
# python -m hoi_recon.train_c_flow_kps3d --epoch 30 --object yogaball --dropout_probability 0.5 --batch_size 16 --save_dir outputs/cflow_kps3d
# python -m hoi_recon.train_c_flow_kps3d --epoch 30 --object yogamat --dropout_probability 0.5 --batch_size 16 --save_dir outputs/cflow_kps3d
# NVIDIAGeForceRTX2080Ti NVIDIATITANV
