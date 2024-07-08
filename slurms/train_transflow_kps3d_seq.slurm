#!/bin/bash
#SBATCH -J 3transflow
#SBATCH -p critical
#SBATCH -A wangjingya-critical
# SBATCH --cpus-per-task=10
#SBATCH --ntasks-per-node=10
#SBATCH --mem-per-cpu=4890MB
#SBATCH -N 1
#SBATCH --gres=gpu:1
# #SBATCH --nodelist=sist-a40-02
#SBATCH -t 05-00:00:00
#SBATCH --output=/inspurfs/group/wangjingya/huochf/Thesis/logs/train_transflow_kps3d_seq3.log
#SBATCH --error=/inspurfs/group/wangjingya/huochf/Thesis/logs/train_transflow_kps3d_seq3.log

# export CUDA_HOME=/public/software/CUDA/cuda-10.2/

# source activate
# conda activate bigdet

cd /inspurfs/group/wangjingya/huochf/Thesis/

# 1
# python -m hoi_recon.train_transflow_kps3d_seq --object backpack --epoch 400 --batch_size 32 --head_dim 256 --flow_width 512 --dropout_probability 0.5 --save_dir ./outputs/transflow_kps3d_seq
# python -m hoi_recon.train_transflow_kps3d_seq --object boxlarge --epoch 400 --batch_size 32 --head_dim 256 --flow_width 512 --dropout_probability 0.5 --save_dir ./outputs/transflow_kps3d_seq

# 2
# python -m hoi_recon.train_transflow_kps3d_seq --object boxlong --epoch 400 --batch_size 32 --head_dim 256 --flow_width 512 --dropout_probability 0.5 --save_dir ./outputs/transflow_kps3d_seq
# python -m hoi_recon.train_transflow_kps3d_seq --object boxmedium --epoch 400 --batch_size 32 --head_dim 256 --flow_width 512 --dropout_probability 0.5 --save_dir ./outputs/transflow_kps3d_seq

# 3
python -m hoi_recon.train_transflow_kps3d_seq --object boxsmall --epoch 400 --batch_size 32 --head_dim 256 --flow_width 512 --dropout_probability 0.5 --save_dir ./outputs/transflow_kps3d_seq
python -m hoi_recon.train_transflow_kps3d_seq --object boxtiny --epoch 400 --batch_size 32 --head_dim 256 --flow_width 512 --dropout_probability 0.5 --save_dir ./outputs/transflow_kps3d_seq

# 4
# python -m hoi_recon.train_transflow_kps3d_seq --object chairblack --epoch 400 --batch_size 32 --head_dim 256 --flow_width 512 --dropout_probability 0.5 --save_dir ./outputs/transflow_kps3d_seq
# python -m hoi_recon.train_transflow_kps3d_seq --object chairwood --epoch 400 --batch_size 32 --head_dim 256 --flow_width 512 --dropout_probability 0.5 --save_dir ./outputs/transflow_kps3d_seq

# 5
# python -m hoi_recon.train_transflow_kps3d_seq --object monitor --epoch 400 --batch_size 32 --head_dim 256 --flow_width 512 --dropout_probability 0.5 --save_dir ./outputs/transflow_kps3d_seq
# python -m hoi_recon.train_transflow_kps3d_seq --object plasticcontainer --epoch 400 --batch_size 32 --head_dim 256 --flow_width 512 --dropout_probability 0.5 --save_dir ./outputs/transflow_kps3d_seq

# 6
# python -m hoi_recon.train_transflow_kps3d_seq --object stool --epoch 400 --batch_size 32 --head_dim 256 --flow_width 512 --dropout_probability 0.5 --save_dir ./outputs/transflow_kps3d_seq
# python -m hoi_recon.train_transflow_kps3d_seq --object suitcase --epoch 400 --batch_size 32 --head_dim 256 --flow_width 512 --dropout_probability 0.5 --save_dir ./outputs/transflow_kps3d_seq

# 7
# python -m hoi_recon.train_transflow_kps3d_seq --object tablesmall --epoch 400 --batch_size 32 --head_dim 256 --flow_width 512 --dropout_probability 0.5 --save_dir ./outputs/transflow_kps3d_seq
# python -m hoi_recon.train_transflow_kps3d_seq --object tablesquare --epoch 400 --batch_size 32 --head_dim 256 --flow_width 512 --dropout_probability 0.5 --save_dir ./outputs/transflow_kps3d_seq

# 8
# python -m hoi_recon.train_transflow_kps3d_seq --object toolbox --epoch 400 --batch_size 32 --head_dim 256 --flow_width 512 --dropout_probability 0.5 --save_dir ./outputs/transflow_kps3d_seq
# python -m hoi_recon.train_transflow_kps3d_seq --object trashbin --epoch 400 --batch_size 32 --head_dim 256 --flow_width 512 --dropout_probability 0.5 --save_dir ./outputs/transflow_kps3d_seq

# 9
# python -m hoi_recon.train_transflow_kps3d_seq --object yogaball --epoch 400 --batch_size 32 --head_dim 256 --flow_width 512 --dropout_probability 0.5 --save_dir ./outputs/transflow_kps3d_seq
# python -m hoi_recon.train_transflow_kps3d_seq --object yogamat --epoch 400 --batch_size 32 --head_dim 256 --flow_width 512 --dropout_probability 0.5 --save_dir ./outputs/transflow_kps3d_seq

# NVIDIAGeForceRTX2080Ti NVIDIATITANV
