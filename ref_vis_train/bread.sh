#!/bin/bash
#SBATCH -o rvt_bread.out
#SBATCH -J rvt_bread
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -p compute1
#SBATCH --gres=gpu:1

source activate mipNeRF
SCENE=bread

FACTOR=4

echo Factor: ${FACTOR}.
ori=`ls -l ./data/nerf_llff_data/${SCENE}/images | grep "^-" | wc -l`
python ref_visualize_train_pose.py --dataset_name llff --scene ${SCENE} --visualize_depth --visualize_normals --train_ad_nerf --train_ad_nerf  --ref_scene ref_${SCENE} --model_weight_path ref_log/ref_${SCENE}/model.pt --ref_log_dir ref_log --chunks 2048 --topk_points 32 --ad_hidden 256 --matching_chunck 256 --factor ${FACTOR}