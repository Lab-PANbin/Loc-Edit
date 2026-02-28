#!/bin/bash
#SBATCH -o rv_cello_case.out
#SBATCH -J rv_cello_case
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -p compute1
#SBATCH --gres=gpu:1

source activate mipNeRF
SCENE=cello_case

FACTOR=4
echo Factor: ${FACTOR}.
python ref_visualize.py --dataset_name llff --scene ${SCENE} --visualize_depth --visualize_normals --train_ad_nerf --train_ad_nerf  --ref_scene ref_${SCENE} --model_weight_path ref_log/ref_${SCENE}/model.pt --ref_log_dir ref_log --chunks 2048 --topk_points 32 --ad_hidden 256 --matching_chunck 256 --factor ${FACTOR}