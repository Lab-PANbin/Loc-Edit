#!/bin/bash
#SBATCH -o rt_painting.out
#SBATCH -p compute
#SBATCH --qos=normal
#SBATCH -J rt_painting
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1 

source activate mipNeRF
nvidia-smi
SCENE=painting

FACTOR=1

echo Factor: ${FACTOR}.
ori=`ls -l ./data/nerf_llff_data/${SCENE}/images | grep "^-" | wc -l`
python -u train_adjust_model.py --dataset_name ad_llff --scene ${SCENE} --batch_size 256 --train_ad_nerf --train_ad_nerf  --ref_scene ref_${SCENE} --base_MipNeRF_path ./log/${SCENE}/model.pt --ref_log_dir ref_log --ad_max_steps 25000 --max_steps 25000 --render_two_side --ad_hidden 256 --ori_image_num ${ori} --factor ${FACTOR}