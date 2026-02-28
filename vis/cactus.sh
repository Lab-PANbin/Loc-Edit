#!/bin/bash
#SBATCH -o v_cactus.out
#SBATCH -p compute
#SBATCH --qos=normal
#SBATCH -J v_cactus
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1 

source activate mipNeRF
nvidia-smi
SCENE=cactus

FACTOR=4

echo Factor: ${FACTOR}.
python -u visualize.py --dataset_name llff --scene ${SCENE} --visualize_depth --visualize_normals --model_weight_path ./log/${SCENE}/model.pt --log_dir log --factor ${FACTOR}