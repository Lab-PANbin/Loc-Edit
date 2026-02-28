#!/bin/bash
#SBATCH -o vt_painting.out
#SBATCH -p compute1
#SBATCH --qos=normal
#SBATCH -J vt_painting
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1 

source activate mipNeRF
nvidia-smi
SCENE=painting

FACTOR=1

echo Factor: ${FACTOR}.
python -u visualize_train_pose.py --dataset_name llff --scene ${SCENE} --visualize_depth --visualize_normals --model_weight_path ./log/${SCENE}/model.pt --log_dir log --factor ${FACTOR}