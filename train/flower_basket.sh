#!/bin/bash
#SBATCH -o mipnerfflower_basket.out
#SBATCH -p compute
#SBATCH --qos=normal
#SBATCH -J flower_basket
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1 

source activate mipNeRF
nvidia-smi
SCENE=flower_basket
echo "Train: $SCENE"

FACTOR=4
echo Factor: ${FACTOR}.
python -u train.py --dataset_name llff --scene ${SCENE} --save_every 1000 --log_dir log --max_steps 200000 --factor ${FACTOR}
