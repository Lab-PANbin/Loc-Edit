#!/bin/bash
#SBATCH -o mipnerfflowerpot.out
#SBATCH -p compute
#SBATCH --qos=normal
#SBATCH -J flowerpot
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1 

source activate mipNeRF
nvidia-smi
SCENE=flowerpot
FACTOR=4
echo "Train: $SCENE"

echo Factor: ${FACTOR}.
python -u train.py --dataset_name llff --scene ${SCENE} --save_every 1000 --log_dir log --max_steps 50000 --factor ${FACTOR} --continue_training
