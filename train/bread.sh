#!/bin/bash
#SBATCH -o mipnerfbread.out
#SBATCH -p compute1
#SBATCH --qos=normal
#SBATCH -J bread
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1 

source activate mipNeRF
nvidia-smi
SCENE=bread
FACTOR=4
echo "Train: $SCENE"

echo Factor: ${FACTOR}.
python -u train.py --dataset_name llff --scene ${SCENE} --save_every 1000 --log_dir log --max_steps 200000 --factor ${FACTOR}
