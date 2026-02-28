#!/bin/bash
#SBATCH -o mipnerffengtiaoyushun.out
#SBATCH -p compute
#SBATCH --qos=normal
#SBATCH -J fengtiaoyushun
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1 

source activate mipNeRF
nvidia-smi
SCENE=fengtiaoyushun
echo "Train: $SCENE"

FACTOR=1
echo Factor: ${FACTOR}.
python -u train.py --dataset_name llff --scene ${SCENE} --save_every 1000 --log_dir log --max_steps 200000 --factor ${FACTOR}
