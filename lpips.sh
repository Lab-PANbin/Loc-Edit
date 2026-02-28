#!/bin/bash
#SBATCH -o lpips.out
#SBATCH -J lpips
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -p compute
#SBATCH --gres=gpu:1

source activate mipNeRF
scenes=(bottle cactus case cello_case cubes cup fengtiaoyushun flower_basket fruits hotel painting room)
methods=(ad-mip refnpr snerf arf)

for method in ${methods[*]}
do
    for scene in ${scenes[*]}
    do
        echo Avg LPIPS of ${scene}, rendered by ${method}:
        python get_lpips.py --scene $scene --method $method --train
    done
done
