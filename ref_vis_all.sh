#!/bin/bash
#SBATCH -o rv_all.%j.out
#SBATCH -p compute
#SBATCH --qos=normal
#SBATCH -J rv_all
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1 

source activate mipNeRF

#scenes=(bottle cactus case cello_case cubes cup fengtiaoyushun flower_basket fruits '''hotel painting room''')
#scenes=(cactus case cello_case cubes cup fengtiaoyushun flower_basket fruits)
scenes=(hotel painting room cubes)

for scene in ${scenes[*]}
do
    echo Ref-Visualizing ${scene} in test pose.
    startTime=$(date +%s)
    bash ./ref_vis/${scene}.sh >& rv_${scene}.out
    endTime=$(date +%s)
    echo Rendering time: $[$endTime-$startTime]s.
done
