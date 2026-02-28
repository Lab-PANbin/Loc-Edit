#!/bin/bash
#SBATCH -o rvt_all.%j.out
#SBATCH -p compute1
#SBATCH --qos=normal
#SBATCH -J rvt_all
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH -w node8

nvidia-smi
source activate mipNeRF

#scenes=(bottle cactus case cello_case cubes cup fengtiaoyushun flower_basket fruits '''hotel painting room''')
#scenes=(case cup bottle cactus cello_case cubes fengtiaoyushun flower_basket fruits hotel painting room)
#scenes=(bottle cactus case cello_case cubes cup fengtiaoyushun flower_basket fruits hotel painting room)
#scenes=(bottle cactus hotel)
scenes=(case)

for scene in ${scenes[*]}
do
    echo Ref-Visualizing ${scene} in train pose.
    startTime=$(date +%s)
    bash ./ref_vis_train/${scene}.sh >& rvt_${scene}.out
    endTime=$(date +%s)
    echo Rendering time: $[$endTime-$startTime]s.
done
