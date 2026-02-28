#!/bin/bash
#SBATCH -o vt_all.%j.out
#SBATCH -p compute1
#SBATCH --qos=normal
#SBATCH -J vis_train_all
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1 
#SBATCH -w node8

source activate mipNeRF

#scenes=(bottle cello_case cubes fruits)
scenes=(bottle cactus case hotel)

for scene in ${scenes[*]}
do
    echo Visualizing ${scene} in train pose.
    startTime=$(date +%s)
    bash ./vis_train/${scene}.sh >& vt_${scene}.out
    endTime=$(date +%s)
    echo Rendering time: $[$endTime-$startTime]s.
done
