#!/bin/bash
#SBATCH -o v_all.out
#SBATCH -p compute
#SBATCH --qos=normal
#SBATCH -J vis_all
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1 

source activate mipNeRF

#scenes=(cup fengtiaoyushun flower_basket)
scenes=(fengtiaoyushun hotel painting room)

for scene in ${scenes[*]}
do
    echo Visualizing ${scene} in test pose.
    bash ./vis/${scene}.sh >& v_${scene}.out
done


scenes=(case cup fengtiaoyushun hotel painting room)

for scene in ${scenes[*]}
do
    echo Visualizing ${scene} in train pose.
    bash ./vis_train/${scene}.sh >& vt_${scene}.out
done

