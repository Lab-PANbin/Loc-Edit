#!/bin/bash
#SBATCH -o t_test.%j.out
#SBATCH -J t_test
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -p compute
#SBATCH --gres=gpu:1


#scenes=(bottle cactus case cello_case cubes cup fengtiaoyushun flower_basket fruits '''hotel painting room''')
#scenes=(bottle cactus case cello_case cubes cup fengtiaoyushun flower_basket fruits)
#scenes=(hotel painting room)
#scenes=(cactus)

#scenes=(bottle cactus case cello_case cubes cup fengtiaoyushun flower_basket fruits hotel painting room)
scenes=(fruits fengtiaoyushun room)

for scene in ${scenes[*]}
do
    source activate mipNeRF
    cd ../ad-mip/
    echo Start training ad-mip, scene: $scene.
    startTime=$(date +%s)
    bash ./ref_train/${scene}.sh >& rt_${scene}.out
    endTime=$(date +%s)
    echo Training time: $[$endTime-$startTime]s.

    #cd ../Ref-NPR-main
    #source activate RefNPR
    #echo Start training refnpr, scene: $scene.
    #startTime=$(date +%s)
    #bash ../Ref-NPR-main/npr_train/${scene}.sh >& ../Ref-NPR-main/npr_${scene}.out
    #endTime=$(date +%s)
    #echo Training time: $[$endTime-$startTime]s.

    #echo Start training snerf, scene: $scene.
    #startTime=$(date +%s)
    #bash ../Ref-NPR-main/s_train/${scene}.sh >& ../Ref-NPR-main/s_${scene}.out
    #endTime=$(date +%s)
    #echo Training time: $[$endTime-$startTime]s.

    #echo Start training arf, scene: $scene.
    #startTime=$(date +%s)
    #bash ../Ref-NPR-main/arf_train/${scene}.sh >& ../Ref-NPR-main/a_${scene}.out
    #endTime=$(date +%s)
    #echo Training time: $[$endTime-$startTime]s.
done

