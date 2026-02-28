import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--type', type=str)
parser.add_argument('--scene', type=str)
parser = parser.parse_args()

scene = parser.scene

print(f"""Type: {parser.type}
Scene: {scene}""")

if parser.type == 'train':
    if scene != 'all':
        path = os.path.join('./train', scene + '.sh')
        with open(path, 'w') as shfile:
            shfile.write(f"""#!/bin/bash
#SBATCH -o mipnerf{scene}.out
#SBATCH -p compute1
#SBATCH --qos=normal
#SBATCH -J {scene}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1 

source activate mipNeRF
nvidia-smi
SCENE={scene}
FACTOR=4
echo "Train: $SCENE"

echo Factor: ${{FACTOR}}.
python -u train.py --dataset_name llff --scene ${{SCENE}} --save_every 1000 --log_dir log --max_steps 200000 --factor ${{FACTOR}}
""")
    else:
        for scene in os.listdir('./data/nerf_llff_data'):
            if scene[:3] != 'ref':
                path = os.path.join('./train', scene + '.sh')
                with open(path, 'w') as shfile:
                    shfile.write(f"""#!/bin/bash
#SBATCH -o mipnerf{scene}.out
#SBATCH -p compute1
#SBATCH --qos=normal
#SBATCH -J {scene}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1 

source activate mipNeRF
nvidia-smi
SCENE={scene}
echo "Train: $SCENE"

FACTOR=4
echo Factor: ${{FACTOR}}.
python -u train.py --dataset_name llff --scene ${{SCENE}} --save_every 1000 --log_dir log --max_steps 200000 --factor ${{FACTOR}}
""")

if parser.type == 'ref_train':
    if scene != 'all':
      path = os.path.join('./ref_train', scene + '.sh')
      with open(path, 'w') as shfile:
          shfile.write(f"""#!/bin/bash
#SBATCH -o rt_{scene}.out
#SBATCH -p compute1
#SBATCH --qos=normal
#SBATCH -J rt_{scene}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1 

source activate mipNeRF
nvidia-smi
SCENE={scene}

FACTOR=4

echo Factor: ${{FACTOR}}.
ori=`ls -l ./data/nerf_llff_data/${{SCENE}}/images | grep "^-" | wc -l`
python -u train_adjust_model.py --dataset_name ad_llff --scene ${{SCENE}} --batch_size 256 --train_ad_nerf --train_ad_nerf  --ref_scene ref_${{SCENE}} --base_MipNeRF_path ./log/${{SCENE}}/model.pt --ref_log_dir ref_log --ad_max_steps 25000 --max_steps 25000 --render_two_side --ad_hidden 256 --ori_image_num ${{ori}} --factor ${{FACTOR}}""")
    else:
        for scene in os.listdir('./log'):
            path = os.path.join('./ref_train', scene + '.sh')
            with open(path, 'w') as shfile:
                shfile.write(f"""#!/bin/bash
#SBATCH -o rt_{scene}.out
#SBATCH -p compute1
#SBATCH --qos=normal
#SBATCH -J rt_{scene}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1 

source activate mipNeRF
nvidia-smi
SCENE={scene}

FACTOR=4

echo Factor: ${{FACTOR}}.
ori=`ls -l ./data/nerf_llff_data/${{SCENE}}/images | grep "^-" | wc -l`
python -u train_adjust_model.py --dataset_name ad_llff --scene ${{SCENE}} --batch_size 256 --train_ad_nerf --train_ad_nerf  --ref_scene ref_${{SCENE}} --base_MipNeRF_path ./log/${{SCENE}}/model.pt --ref_log_dir ref_log --ad_max_steps 25000 --max_steps 25000 --render_two_side --ad_hidden 256 --ori_image_num ${{ori}} --factor ${{FACTOR}}""")

if parser.type == 'vis':
    if scene != 'all':
        path = os.path.join('./vis', scene + '.sh')
        with open(path, 'w') as shfile:
            shfile.write(f"""#!/bin/bash
#SBATCH -o v_{scene}.out
#SBATCH -p compute1
#SBATCH --qos=normal
#SBATCH -J v_{scene}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1 

source activate mipNeRF
nvidia-smi
SCENE={scene}

FACTOR=4

echo Factor: ${{FACTOR}}.
python -u visualize.py --dataset_name llff --scene ${{SCENE}} --visualize_depth --visualize_normals --model_weight_path ./log/${{SCENE}}/model.pt --log_dir log --factor ${{FACTOR}}""")
    else:
        for scene in os.listdir('./log'):
            path = os.path.join('./vis', scene + '.sh')
            with open(path, 'w') as shfile:
                shfile.write(f"""#!/bin/bash
#SBATCH -o v_{scene}.out
#SBATCH -p compute1
#SBATCH --qos=normal
#SBATCH -J v_{scene}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1 

source activate mipNeRF
nvidia-smi
SCENE={scene}

FACTOR=4
echo Factor: ${{FACTOR}}.
python -u visualize.py --dataset_name llff --scene ${{SCENE}} --visualize_depth --visualize_normals --model_weight_path ./log/${{SCENE}}/model.pt --log_dir log --factor ${{FACTOR}}""")


if parser.type == 'vis_train':
    if scene != 'all':
        path = os.path.join('./vis_train', scene + '.sh')
        with open(path, 'w') as shfile:
            shfile.write(f"""#!/bin/bash
#SBATCH -o vt_{scene}.out
#SBATCH -p compute1
#SBATCH --qos=normal
#SBATCH -J vt_{scene}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1 

source activate mipNeRF
nvidia-smi
SCENE={scene}

FACTOR=4

echo Factor: ${{FACTOR}}.
python -u visualize_train_pose.py --dataset_name llff --scene ${{SCENE}} --visualize_depth --visualize_normals --model_weight_path ./log/${{SCENE}}/model.pt --log_dir log --factor ${{FACTOR}}""")
    else:
        for scene in os.listdir('./log'):
            path = os.path.join('./vis_train', scene + '.sh')
            with open(path, 'w') as shfile:
                shfile.write(f"""#!/bin/bash
#SBATCH -o vt_{scene}.out
#SBATCH -p compute1
#SBATCH --qos=normal
#SBATCH -J vt_{scene}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1 

source activate mipNeRF
nvidia-smi
SCENE={scene}

FACTOR=4
echo Factor: ${{FACTOR}}.
python -u visualize_train_pose.py --dataset_name llff --scene ${{SCENE}} --visualize_depth --visualize_normals --model_weight_path ./log/${{SCENE}}/model.pt --log_dir log --factor ${{FACTOR}}""")



if parser.type == 'ref_vis':
    if scene != 'all':
      path = os.path.join('./ref_vis', scene + '.sh')
      with open(path, 'w') as shfile:
          shfile.write(f"""#!/bin/bash
#SBATCH -o rv_{scene}.out
#SBATCH -J rv_{scene}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -p compute1
#SBATCH --gres=gpu:1

source activate mipNeRF
SCENE={scene}

FACTOR=4

echo Factor: ${{FACTOR}}.
python ref_visualize.py --dataset_name llff --scene ${{SCENE}} --visualize_depth --visualize_normals --train_ad_nerf --train_ad_nerf  --ref_scene ref_${{SCENE}} --model_weight_path ref_log/ref_${{SCENE}}/model.pt --ref_log_dir ref_log --chunks 2048 --topk_points 32 --ad_hidden 256 --matching_chunck 256 --factor ${{FACTOR}}""")
    else:
        for scene in os.listdir('./log'):
            path = os.path.join('./ref_vis', scene + '.sh')
            with open(path, 'w') as shfile:
                shfile.write(f"""#!/bin/bash
#SBATCH -o rv_{scene}.out
#SBATCH -J rv_{scene}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -p compute1
#SBATCH --gres=gpu:1

source activate mipNeRF
SCENE={scene}

FACTOR=4
echo Factor: ${{FACTOR}}.
python ref_visualize.py --dataset_name llff --scene ${{SCENE}} --visualize_depth --visualize_normals --train_ad_nerf --train_ad_nerf  --ref_scene ref_${{SCENE}} --model_weight_path ref_log/ref_${{SCENE}}/model.pt --ref_log_dir ref_log --chunks 2048 --topk_points 32 --ad_hidden 256 --matching_chunck 256 --factor ${{FACTOR}}""")


if parser.type == 'ref_vis_train':
    if scene != 'all':
      path = os.path.join('./ref_vis_train', scene + '.sh')
      with open(path, 'w') as shfile:
          shfile.write(f"""#!/bin/bash
#SBATCH -o rvt_{scene}.out
#SBATCH -J rvt_{scene}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -p compute1
#SBATCH --gres=gpu:1

source activate mipNeRF
SCENE={scene}

FACTOR=4

echo Factor: ${{FACTOR}}.
ori=`ls -l ./data/nerf_llff_data/${{SCENE}}/images | grep "^-" | wc -l`
python ref_visualize_train_pose.py --dataset_name llff --scene ${{SCENE}} --visualize_depth --visualize_normals --train_ad_nerf --train_ad_nerf  --ref_scene ref_${{SCENE}} --model_weight_path ref_log/ref_${{SCENE}}/model.pt --ref_log_dir ref_log --chunks 2048 --topk_points 32 --ad_hidden 256 --matching_chunck 256 --factor ${{FACTOR}}""")
    else:
        for scene in os.listdir('./log'):
            path = os.path.join('./ref_vis_train', scene + '.sh')
            with open(path, 'w') as shfile:
                shfile.write(f"""#!/bin/bash
#SBATCH -o rvt_{scene}.out
#SBATCH -J rvt_{scene}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -p compute1
#SBATCH --gres=gpu:1

source activate mipNeRF
SCENE={scene}

FACTOR=4
echo Factor: ${{FACTOR}}.
ori=`ls -l ./data/nerf_llff_data/${{SCENE}}/images | grep "^-" | wc -l`
python ref_visualize_train_pose.py --dataset_name llff --scene ${{SCENE}} --visualize_depth --visualize_normals --train_ad_nerf --train_ad_nerf  --ref_scene ref_${{SCENE}} --model_weight_path ref_log/ref_${{SCENE}}/model.pt --ref_log_dir ref_log --chunks 2048 --topk_points 32 --ad_hidden 256 --matching_chunck 256 --factor ${{FACTOR}}""")





if parser.type == 'ref_vis_r':
    if scene != 'all':
      path = os.path.join('./ref_vis_r', scene + '.sh')
      with open(path, 'w') as shfile:
          shfile.write(f"""#!/bin/bash
#SBATCH -o rv_{scene}.out
#SBATCH -J rv_{scene}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -p compute1
#SBATCH --gres=gpu:1

source activate mipNeRF
SCENE={scene}

FACTOR=4
echo Factor: ${{FACTOR}}.
python ref_visualize_r.py --dataset_name llff --scene ${{SCENE}} --visualize_depth --visualize_normals --train_ad_nerf --train_ad_nerf  --ref_scene ref_${{SCENE}} --model_weight_path ref_log/ref_${{SCENE}}/model.pt --ref_log_dir ref_log --chunks 2048 --topk_points 32 --ad_hidden 256 --matching_chunck 256 --factor ${{FACTOR}}""")
    
    else:
        for scene in os.listdir('./log'):
            path = os.path.join('./ref_vis_r', scene + '.sh')
            with open(path, 'w') as shfile:
                shfile.write(f"""#!/bin/bash
#SBATCH -o rv_{scene}.out
#SBATCH -J rv_{scene}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -p compute1
#SBATCH --gres=gpu:1

source activate mipNeRF
SCENE={scene}

FACTOR=4
echo Factor: ${{FACTOR}}.
python ref_visualize_r.py --dataset_name llff --scene ${{SCENE}} --visualize_depth --visualize_normals --train_ad_nerf --train_ad_nerf  --ref_scene ref_${{SCENE}} --model_weight_path ref_log/ref_${{SCENE}}/model.pt --ref_log_dir ref_log --chunks 2048 --topk_points 32 --ad_hidden 256 --matching_chunck 256 --factor ${{FACTOR}}""")

print('Finished.')
