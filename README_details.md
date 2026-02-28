# Directory Structure

- data (Dataset path, all in LLFF format)
  - data_LLFF (DTU dataset)
    - See below
  - nerf_llff_data (Custom datasets)
    - {scene} (scene is the name of the **original scene** represented by the dataset, can be understood as the dataset name)
      - images (Images from various poses, i.e., training data)
      - images_x (Some datasets may have this, representing images downsampled by x times. Since high resolution may cause out-of-memory issues, these images are sometimes used for training. **Note: You must specify the downsampling factor in the training script (see --factor in train/{scene}.sh), otherwise the results will be problematic**.)
      - sparse (Contains camera pose parameters estimated by COLMAP when creating the dataset)
      - database.db (Also parameters from COLMAP when creating the dataset)
      - poses_bounds.npy (Key file used for training, stores pose parameters)
    - ref_{scene} (Dataset of the **modified scene**)
      - images or images_x (Contains only two images: one is from data/nerf_llff_data/{scene}/images or images_x, and the other is the edited version based on it (this image is usually named 0 and needs to be first in order))
      - Other information is similar to {scene}, only poses_bounds.npy is meaningful (same as {scene})
- log (Training records path for **original scenes**)
  - {scene} (scene is the scene name, i.e., training records path for the scene)
    - test_rendering (Rendered images from test viewpoints)
    - train
    - train_rendering (Rendered images from training viewpoints)
    - depth.mp4 (Depth estimation video, obtained by rendering frame by frame and concatenating. Note that many videos may be corrupted.)
    - model.pt (Model parameter file)
    - normals.mp4
    - optim.pt (Optimizer parameter file)
    - video.mp4
- ref_log (Training records path for **edited scenes**)
  - ref_{scene} (scene is the scene name, i.e., training records path for the **edited** scene)
    - Files inside are similar to those in log
    - diff_rgb_rays.npy (Should be the ray dictionary for the modified region)
- ref_train (Slurm scripts path for *training* **edited scenes**)
  - {scene}.sh (scene is the scene name, i.e., script for training the **edited** scene)
- ref_vis (Slurm scripts path for *visualizing* **edited scenes** from **test viewpoints**. Only after running ref_vis will **ref_log** contain rendered image files like test_rendering from **test viewpoints**)
- ref_vis_train (Slurm scripts path for *visualizing* **edited scenes** from **training viewpoints**. Only after running ref_vis_train will **ref_log** contain rendered image files like train_rendering from **training viewpoints**)
- train (Slurm scripts path for *training* **original scenes**)
- vis (Slurm scripts path for *visualizing* **original scenes** from **test viewpoints**. Only after running vis will **log** contain rendered image files like test_rendering from **test viewpoints**)
- vis_train (Slurm scripts path for *visualizing* **original scenes** from **training viewpoints**. Only after running vis_train will **log** contain rendered image files like train_rendering from **training viewpoints**)

# File Descriptions

- bool_visualization.png (Mask image generated during training)
- config.py
- datasets.py (Dataset)
- extract_mesh.py
- g_sh.py (File used to generate run scripts)
- get_lpips.py (Python script to calculate LPIPS metric)
- get_mask.py (File to obtain mask for modified region, unrelated to model training and rendering)
- loss.py (Loss function)
- lpips.sh (Slurm script to calculate LPIPS)
- mipNeRF.yml (Environment dependencies)
- model.py (Model file)
- mp.py (Python script to calculate MP metric)
- mp.sh (Slurm script to calculate MP metric)
- pose_utils.py (For processing poses in data)
- ray_utils.py (For processing rays)
- README.md
- ref_vis_all.sh (Slurm script for batch rendering **edited scenes** from **test viewpoints**)
- ref_vis_train_all.sh (Slurm script for batch rendering **edited scenes** from **training viewpoints**)
- ref_visualize_train_pose.py (For rendering **edited scenes** from **training viewpoints**)
- ref_visualize.py (For rendering **edited scenes** from **test viewpoints**)
- scheduler.py (Scheduler during training)
- t_test.sh (For testing time)
- train_adjust_model.py (For training **edited scene** models)
- train.py (For training **original scene** models)
- vis_all.sh (Slurm script for batch rendering **original scenes** from **test viewpoints**)
- vis_train_all.sh (Slurm script for batch rendering **original scenes** from **training viewpoints**)
- visualize_train_pose.py (For rendering **original scenes** from **training viewpoints**)
- visualize.py (For rendering **original scenes** from **test viewpoints**)



# Complete Workflow

- **Prepare original scene data**: Collect original scene data in LLFF format and place it under data/nerf_llff_data

- **Train original scene**: Use scripts in train to train the original scene model, results will be in log/{scene}

- **Prepare edited scene data**: To modify colors/textures in the original scene, select one image from the original scene data and edit it. Place the edited image in ref_{scene}, such as data\nerf_llff_data\ref_bottle\images\0.jpg (Note: it should be sorted first for easy reading by Python scripts, so I named them all 0). There's another image in this path, which is the original image before modification. Besides, the sparse folder, database.db, and poses_bounds.npy are the same as the original scene data, just copy and paste them. In other words, the only difference between ref_{scene} and {scene} data is in the images directory: one has only two images, one is an image from {scene}, and the other is the edited version based on it.

  - The order position of images in data\nerf_llff_data\{scene}\images is meaningful and will be used as input next, i.e., --ori_image_num in ref_train

- **Train edited scene**: Use scripts in ref_train to train the edited scene model, results will be in ref_log/{scene}

- **Rendering**: Run vis to render the original scene, run ref_vis to render the edited scene, outputs will be in log/{scene} and ref_log/{scene} respectively
