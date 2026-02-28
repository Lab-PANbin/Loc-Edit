import torch
from os import path
import os
from config import get_config
from model import MipNeRF, ad_MipNeRF
import imageio
from datasets import render_train_dataloader, get_dataloader, get_dataset
from tqdm import tqdm
from pose_utils import visualize_depth, visualize_normals, to8b
import cv2
import numpy as np


def visualize(config):
    print('visualize config:')
    print(config)
    mask_c = np.load( path.join(config.ref_log_dir, config.ref_scene, 'diff_rgb_rays.npy'))
    


    diff_origins = torch.tensor(mask_c[:,0:3])
    diff_directions = torch.tensor(mask_c[:,3:6])
    t_p1 = torch.tensor(mask_c[:,6])
    t_p2 = torch.tensor(mask_c[:,7])
    

    data = get_dataset(config.dataset_name, config.base_dir, split="train", factor=config.factor)

    model = ad_MipNeRF(
        ad_hidden = config.ad_hidden,
        use_viewdirs=config.use_viewdirs,
        randomized=config.randomized,
        ray_shape=config.ray_shape,
        white_bkgd=config.white_bkgd,
        num_levels=config.num_levels,
        num_samples=config.num_samples,
        hidden=config.hidden,
        density_noise=config.density_noise,
        density_bias=config.density_bias,
        rgb_padding=config.rgb_padding,
        resample_padding=config.resample_padding,
        min_deg=config.min_deg,
        max_deg=config.max_deg,
        viewdirs_min_deg=config.viewdirs_min_deg,
        viewdirs_max_deg=config.viewdirs_max_deg,
        device=config.device,
        matching_chunck = config.matching_chunck)
    model.load_state_dict(torch.load(config.model_weight_path))
    model.eval()

    print("Generating Video using", len(data), "different view points")

    output_path = os.path.join(config.ref_log_dir, config.ref_scene, 'train_rendering')
    os.makedirs(output_path, exist_ok=True)

    num_rays = len(data)
    batch_size = data.h * data.w
    #N = num_rays//batch_size
    #num_img = 10
    for i in tqdm(range(2*batch_size, 7*batch_size, batch_size)):
        ray = data[i:i+batch_size][0]
        img, dist, acc = model.render_image(ray, data.h, data.w, chunks=config.chunks,
                                            diff_origins=diff_origins, diff_directions= diff_directions,
                                            t_p1 = t_p1 ,t_p2 = t_p2 ,render_two_side = config.render_two_side,
                                            topk_points=config.topk_points)

        ##cv2 RGB  Image BGR
        print('\n')
        print(path.join(output_path, "render_train_pose_rgb{}.png".format(i//batch_size)))
        cv2.imwrite(path.join(output_path, "render_train_pose_rgb{}.png".format(i//batch_size)),np.array(img)[:,:,::-1])        

        cv2.imwrite(path.join(output_path, "render_train_pose_depth_{}.png".format(i//batch_size)),np.array(to8b(visualize_depth(dist, acc, data.near, data.far)))[:,:,::-1]) 

        cv2.imwrite(path.join(config.ref_log_dir, config.ref_scene, "render_train_pose_normals_{}.png".format(i)),np.array(to8b(visualize_normals(dist, acc)))[:,:,::-1])



if __name__ == "__main__":
    config = get_config()
    visualize(config)
