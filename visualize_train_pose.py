import torch
from os import path
import os
from config import get_config
from model import MipNeRF
import imageio
from datasets import render_train_dataloader, get_dataset
from tqdm import tqdm
from pose_utils import visualize_depth, visualize_normals, to8b
import cv2
import numpy as np


def visualize(config):
    data = get_dataset(config.dataset_name, config.base_dir, split="train", factor=config.factor)

    model = MipNeRF(
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
        return_raw=False
    )
    model.load_state_dict(torch.load(config.model_weight_path))
    model.eval()

    print("Generating Video using", len(data), "different view points")

    output_path = os.path.join(config.log_dir, config.scene, 'train_rendering')
    os.makedirs(output_path, exist_ok=True)

    num_rays = len(data)
    batch_size = data.h * data.w
    #N = num_rays//batch_size
    #num_img = 10
    for i in tqdm(range(2*batch_size, 7*batch_size, batch_size)):
        ray = data[i:i+batch_size][0]
        ##是否返回raw  raw: (H,W,n_sample,4)  4--rgb+density
        ##修改此处即可
        """
        img, dist, acc, raws = model.render_image(ray, data.h, data.w, chunks=config.chunks)

        np.save("raws_{}".format(i),np.array(raws))  ##很大,1.76G左右
        """
        img, dist, acc = model.render_image(ray, data.h, data.w, chunks=config.chunks)

        ##cv2 RGB  Image BGR
        print('\n')
        print(path.join(output_path, "render_train_pose_rgb_{}.png".format(i//batch_size)))
        cv2.imwrite(path.join(output_path, "render_train_pose_rgb_{}.png".format(i//batch_size)),np.array(img)[:,:,::-1]) 
        #np.save("dist_{}".format(i),np.array(dist))      

        cv2.imwrite(path.join(output_path, "render_train_pose_depth_{}.png".format(i//batch_size)),np.array(to8b(visualize_depth(dist, acc, data.near, data.far)))[:,:,::-1]) 

        cv2.imwrite(path.join(config.log_dir, config.scene, "render_train_pose_normals_{}.png".format(i)),np.array(to8b(visualize_normals(dist, acc)))[:,:,::-1])
 


if __name__ == "__main__":
    config = get_config()
    visualize(config)
