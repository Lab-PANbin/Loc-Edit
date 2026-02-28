import torch
from os import path
import os
from config import get_config
from model import MipNeRF
import imageio
from datasets import get_dataloader, get_dataset
import cv2
import numpy as np
from tqdm import tqdm
from pose_utils import visualize_depth, visualize_normals, to8b


def visualize(config):
    data = get_dataset(config.dataset_name, config.base_dir, split="render", factor=config.factor)

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
    )
    model.load_state_dict(torch.load(config.model_weight_path))
    model.eval()

    print("Generating Video using", len(data), "different view points")
    rgb_frames = []
    if config.visualize_depth:
        depth_frames = []
    if config.visualize_normals:
        normal_frames = []

    output_path = os.path.join(config.log_dir, config.scene, 'test_rendering')
    os.makedirs(output_path, exist_ok=True)

    num_rays = len(data)
    batch_size = data.h * data.w
    N = num_rays//batch_size
    num_img = 10
    for i in tqdm(range(0, num_rays, (N//num_img)*batch_size)):
        ray = data[i:i+batch_size]
        img, dist, acc = model.render_image(ray, data.h, data.w, chunks=config.chunks)
        rgb_frames.append(img)
        cv2.imwrite(path.join(output_path, "render_{}.png".format(i//batch_size)),np.array(img)[:,:,::-1])  
        if config.visualize_depth:
            depth_frames.append(to8b(visualize_depth(dist, acc, data.near, data.far)))
        if config.visualize_normals:
            normal_frames.append(to8b(visualize_normals(dist, acc)))


    imageio.mimwrite(path.join(config.log_dir, config.scene, "video.mp4"), rgb_frames, fps=30, quality=10)
    if config.visualize_depth:
        imageio.mimwrite(path.join(config.log_dir, config.scene, "depth.mp4"), depth_frames, fps=30, quality=10)
    if config.visualize_normals:
        imageio.mimwrite(path.join(config.log_dir, config.scene, "normals.mp4"), normal_frames, fps=30, quality=10)


if __name__ == "__main__":
    config = get_config()
    visualize(config)
