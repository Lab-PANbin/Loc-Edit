import torch
from torchvision import transforms
from PIL import Image
import os
import argparse

args = argparse.ArgumentParser()
args = args.parse_args()

ref_path = './data/nerf_llff_data/ref_room/images_4'

ref_names = sorted(os.listdir(ref_path))
ref_img = transforms.ToTensor()(Image.open(os.path.join(ref_path, ref_names[0])))  # [3, H, W]
ori_img = transforms.ToTensor()(Image.open(os.path.join(ref_path, ref_names[1])))  # [3, H, W]
mask = torch.any(torch.abs(ref_img - ori_img) > 0.1, dim=0).unsqueeze(0).repeat_interleave(3,0).float()  # [N, 3, H, W]
transforms.ToPILImage()(mask).save('mask.png')
