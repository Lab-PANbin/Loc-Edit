import os
import argparse
import torch
from torchvision import transforms
import lpips
from PIL import Image

args = argparse.ArgumentParser()
args.add_argument('--scene', type=str)
args.add_argument('--method', type=str)
args.add_argument('--train', action='store_true', help='Using train pose or not.', default=False)
args = args.parse_args()

loss_fn_alex = lpips.LPIPS(net='alex')
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
])

if args.scene in set(['bottle', 'case', 'fengtiaoyushun', 'hotel', 'painting']):
    ref_path = f'./data/nerf_llff_data/ref_{args.scene}/images/0.jpg'
else:
    ref_path = f'./data/nerf_llff_data/ref_{args.scene}/images_4/0.jpg'

if args.method == 'ad-mip':
    if not args.train:
        rd_path = f'./ref_log/ref_{args.scene}/test_rendering'
    else:
        rd_path = f'./ref_log/ref_{args.scene}/train_rendering'
else:
    if not args.train:
        rd_path = f'../Ref-NPR-main/exps/{args.method}/{args.scene}/exp_out/test_renders_path'
    else:
        rd_path = f'../Ref-NPR-main/exps/{args.method}/{args.scene}/exp_out/train_renders'

rd_names = sorted(os.listdir(rd_path))



rd_imgs=[]
N = 0
for rd_name in rd_names:
    if rd_name[-3:] in set(['png', 'jpg']):
        rd_img = Image.open(os.path.join(rd_path, rd_name))
        rd_img = transform(rd_img).unsqueeze(0)  # [1, 3, H, W]
        rd_imgs.append(rd_img)
        N += 1
rd_imgs = torch.cat(rd_imgs, dim=0)  # [N, 3, H, W]

ref_img = Image.open(ref_path)
ref_img = transform(ref_img).unsqueeze(0).repeat_interleave(N, 0)  # [N, 3, H, W]

lpips_loss = loss_fn_alex.forward(ref_img, rd_imgs)
print(lpips_loss.mean().item())
