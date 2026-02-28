import os
import argparse
import torch
from torchvision import transforms
import clip
from PIL import Image
import torch.nn.functional as F

args = argparse.ArgumentParser()
args.add_argument('--scene', type=str)
args.add_argument('--method', type=str)
args.add_argument('--train', action='store_true', help='Using train pose or not.', default=False)
args = args.parse_args()

if args.scene in set(['bottle', 'case', 'fengtiaoyushun', 'hotel', 'painting']):
    ref_path = f'./data/nerf_llff_data/ref_{args.scene}/images/'
else:
    ref_path = f'./data/nerf_llff_data/ref_{args.scene}/images_4/'

if args.method == 'ad-mip':
    ref_rd_path = f'./ref_log/ref_{args.scene}/test_rendering' if not args.train else f'./ref_log/ref_{args.scene}/train_rendering'
    rd_path = f'./log/{args.scene}/test_rendering' if not args.train else f'./log/{args.scene}/train_rendering'
else:
    if not args.train: 
        ref_rd_path = f'../Ref-NPR-main/exps/{args.method}/{args.scene}/exp_out/test_renders_path'
        rd_path = f'../Ref-NPR-main/exps/base_pr/ckpt_svox2/llff/{args.scene}/test_renders_path'
    else:
        ref_rd_path = f'../Ref-NPR-main/exps/{args.method}/{args.scene}/exp_out/train_renders'
        rd_path = f'../Ref-NPR-main/exps/base_pr/ckpt_svox2/llff/{args.scene}/train_renders'

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# load rd
ref_rd_names = sorted(os.listdir(ref_rd_path))
ref_rd_imgs=[]
N_ref = 0
for ref_rd_name in ref_rd_names:
    if ref_rd_name[-3:] in set(['png', 'jpg']):
        ref_rd_img = Image.open(os.path.join(ref_rd_path, ref_rd_name))
        ref_rd_img = preprocess(ref_rd_img).unsqueeze(0)
        ref_rd_imgs.append(ref_rd_img)
        N_ref += 1
ref_rd_imgs = torch.cat(ref_rd_imgs, dim=0).to(device)  # [N, 3, H, W]

rd_names = sorted(os.listdir(rd_path))
rd_imgs=[]
N = 0
for rd_name in rd_names:
    if rd_name[-3:] in set(['png', 'jpg']):
        rd_img = Image.open(os.path.join(rd_path, rd_name))
        rd_img = preprocess(rd_img).unsqueeze(0)
        rd_imgs.append(rd_img)
        N += 1
rd_imgs = torch.cat(rd_imgs, dim=0).to(device)  # [N, 3, H, W]

rd_mask = torch.any(torch.abs(ref_rd_imgs - rd_imgs)>0.08, dim=1).unsqueeze(1).repeat_interleave(3,1).to(device)  # [N, 3, H, W]

if N != N_ref:
    raise ValueError

# load ref
ref_names = sorted(os.listdir(ref_path))
ref_img = preprocess(Image.open(os.path.join(ref_path, ref_names[0]))).unsqueeze(0).repeat_interleave(N,0).to(device)  # [N, 3, H, W]
ori_img = preprocess(Image.open(os.path.join(ref_path, ref_names[1]))).unsqueeze(0).repeat_interleave(N,0).to(device)  # [N, 3, H, W]
mask = torch.any(torch.abs(ref_img - ori_img) > 0.08, dim=1).unsqueeze(1).repeat_interleave(3,1).to(device)  # [N, 3, H, W]


# calculate
with torch.no_grad():
    # diff
    diff = torch.abs(ref_rd_imgs - rd_imgs).mean(dim=[1,2,3])  # [N]
    # cos_sim
    ref_features = model.encode_image(ref_img*mask)  # [N, 512]
    rd_features = model.encode_image(rd_imgs*rd_mask)  # [N, 512]
    cos_sim = F.cosine_similarity(ref_features, rd_features)
    
    mp = (1-diff)*cos_sim

print('Diff:', diff.mean().item())
print('Sim:', cos_sim.mean().item())
print('MP:', mp.mean().item())
