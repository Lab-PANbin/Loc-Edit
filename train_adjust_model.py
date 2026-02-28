import os.path
import shutil
from config import get_config
from scheduler import MipLRDecay
from loss import ad_NeRFLoss, mse_to_psnr
from model import MipNeRF, ad_MipNeRF
import torch
import torch.optim as optim
import torch.utils.tensorboard as tb
from os import path
from datasets import get_dataloader, cycle
import numpy as np
from tqdm import tqdm
from ray_utils import Rays, convert_to_ndc, namedtuple_map


def train_model(config):
    print('training config:')
    print(config)
    ref_model_save_path = path.join(config.ref_log_dir, config.ref_scene, "model.pt")
    ref_optimizer_save_path = path.join(config.ref_log_dir, config.ref_scene, "optim.pt")

    dataloader,diff_rays,diff_images = get_dataloader(dataset_name=config.dataset_name, base_dir=config.ref_base_dir, split="train", factor=config.factor, 
                                     batch_size=config.batch_size, shuffle=True, device=config.device,train_ad_nerf = config.train_ad_nerf,ori_image_num=config.ori_image_num)
    data = iter(cycle(dataloader))

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
        device=config.device
    )

    ##加载base mipnerf，并冻结部分参数
    model.load_state_dict(torch.load(config.base_MipNeRF_path), strict=False)

    layer_i = 0
    for name, param in model.named_parameters():
        layer_i += 1
        if layer_i<=26:
            print('freeze layer:' + name)
            param.requires_grad = False
        else:
            print('unfreeze layer:' + name)

    # optimizer = optim.AdamW( model.parameters(), lr=config.lr_init, weight_decay=config.weight_decay)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=config.lr_init, weight_decay=config.weight_decay)

    if config.ref_continue_training:
        model.load_state_dict(torch.load(ref_model_save_path), strict=True)
        optimizer.load_state_dict(torch.load(ref_optimizer_save_path), strict=True)

    scheduler = MipLRDecay(optimizer, lr_init=config.lr_init, lr_final=config.lr_final, max_steps=config.ad_max_steps, lr_delay_steps=config.lr_delay_steps, lr_delay_mult=config.lr_delay_mult)
    loss_func = ad_NeRFLoss(config.coarse_weight_decay)

    os.makedirs(path.join(config.ref_log_dir, config.ref_scene), exist_ok=True)
    shutil.rmtree(path.join(config.ref_log_dir, config.ref_scene, 'train'), ignore_errors=True)
    logger = tb.SummaryWriter(path.join(config.ref_log_dir, config.ref_scene, 'train'), flush_secs=1)

    ##generate mask

    model.eval()
    l = diff_rays.origins.shape[0]
    test_mask = []
    print('total {} rays need adjust'.format(l))

    ln = int(l/200)
    with torch.no_grad():
        for i in range(ln):

            inputs = namedtuple_map(lambda r: (r[200*i:200*(i+1)]).cuda(), diff_rays)
            #labels = diff_images[200*i:200*(i+1)]
            comp_rgb, _, _, t_p1, t_p2 = model(inputs,return_tquantile=True)

            test_mask.append(torch.cat([inputs.origins,inputs.directions,t_p1[:,None],t_p2[:,None]],dim = -1))

        inputs = namedtuple_map(lambda r: (r[200*(i+1):l]).cuda(), diff_rays)
        #labels = diff_images[200*(i+1):l]
        comp_rgb, _, _, t_p1, t_p2 = model(inputs,return_tquantile=True)

        test_mask.append(torch.cat([inputs.origins,inputs.directions,t_p1[:,None],t_p2[:,None]],dim = -1))

        test_mask = torch.cat(test_mask,dim=0)

        print('{} rays need to adjust rgb'.format(test_mask.shape[0]))
        np.save(path.join(config.ref_log_dir, config.ref_scene, 'diff_rgb_rays'),np.array(test_mask.cpu()))

    model.train()

    for step in tqdm(range(0, config.ad_max_steps)):
        rays, pixels = next(data)
        comp_rgb, _, _,= model(rays)
        pixels = pixels.to(config.device)

        # Compute loss and update model weights.
        loss_val, psnr = loss_func(comp_rgb, pixels, rays.lossmult.to(config.device))
        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()
        scheduler.step()

        psnr = psnr.detach().cpu().numpy()
        logger.add_scalar('train/loss', float(loss_val.detach().cpu().numpy()), global_step=step)
        logger.add_scalar('train/coarse_psnr', float(np.mean(psnr[:-1])), global_step=step)
        logger.add_scalar('train/fine_psnr', float(psnr[-1]), global_step=step)
        logger.add_scalar('train/avg_psnr', float(np.mean(psnr)), global_step=step)
        logger.add_scalar('train/lr', float(scheduler.get_last_lr()[-1]), global_step=step)

        if step % config.save_every == 0:
            torch.save(model.state_dict(), ref_model_save_path)
            torch.save(optimizer.state_dict(), ref_optimizer_save_path)
            print('\nstep: {} train/loss: {}'.format(step, float(loss_val.detach().cpu().numpy())))
            print('step: {} train/coarse_psnr: {}'.format(step, float(np.mean(psnr[:-1]))))
            print('step: {} train/fine_psnr: {}'.format(step, float(psnr[-1])))
            print('step: {} train/avg_psnr: {}'.format(step, float(np.mean(psnr))))
            print('step: {} train/lr: {}'.format(step, float(scheduler.get_last_lr()[-1])))
            
    torch.save(model.state_dict(), ref_model_save_path)
    torch.save(optimizer.state_dict(), ref_optimizer_save_path)


def eval_model(config, model, data):
    model.eval()
    rays, pixels = next(data)
    with torch.no_grad():
        comp_rgb, _, _, _, _= model(rays)
    pixels = pixels.to(config.device)
    model.train()
    return torch.tensor([mse_to_psnr(torch.mean((rgb - pixels[..., :3])**2)) for rgb in comp_rgb])


if __name__ == "__main__":
    config = get_config()
    train_model(config)

