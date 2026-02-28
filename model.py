import torch
import torch.nn as nn
from ray_utils import sample_along_rays, resample_along_rays, volumetric_rendering, namedtuple_map
from pose_utils import to8b


class PositionalEncoding(nn.Module):
    def __init__(self, min_deg, max_deg):
        super(PositionalEncoding, self).__init__()
        self.min_deg = min_deg
        self.max_deg = max_deg
        self.scales = nn.Parameter(torch.tensor([2 ** i for i in range(min_deg, max_deg)]), requires_grad=False)

    def forward(self, x, y=None):
        shape = list(x.shape[:-1]) + [-1]
        x_enc = (x[..., None, :] * self.scales[:, None]).reshape(shape)
        x_enc = torch.cat((x_enc, x_enc + 0.5 * torch.pi), -1)
        if y is not None:
            # IPE
            y_enc = (y[..., None, :] * self.scales[:, None]**2).reshape(shape)
            y_enc = torch.cat((y_enc, y_enc), -1)
            x_ret = torch.exp(-0.5 * y_enc) * torch.sin(x_enc)
            y_ret = torch.maximum(torch.zeros_like(y_enc), 0.5 * (1 - torch.exp(-2 * y_enc) * torch.cos(2 * x_enc)) - x_ret ** 2)
            return x_ret, y_ret
        else:
            # PE
            x_ret = torch.sin(x_enc)
            return x_ret
        
class trigonometric_Encoding(nn.Module):
    def __init__(self, min_deg, max_deg):
        super(trigonometric_Encoding, self).__init__()
        self.min_deg = min_deg
        self.max_deg = max_deg
        self.scales = nn.Parameter(torch.tensor([2 ** i for i in range(min_deg, max_deg)]), requires_grad=False)

    def forward(self, x):
        batch_size = x.shape[0]
        x_enc = (x[:,None] * self.scales[:, None]).reshape(batch_size,-1)
        x_enc = torch.cat((x_enc, x_enc + 0.5 * torch.pi), -1)
        x_ret = torch.sin(x_enc)
        return x_ret



class MipNeRF(nn.Module):
    def __init__(self,
                 use_viewdirs=True,
                 randomized=False,
                 ray_shape="cone",
                 white_bkgd=True,
                 num_levels=2,
                 num_samples=128,
                 hidden=256,
                 density_noise=1,
                 density_bias=-1,
                 rgb_padding=0.001,
                 resample_padding=0.01,
                 min_deg=0,
                 max_deg=16,
                 viewdirs_min_deg=0,
                 viewdirs_max_deg=4,
                 device=torch.device("cpu"),
                 return_raw=False
                 ):
        super(MipNeRF, self).__init__()
        self.use_viewdirs = use_viewdirs
        self.init_randomized = randomized
        self.randomized = randomized
        self.ray_shape = ray_shape
        self.white_bkgd = white_bkgd
        self.num_levels = num_levels
        self.num_samples = num_samples
        self.density_input = (max_deg - min_deg) * 3 * 2
        self.rgb_input = 3 + ((viewdirs_max_deg - viewdirs_min_deg) * 3 * 2)
        self.density_noise = density_noise
        self.rgb_padding = rgb_padding
        self.resample_padding = resample_padding
        self.density_bias = density_bias
        self.hidden = hidden
        self.device = device
        self.return_raw = return_raw
        self.density_activation = nn.Softplus()

        self.positional_encoding = PositionalEncoding(min_deg, max_deg)
        self.density_net0 = nn.Sequential(
            nn.Linear(self.density_input, hidden),
            nn.ReLU(True),
            nn.Linear(hidden, hidden),
            nn.ReLU(True),
            nn.Linear(hidden, hidden),
            nn.ReLU(True),
            nn.Linear(hidden, hidden),
            nn.ReLU(True),
        )
        self.density_net1 = nn.Sequential(
            nn.Linear(self.density_input + hidden, hidden),
            nn.ReLU(True),
            nn.Linear(hidden, hidden),
            nn.ReLU(True),
            nn.Linear(hidden, hidden),
            nn.ReLU(True),
            nn.Linear(hidden, hidden),
            nn.ReLU(True),
        )
        self.final_density = nn.Sequential(
            nn.Linear(hidden, 1),
        )

        input_shape = hidden
        if self.use_viewdirs:
            input_shape = num_samples

            self.rgb_net0 = nn.Sequential(
                nn.Linear(hidden, hidden)
            )
            self.viewdirs_encoding = PositionalEncoding(viewdirs_min_deg, viewdirs_max_deg)
            self.rgb_net1 = nn.Sequential(
                nn.Linear(hidden + self.rgb_input, num_samples),
                nn.ReLU(True),
            )
        self.final_rgb = nn.Sequential(
            nn.Linear(input_shape, 3),
            nn.Sigmoid()
        )
        _xavier_init(self)
        self.to(device)

    def forward(self, rays):
        comp_rgbs = []
        distances = []
        accs = []
        ret_weights = []
        ret_t_samples = []
        for l in range(self.num_levels):
            # sample
            if l == 0:  # coarse grain sample  
                """     Returns:
                        t_vals: torch.tensor, [batch_size, num_samples], sampled z values.
                        means: torch.tensor, [batch_size, num_samples, 3], sampled means.
                        covs: torch.tensor, [batch_size, num_samples, 3, 3], sampled covariances."""
                t_vals, (mean, var) = sample_along_rays(rays.origins, rays.directions, rays.radii, self.num_samples,
                                                        rays.near, rays.far, randomized=self.randomized, lindisp=False,
                                                        ray_shape=self.ray_shape)
            else:  # fine grain sample/s
                """     Returns:
                        t_vals: torch.tensor(float32), [batch_size, num_samples+1].
                        points: torch.tensor(float32), [batch_size, num_samples, 3]."""
                t_vals, (mean, var) = resample_along_rays(rays.origins, rays.directions, rays.radii,
                                                          t_vals.to(rays.origins.device),
                                                          weights.to(rays.origins.device), randomized=self.randomized,
                                                          stop_grad=True, resample_padding=self.resample_padding,
                                                          ray_shape=self.ray_shape)
            # do integrated positional encoding of samples
            samples_enc = self.positional_encoding(mean, var)[0]
            samples_enc = samples_enc.reshape([-1, samples_enc.shape[-1]])

            # predict density
            new_encodings = self.density_net0(samples_enc)
            new_encodings = torch.cat((new_encodings, samples_enc), -1)
            new_encodings = self.density_net1(new_encodings)
            raw_density = self.final_density(new_encodings).reshape((-1, self.num_samples, 1))

            # predict rgb
            if self.use_viewdirs:
                #  do positional encoding of viewdirs
                viewdirs = self.viewdirs_encoding(rays.viewdirs.to(self.device))
                viewdirs = torch.cat((viewdirs, rays.viewdirs.to(self.device)), -1)
                viewdirs = torch.tile(viewdirs[:, None, :], (1, self.num_samples, 1))
                viewdirs = viewdirs.reshape((-1, viewdirs.shape[-1]))
                new_encodings = self.rgb_net0(new_encodings)
                new_encodings = torch.cat((new_encodings, viewdirs), -1)
                new_encodings = self.rgb_net1(new_encodings)
            raw_rgb = self.final_rgb(new_encodings).reshape((-1, self.num_samples, 3))

            # Add noise to regularize the density predictions if needed.
            if self.randomized and self.density_noise:
                raw_density += self.density_noise * torch.rand(raw_density.shape, dtype=raw_density.dtype, device=raw_density.device)

            # volumetric rendering
            rgb = raw_rgb * (1 + 2 * self.rgb_padding) - self.rgb_padding
            density = self.density_activation(raw_density + self.density_bias)
            comp_rgb, distance, acc, weights, alpha = volumetric_rendering(rgb, density, t_vals, rays.directions.to(rgb.device), self.white_bkgd)
            comp_rgbs.append(comp_rgb)
            distances.append(distance)
            accs.append(acc)
            ## used by distortion loss
            ret_weights.append(weights)
            ret_t_samples.append(t_vals)
        if self.return_raw:
            raws = torch.cat((torch.clone(rgb).detach(), torch.clone(density).detach()), -1).cpu()
            # Predicted RGB values for rays, Disparity map (inverse of depth), Accumulated opacity (alpha) along a ray
            return torch.stack(comp_rgbs), torch.stack(distances), torch.stack(accs),torch.stack(ret_weights),torch.stack(ret_t_samples), raws
        else:
            # Predicted RGB values for rays, Disparity map (inverse of depth), Accumulated opacity (alpha) along a ray
            return torch.stack(comp_rgbs), torch.stack(distances),torch.stack(accs),torch.stack(ret_weights),torch.stack(ret_t_samples)

    def render_image(self, rays, height, width, chunks=8192):
        """
        Return image, disparity map, accumulated opacity (shaped to height x width) created using rays as input.
        Rays should be all of the rays that correspond to this one single image.
        Batches the rays into chunks to not overload memory of device
        """
        length = rays[0].shape[0]
        rgbs = []
        dists = []
        accs = []
        raws = []

        with torch.no_grad():
            for i in range(0, length, chunks):
                # put chunk of rays on device
                chunk_rays = namedtuple_map(lambda r: r[i:i+chunks].to(self.device), rays)
                if self.return_raw:
                    rgb, distance, acc,_,_,raw = self(chunk_rays)
                    raws.append(raw)
                else:
                    rgb, distance, acc,_,_ = self(chunk_rays)
                rgbs.append(rgb[-1].cpu())
                dists.append(distance[-1].cpu())
                accs.append(acc[-1].cpu())

        if self.return_raw:
            raws = torch.cat(raws, dim=0).reshape(height, width, self.num_samples, 4).numpy()  ##self.num_samples为采样的个数
        rgbs = to8b(torch.cat(rgbs, dim=0).reshape(height, width, 3).numpy())
        dists = torch.cat(dists, dim=0).reshape(height, width).numpy()
        accs = torch.cat(accs, dim=0).reshape(height, width).numpy()
        if self.return_raw:
            return rgbs, dists, accs, raws
        else:
            return rgbs, dists, accs

    def train(self, mode=True):
        self.randomized = self.init_randomized
        super().train(mode)
        return self

    def eval(self):
        self.randomized = False
        return super().eval()


def _xavier_init(model):
    """
    Performs the Xavier weight initialization.
    """
    for module in model.modules():
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)


def _normal_init(model):
    """
    Performs normal weight initialization.
    """
    for module in model.modules():
        if isinstance(module, nn.Linear):
             nn.init.normal_(module.weight,mean=0,std=0.0001)



class ad_MipNeRF(nn.Module):
    def __init__(self,
                 ad_hidden = 256,
                 use_viewdirs=True,
                 randomized=False,
                 ray_shape="cone",
                 white_bkgd=True,
                 num_levels=2,
                 num_samples=128,
                 hidden=256,
                 density_noise=1,
                 density_bias=-1,
                 rgb_padding=0.001,
                 resample_padding=0.01,
                 min_deg=0,
                 max_deg=16,
                 viewdirs_min_deg=0,
                 viewdirs_max_deg=4,
                 device=torch.device("cpu"),
                 return_raw=False,
                 matching_chunck = 256
                 ):
        super(ad_MipNeRF, self).__init__()

        self.use_viewdirs = use_viewdirs
        self.init_randomized = randomized
        self.randomized = randomized
        self.ray_shape = ray_shape
        self.white_bkgd = white_bkgd
        self.num_levels = num_levels
        self.num_samples = num_samples
        self.density_input = (max_deg - min_deg) * 3 * 2
        self.rgb_input = 3 + ((viewdirs_max_deg - viewdirs_min_deg) * 3 * 2)
        self.density_noise = density_noise
        self.rgb_padding = rgb_padding
        self.resample_padding = resample_padding
        self.density_bias = density_bias
        self.device = device
        self.return_raw = return_raw
        self.matching_chunck = matching_chunck
        self.density_activation = nn.Softplus()

        self.positional_encoding = PositionalEncoding(min_deg, max_deg)
        self.density_net0 = nn.Sequential(
            nn.Linear(self.density_input, hidden),
            nn.ReLU(True),
            nn.Linear(hidden, hidden),
            nn.ReLU(True),
            nn.Linear(hidden, hidden),
            nn.ReLU(True),
            nn.Linear(hidden, hidden),
            nn.ReLU(True),
        )
        self.density_net1 = nn.Sequential(
            nn.Linear(self.density_input + hidden, hidden),
            nn.ReLU(True),
            nn.Linear(hidden, hidden),
            nn.ReLU(True),
            nn.Linear(hidden, hidden),
            nn.ReLU(True),
            nn.Linear(hidden, hidden),
            nn.ReLU(True),
        )
        self.final_density = nn.Sequential(
            nn.Linear(hidden, 1),
        )

        input_shape = hidden
        if self.use_viewdirs:
            input_shape = num_samples

            self.rgb_net0 = nn.Sequential(
                nn.Linear(hidden, hidden)
            )
            self.viewdirs_encoding = PositionalEncoding(viewdirs_min_deg, viewdirs_max_deg)
            self.rgb_net1 = nn.Sequential(
                nn.Linear(hidden + self.rgb_input, num_samples),
                nn.ReLU(True),
            )
        self.final_rgb = nn.Sequential(
            nn.Linear(input_shape, 3),
            nn.Sigmoid()
            )

        self.ad_net0 = nn.Sequential(
            nn.Linear(self.density_input, ad_hidden),
            nn.ReLU(True), 
            nn.Linear(ad_hidden, ad_hidden),
            nn.ReLU(True),
            nn.Linear(ad_hidden, ad_hidden),
            nn.ReLU(True),
            nn.Linear(ad_hidden, ad_hidden),
            nn.ReLU(True),
        )

        self.ad_net1 = nn.Sequential(
            nn.Linear(self.density_input + ad_hidden, ad_hidden),
            nn.ReLU(True), 
            nn.Linear(ad_hidden, ad_hidden),
            nn.ReLU(True),
            nn.Linear(ad_hidden, ad_hidden),
            nn.ReLU(True),
            nn.Linear(ad_hidden, ad_hidden),
            nn.ReLU(True),
            nn.Linear(ad_hidden, 3),
            nn.Sigmoid()
            ) 

        _xavier_init(self)
        self.to(device)

    def forward(self, rays, diff_origins=None, diff_directions=None,
                test_flag=False,return_tquantile = False,
                t_p1 = None ,t_p2 = None ,render_two_side = False,topk_points=4):
        comp_rgbs = []
        distances = []
        accs = []
        if test_flag: 
            for l in range(self.num_levels):
                # sample
                if l == 0:  # coarse grain sample  
                    """     Returns:
                            t_vals: torch.tensor, [batch_size, num_samples], sampled z values.
                            means: torch.tensor, [batch_size, num_samples, 3], sampled means.
                            covs: torch.tensor, [batch_size, num_samples, 3, 3], sampled covariances."""
                    t_vals, (mean, var) = sample_along_rays(rays.origins, rays.directions, rays.radii, self.num_samples,
                                                            rays.near, rays.far, randomized=self.randomized, lindisp=False,
                                                            ray_shape=self.ray_shape)
                else:  # fine grain sample/s
                    t_vals, (mean, var) = resample_along_rays(rays.origins, rays.directions, rays.radii,
                                                            t_vals.to(rays.origins.device),
                                                            weights.to(rays.origins.device), randomized=self.randomized,
                                                            stop_grad=True, resample_padding=self.resample_padding,
                                                            ray_shape=self.ray_shape)
                # do integrated positional encoding of samples
                samples_enc = self.positional_encoding(mean, var)[0]
                samples_enc = samples_enc.reshape([-1, samples_enc.shape[-1]])

                # predict density
                new_encodings = self.density_net0(samples_enc)
                new_encodings = torch.cat((new_encodings, samples_enc), -1)
                new_encodings = self.density_net1(new_encodings)
                raw_density = self.final_density(new_encodings).reshape((-1, self.num_samples, 1))

                # predict rgb
                if self.use_viewdirs:
                    #  do positional encoding of viewdirs
                    viewdirs = self.viewdirs_encoding(rays.viewdirs.to(self.device))
                    viewdirs = torch.cat((viewdirs, rays.viewdirs.to(self.device)), -1)
                    viewdirs = torch.tile(viewdirs[:, None, :], (1, self.num_samples, 1))
                    viewdirs = viewdirs.reshape((-1, viewdirs.shape[-1]))
                    new_encodings = self.rgb_net0(new_encodings)
                    new_encodings = torch.cat((new_encodings, viewdirs), -1)
                    new_encodings = self.rgb_net1(new_encodings)
                raw_rgb = self.final_rgb(new_encodings).reshape((-1, self.num_samples, 3))

                # Add noise to regularize the density predictions if needed.
                if self.randomized and self.density_noise:
                    raw_density += self.density_noise * torch.rand(raw_density.shape, dtype=raw_density.dtype, device=raw_density.device)

                ##original mipnerf output
                rgb = raw_rgb * (1 + 2 * self.rgb_padding) - self.rgb_padding
                density = self.density_activation(raw_density + self.density_bias)

                if l == 0:  # coarse grain sample
                    comp_rgb, distance, acc, weights, alpha = volumetric_rendering(rgb, density, t_vals, rays.directions.to(rgb.device),
                                                                                            self.white_bkgd) 
                ##adjust
                else:
                    d0 = self.ad_net0(samples_enc) 
                    ad_raw_rgb = self.ad_net1(torch.cat((samples_enc, d0), -1)).reshape((-1, self.num_samples, 3))

                    ad_rgb = ad_raw_rgb * (1 + 2 * self.rgb_padding) - self.rgb_padding

                    ##adjust density
                    t_mid = (t_vals[:,1:] + t_vals[:,0:-1])/2

                    #render_mask_d: tensor bool [batch_size, num_samples,1]
                    #render_mask_d = use_ad_MLP(diff_origins_d, diff_directions_d, rays.origins, rays.directions, t_mid)

                    #d1 = ad_density
                    #d1[torch.logical_not(render_mask_d)] = 0
                    #d2 = density
                    #d2[render_mask_d] = 0
                    #ad_density = d1 + d2

                    # get weights   [batch_size, num_samples]
                    _, _, _, weights, _ = volumetric_rendering(ad_rgb, density, t_vals, rays.directions.to(rgb.device), self.white_bkgd)
                    # top_values 包含前32个最大值
                    # top_indices 包含前32个最大值对应的索引
                    _, top_indices = torch.topk(weights, k=topk_points, dim=1)              
                    bool_index = torch.zeros_like(weights, dtype=torch.bool)
                    #bool_index 输出 torch.Size([1024, 128])
                    bool_index.scatter_(dim=1, index=top_indices, src=torch.ones_like(top_indices, dtype=torch.bool)) 
                    t_mid_2 = t_mid[bool_index].reshape(-1,topk_points)
                    #print(top_values.shape)  # 输出 torch.Size([1024, 32])
                    #print(top_indices.shape)  # 输出 torch.Size([1024, 32])

                    if render_two_side:
                        #render_mask_c: tensor bool [batch_size, 32]
                        render_mask_c = use_ad_MLP_RGB(diff_origins, diff_directions, rays.origins, rays.directions, t_mid_2 ,t_p = t_p2, matching_chunck = self.matching_chunck)
                    else:
                        render_mask_c = use_ad_MLP_RGB(diff_origins, diff_directions, rays.origins, rays.directions, t_mid_2 ,t_p = t_p1, matching_chunck = self.matching_chunck)

                    ##adjust rgb
                    ## bool_indextorch.Size([1024, 128])
                    ins = bool_index.clone()
                    ins[bool_index] = render_mask_c.flatten()

                    a1 = torch.zeros_like(ad_rgb)
                    a1[ins[:,:,None].repeat(1,1,3)]= ad_rgb[ins[:,:,None].repeat(1,1,3)]

                    a2 = rgb
                    a2[ins[:,:,None].repeat(1,1,3)] = 0
                    ad_rgb = a1 + a2

                    #ins2 = (torch.logical_not(bool_index) & torch.any(render_mask_c,dim=1)[:,None].repeat(1,self.num_samples))
                    #density[ins2] = 0.001

                    # volumetric rendering
                    comp_rgb, distance, acc, weights, alpha = volumetric_rendering(ad_rgb, density, t_vals, rays.directions.to(rgb.device), self.white_bkgd)
                comp_rgbs.append(comp_rgb)
                distances.append(distance)
                accs.append(acc)

        else:
            for l in range(self.num_levels):
                # sample
                if l == 0:  # coarse grain sample  
                    """     Returns:
                            t_vals: torch.tensor, [batch_size, num_samples], sampled z values.
                            means: torch.tensor, [batch_size, num_samples, 3], sampled means.
                            covs: torch.tensor, [batch_size, num_samples, 3, 3], sampled covariances."""
                    t_vals, (mean, var) = sample_along_rays(rays.origins, rays.directions, rays.radii, self.num_samples,
                                                            rays.near, rays.far, randomized=self.randomized, lindisp=False,
                                                            ray_shape=self.ray_shape)
                else:  # fine grain sample/s
                    t_vals, (mean, var) = resample_along_rays(rays.origins, rays.directions, rays.radii,
                                                            t_vals.to(rays.origins.device),
                                                            weights.to(rays.origins.device), randomized=self.randomized,
                                                            stop_grad=True, resample_padding=self.resample_padding,
                                                            ray_shape=self.ray_shape)
                # do integrated positional encoding of samples
                samples_enc = self.positional_encoding(mean, var)[0]
                samples_enc = samples_enc.reshape([-1, samples_enc.shape[-1]])

                # predict density
                new_encodings = self.density_net0(samples_enc)
                new_encodings = torch.cat((new_encodings, samples_enc), -1)
                new_encodings = self.density_net1(new_encodings)
                raw_density = self.final_density(new_encodings).reshape((-1, self.num_samples, 1))

                # predict rgb
                if self.use_viewdirs:
                    #  do positional encoding of viewdirs
                    viewdirs = self.viewdirs_encoding(rays.viewdirs.to(self.device))
                    viewdirs = torch.cat((viewdirs, rays.viewdirs.to(self.device)), -1)
                    viewdirs = torch.tile(viewdirs[:, None, :], (1, self.num_samples, 1))
                    viewdirs = viewdirs.reshape((-1, viewdirs.shape[-1]))
                    new_encodings = self.rgb_net0(new_encodings)
                    new_encodings = torch.cat((new_encodings, viewdirs), -1)
                    new_encodings = self.rgb_net1(new_encodings)
                raw_rgb = self.final_rgb(new_encodings).reshape((-1, self.num_samples, 3))

                # Add noise to regularize the density predictions if needed.
                if self.randomized and self.density_noise:
                    raw_density += self.density_noise * torch.rand(raw_density.shape, dtype=raw_density.dtype, device=raw_density.device)

                ##original mipnerf output
                rgb = raw_rgb * (1 + 2 * self.rgb_padding) - self.rgb_padding
                density = self.density_activation(raw_density + self.density_bias)

                ##adjust
                d0 = self.ad_net0(samples_enc) 
                ad_raw_rgb = self.ad_net1(torch.cat((samples_enc, d0), -1)).reshape((-1, self.num_samples, 3)) 

                '''ad_raw_rgb = ad_raw_rgb0*density + rgb'''
                ad_rgb = ad_raw_rgb * (1 + 2 * self.rgb_padding) - self.rgb_padding

                ## use stage2_mask-- mask=0:only change density, mask=1:only change rgb
                '''
                ad_rgb_ =  ad_rgb.clone()  
                rgb_ = rgb.clone()  
                ad_rgb_[stage2_mask==0,:,:] = 0
                rgb_[stage2_mask==1,:,:] = 0
                '''
                # volumetric rendering
                if return_tquantile:
                    comp_rgb, distance, acc, weights, alpha , t_p1, t_p2 = volumetric_rendering(ad_rgb, density, t_vals, rays.directions.to(rgb.device),
                                                                                            self.white_bkgd, return_tquantile = return_tquantile)
                else:
                    comp_rgb, distance, acc, weights, alpha = volumetric_rendering(ad_rgb, density, t_vals, rays.directions.to(rgb.device),
                                                                                            self.white_bkgd, return_tquantile = return_tquantile)
                comp_rgbs.append(comp_rgb)
                distances.append(distance)
                accs.append(acc)

        if return_tquantile:
            return (torch.stack(comp_rgbs), torch.stack(distances), torch.stack(accs), t_p1, t_p2)

        if self.return_raw:
            raws = torch.cat((torch.clone(rgb).detach(), torch.clone(density).detach()), -1).cpu()
            # Predicted RGB values for rays, Disparity map (inverse of depth), Accumulated opacity (alpha) along a ray
            return torch.stack(comp_rgbs), torch.stack(distances), torch.stack(accs), raws
        else:
            # Predicted RGB values for rays, Disparity map (inverse of depth), Accumulated opacity (alpha) along a ray
            return (torch.stack(comp_rgbs), torch.stack(distances), torch.stack(accs))

    def render_image(self, rays, height, width,diff_origins=None, diff_directions=None, 
                    t_p1 = None ,t_p2 = None ,render_two_side = False, chunks=8192,topk_points=4):
        """
        Return image, disparity map, accumulated opacity (shaped to height x width) created using rays as input.
        Rays should be all of the rays that correspond to this one single image.
        Batches the rays into chunks to not overload memory of device
        """
        length = rays[0].shape[0]
        rgbs = []
        dists = []
        accs = []
        with torch.no_grad():
            for i in range(0, length, chunks):
                # put chunk of rays on device
                chunk_rays = namedtuple_map(lambda r: r[i:i+chunks].to(self.device), rays)
                rgb, distance, acc = self(chunk_rays,
                                        diff_origins=diff_origins, diff_directions= diff_directions,
                                        test_flag=True,t_p1 = t_p1 ,t_p2 = t_p2 ,render_two_side = render_two_side,
                                        topk_points=topk_points)
                rgbs.append(rgb[-1].cpu())
                dists.append(distance[-1].cpu())
                accs.append(acc[-1].cpu())

        rgbs = to8b(torch.cat(rgbs, dim=0).reshape(height, width, 3).numpy())
        dists = torch.cat(dists, dim=0).reshape(height, width).numpy()
        accs = torch.cat(accs, dim=0).reshape(height, width).numpy()
        return rgbs, dists, accs

    def train(self, mode=True):
        self.randomized = self.init_randomized
        super().train(mode)
        return self

    def eval(self):
        self.randomized = False
        return super().eval()


def use_ad_MLP_RGB(diff_origins, diff_directions, origins, directions, t_mid, t_p, matching_chunck):
    """
    diff_origins: [diff_n,3]
    diff_directions: [diff_n,3]
    origins: torch.tensor(float32), [batch_size, 3], ray origins.
    directions: torch.tensor(float32), [batch_size, 3], ray directions.
    t_mid: torch.tensor, [batch_size, num_samples/4], sampled z values.
    t_p: torch.tensor, [diff_n]
    return:
    render_mask: tensor bool [batch_size, num_samples/4, 1]  True--need adjust
    """

    if diff_origins.shape[-1]==0:   ## do not need adjust (no diff rays)
        return torch.zeros((t_mid.shape[0],t_mid.shape[1]-1,1) ,dtype=torch.bool).cuda()

    l = t_mid.shape[0]
    ln = int(l/matching_chunck)
    ri = 0.002
    t_p = t_p.cuda()
    diff_origins = diff_origins.cuda()
    diff_directions = diff_directions.cuda()
    Batches = []
    i = -1
    for i in range(ln):
        
        t_b = t_mid[matching_chunck*i:matching_chunck*(i+1)]
        location = (t_b[:,:,None] * directions[matching_chunck*i:matching_chunck*(i+1)][:,None,:] + origins[matching_chunck*i:matching_chunck*(i+1)][:,None,:]).reshape(-1,3) # [batch_size*num_samples,3]

        # Calculate the vector from the point on the line to the target point
        P0P = diff_origins[None,:,:] - location[:,None,:].repeat(1,diff_origins.shape[0],1)  # [batch_size*num_samples, diff_n, 3]

        cross_product = torch.cross(P0P, diff_directions[None,:,:])  # [batch_size*num_samples, diff_n, 3]

        # Calculate the distance using the formula
        distance = torch.norm(cross_product,dim=2) / torch.norm(diff_directions,dim=1)[None,:] # [batch_size*num_samples, diff_n]
        side_mask = torch.any((torch.sum(-P0P*diff_directions[None,:,:],dim = 2)/(torch.norm(diff_directions,dim=1)[None,:])**2)<=t_p[None,:],dim=1)
        render_mask = torch.all(torch.stack([torch.any((distance<=ri),dim=1),side_mask],dim=-1),dim=1).reshape(t_b.shape[0],t_b.shape[1])
        Batches.append(render_mask)

    t_b= t_mid[matching_chunck*(i+1):l]

    location = (t_b[:,:,None] * directions[matching_chunck*(i+1):l][:,None,:] + origins[matching_chunck*(i+1):l][:,None,:]).reshape(-1,3) # [batch_size*num_samples,3]

    # Calculate the vector from the point on the line to the target point
    P0P = diff_origins[None,:,:] - location[:,None,:].repeat(1,diff_origins.shape[0],1)  # [batch_size*num_samples, diff_n, 3]

    cross_product = torch.cross(P0P, diff_directions[None,:,:])  # [batch_size*num_samples, diff_n, 3]

    # Calculate the distance using the formula
    distance = torch.norm(cross_product,dim=2) / torch.norm(diff_directions,dim=1)[None,:] # [batch_size*num_samples, diff_n]
    side_mask = torch.any((torch.sum(-P0P*diff_directions[None,:,:],dim = 2)/(torch.norm(diff_directions,dim=1)[None,:])**2)<=t_p[None,:],dim=1)
    render_mask = torch.all(torch.stack([torch.any((distance<=ri),dim=1),side_mask],dim=-1),dim=1).reshape(t_b.shape[0],t_b.shape[1])
    Batches.append(render_mask)

    return torch.cat(Batches,dim = 0)
