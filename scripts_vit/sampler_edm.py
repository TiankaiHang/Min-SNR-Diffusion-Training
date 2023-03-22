import os
import sys
import numpy as np
import torch
import argparse

from functools import partial

import torch.nn.functional as F
from torchvision.utils import save_image
from diffusers.models import AutoencoderKL

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)

from torch_utils import distributed as dist

from guided_diffusion import dist_util
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    classifier_defaults,
    create_model_and_diffusion,
    create_classifier,
    add_dict_to_argparser,
    args_to_dict,
)


def build_model(**kwargs):
    if kwargs['model_name'] == 'vit_base_patch2_32':
        from guided_diffusion.vision_transformer import vit_base_patch2_32
        _model = vit_base_patch2_32(**kwargs)
    elif kwargs['model_name'] == 'vit_large_patch2_32':
        from guided_diffusion.vision_transformer import vit_large_patch2_32
        _model = vit_large_patch2_32(**kwargs)
    elif kwargs['model_name'] == 'vit_large_patch4_64':
        from guided_diffusion.vision_transformer import vit_large_patch4_64
        _model = vit_large_patch4_64(**kwargs)
    elif kwargs['model_name'] == 'vit_xl_patch2_32':
        from guided_diffusion.vision_transformer import vit_xl_patch2_32
        _model = vit_xl_patch2_32(**kwargs)
    else:
        raise NotImplementedError(f'Such model is not supported')
    return _model


class Net(torch.nn.Module):
    def __init__(self,
        model,
        img_resolution,                     # Image resolution.
        img_channels,                       # Number of color channels.
        pred_x0         = False,
        label_dim       = 0,                # Number of class labels, 0 = unconditional.
        use_fp16        = False,            # Execute the underlying model at FP16 precision?
        C_1             = 0.001,            # Timestep adjustment at low noise levels.
        C_2             = 0.008,            # Timestep adjustment at high noise levels.
        M               = 1000,             # Original number of timesteps in the DDPM formulation.
        noise_schedule  = 'cosine',
        # model_type      = 'DhariwalUNet',   # Class name of the underlying model.
        # **model_kwargs,                     # Keyword arguments for the underlying model.
    ):
        super().__init__()
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.label_dim = label_dim
        self.use_fp16 = use_fp16
        self.C_1 = C_1
        self.C_2 = C_2
        self.M = M
        # self.model = globals()[model_type](img_resolution=img_resolution, in_channels=img_channels, out_channels=img_channels*2, label_dim=label_dim, **model_kwargs)
        self.model = model
        self.noise_schedule = noise_schedule

        u = torch.zeros(M + 1)
        for j in range(M, 0, -1): # M, ..., 1
            u[j - 1] = ((u[j] ** 2 + 1) / (self.alpha_bar(j - 1) / self.alpha_bar(j)).clip(min=C_1) - 1).sqrt()
        self.register_buffer('u', u)
        self.sigma_min = float(u[M - 1])
        self.sigma_max = float(u[0])

        self.pred_x0 = pred_x0

    def forward(self, x, sigma, class_labels=None, force_fp32=False, **model_kwargs):
        x = x.to(torch.float32)
        sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1)
        # class_labels = None if self.label_dim == 0 else torch.zeros([1, self.label_dim], device=x.device) if class_labels is None else class_labels.to(torch.float32).reshape(-1, self.label_dim)
        dtype = torch.float16 if (self.use_fp16 and not force_fp32 and x.device.type == 'cuda') else torch.float32

        c_skip = 1
        c_out = -sigma
        c_in = 1 / (sigma ** 2 + 1).sqrt()
        c_noise = self.M - 1 - self.round_sigma(sigma, return_index=True).to(torch.float32)

        if model_kwargs.get('guidance_scale', 0) > 0:
            half = x[: len(x) // 2]
            combined = torch.cat([half, half], dim=0)
        else:
            combined = x

        F_x = self.model((c_in * combined).to(dtype), c_noise.flatten().repeat(x.shape[0]).int(), y=class_labels, **model_kwargs)

        assert F_x.dtype == dtype
        if not self.pred_x0:
            if model_kwargs.get('guidance_scale', 0) > 0:
                cond, uncond = torch.split(F_x, len(F_x) // 2, dim=0)
                cond = uncond + model_kwargs['guidance_scale'] * (cond - uncond)
                F_x = torch.cat([cond, cond], dim=0)

            D_x = c_skip * x + c_out * F_x[:, :self.img_channels].to(torch.float32)
        else:
            D_x = F_x
            if model_kwargs.get('guidance_scale', 0) > 0:
                cond, uncond = torch.split(D_x, len(D_x) // 2, dim=0)
                cond = uncond + model_kwargs['guidance_scale'] * (cond - uncond)
                D_x = torch.cat([cond, cond], dim=0)

        return D_x

    def alpha_bar(self, j):
        if self.noise_schedule == 'cosine':
            j = torch.as_tensor(j)
            return (0.5 * np.pi * j / self.M / (self.C_2 + 1)).sin() ** 2
        elif self.noise_schedule == 'linear':
            j = torch.as_tensor(j)
            betas = np.linspace(0.0001, 0.02, self.M + 1, dtype=np.float64)
            alphas = 1.0 - betas
            alphas_cumprod = np.cumprod(alphas, axis=0)
            return alphas_cumprod[self.M - j]

    def round_sigma(self, sigma, return_index=False):
        sigma = torch.as_tensor(sigma)
        index = torch.cdist(sigma.to(self.u.device).to(torch.float32).reshape(1, -1, 1), self.u.reshape(1, -1, 1)).argmin(2)
        result = index if return_index else self.u[index.flatten()].to(sigma.dtype)
        return result.reshape(sigma.shape).to(sigma.device)



def ablation_sampler(
    net, latents, class_labels=None, randn_like=torch.randn_like,
    num_steps=18, sigma_min=None, sigma_max=None, rho=7,
    solver='heun', discretization='edm', schedule='linear', scaling='none',
    epsilon_s=1e-3, C_1=0.001, C_2=0.008, M=1000, alpha=1,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
    **model_kwargs,
):
    assert solver in ['euler', 'heun']
    assert discretization in ['vp', 've', 'iddpm', 'edm']
    assert schedule in ['vp', 've', 'linear']
    assert scaling in ['vp', 'none']

    # Helper functions for VP & VE noise level schedules.
    vp_sigma = lambda beta_d, beta_min: lambda t: (np.e ** (0.5 * beta_d * (t ** 2) + beta_min * t) - 1) ** 0.5
    vp_sigma_deriv = lambda beta_d, beta_min: lambda t: 0.5 * (beta_min + beta_d * t) * (sigma(t) + 1 / sigma(t))
    vp_sigma_inv = lambda beta_d, beta_min: lambda sigma: ((beta_min ** 2 + 2 * beta_d * (sigma ** 2 + 1).log()).sqrt() - beta_min) / beta_d
    ve_sigma = lambda t: t.sqrt()
    ve_sigma_deriv = lambda t: 0.5 / t.sqrt()
    ve_sigma_inv = lambda sigma: sigma ** 2

    # Select default noise level range based on the specified time step discretization.
    if sigma_min is None:
        vp_def = vp_sigma(beta_d=19.9, beta_min=0.1)(t=epsilon_s)
        sigma_min = {'vp': vp_def, 've': 0.02, 'iddpm': 0.002, 'edm': 0.002}[discretization]
    if sigma_max is None:
        vp_def = vp_sigma(beta_d=19.9, beta_min=0.1)(t=1)
        sigma_max = {'vp': vp_def, 've': 100, 'iddpm': 81, 'edm': 80}[discretization]

    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    # Compute corresponding betas for VP.
    vp_beta_d = 2 * (np.log(sigma_min ** 2 + 1) / epsilon_s - np.log(sigma_max ** 2 + 1)) / (epsilon_s - 1)
    vp_beta_min = np.log(sigma_max ** 2 + 1) - 0.5 * vp_beta_d

    # Define time steps in terms of noise level.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    if discretization == 'vp':
        orig_t_steps = 1 + step_indices / (num_steps - 1) * (epsilon_s - 1)
        sigma_steps = vp_sigma(vp_beta_d, vp_beta_min)(orig_t_steps)
    elif discretization == 've':
        orig_t_steps = (sigma_max ** 2) * ((sigma_min ** 2 / sigma_max ** 2) ** (step_indices / (num_steps - 1)))
        sigma_steps = ve_sigma(orig_t_steps)
    elif discretization == 'iddpm':
        u = torch.zeros(M + 1, dtype=torch.float64, device=latents.device)
        alpha_bar = lambda j: (0.5 * np.pi * j / M / (C_2 + 1)).sin() ** 2
        for j in torch.arange(M, 0, -1, device=latents.device): # M, ..., 1
            u[j - 1] = ((u[j] ** 2 + 1) / (alpha_bar(j - 1) / alpha_bar(j)).clip(min=C_1) - 1).sqrt()
        u_filtered = u[torch.logical_and(u >= sigma_min, u <= sigma_max)]
        sigma_steps = u_filtered[((len(u_filtered) - 1) / (num_steps - 1) * step_indices).round().to(torch.int64)]
    else:
        assert discretization == 'edm'
        sigma_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho

    # Define noise level schedule.
    if schedule == 'vp':
        sigma = vp_sigma(vp_beta_d, vp_beta_min)
        sigma_deriv = vp_sigma_deriv(vp_beta_d, vp_beta_min)
        sigma_inv = vp_sigma_inv(vp_beta_d, vp_beta_min)
    elif schedule == 've':
        sigma = ve_sigma
        sigma_deriv = ve_sigma_deriv
        sigma_inv = ve_sigma_inv
    else:
        assert schedule == 'linear'
        sigma = lambda t: t
        sigma_deriv = lambda t: 1
        sigma_inv = lambda sigma: sigma

    # Define scaling schedule.
    if scaling == 'vp':
        s = lambda t: 1 / (1 + sigma(t) ** 2).sqrt()
        s_deriv = lambda t: -sigma(t) * sigma_deriv(t) * (s(t) ** 3)
    else:
        assert scaling == 'none'
        s = lambda t: 1
        s_deriv = lambda t: 0

    # Compute final time steps based on the corresponding noise levels.
    t_steps = sigma_inv(net.round_sigma(sigma_steps))
    t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])]) # t_N = 0

    # Main sampling loop.
    t_next = t_steps[0]
    x_next = latents.to(torch.float64) * (sigma(t_next) * s(t_next))
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
        x_cur = x_next

        # Increase noise temporarily.
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= sigma(t_cur) <= S_max else 0
        t_hat = sigma_inv(net.round_sigma(sigma(t_cur) + gamma * sigma(t_cur)))
        x_hat = s(t_hat) / s(t_cur) * x_cur + (sigma(t_hat) ** 2 - sigma(t_cur) ** 2).clip(min=0).sqrt() * s(t_hat) * S_noise * randn_like(x_cur)

        # Euler step.
        h = t_next - t_hat
        denoised = net(x_hat / s(t_hat), sigma(t_hat), class_labels, **model_kwargs).to(torch.float64)
        d_cur = (sigma_deriv(t_hat) / sigma(t_hat) + s_deriv(t_hat) / s(t_hat)) * x_hat - sigma_deriv(t_hat) * s(t_hat) / sigma(t_hat) * denoised
        x_prime = x_hat + alpha * h * d_cur
        t_prime = t_hat + alpha * h

        # Apply 2nd order correction.
        if solver == 'euler' or i == num_steps - 1:
            x_next = x_hat + h * d_cur
        else:
            assert solver == 'heun'
            denoised = net(x_prime / s(t_prime), sigma(t_prime), class_labels, **model_kwargs).to(torch.float64)
            d_prime = (sigma_deriv(t_prime) / sigma(t_prime) + s_deriv(t_prime) / s(t_prime)) * x_prime - sigma_deriv(t_prime) * s(t_prime) / sigma(t_prime) * denoised
            x_next = x_hat + h * ((1 - 1 / (2 * alpha)) * d_cur + 1 / (2 * alpha) * d_prime)
    
    return x_next


@torch.no_grad()
def main():
    dist.init()

    args, unknown = create_argparser().parse_known_args()
    dist.print0(f"====> args: {args}")
    dist.print0(f"====> unkown args: {unknown}")
    os.makedirs(os.environ.get('OPENAI_LOGDIR', 'exp'), exist_ok=True)

    seed = args.seed + dist.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    if args.in_chans == 4:
        vae = AutoencoderKL.from_pretrained(
            "stabilityai/sd-vae-ft-mse").cuda()
    
    if args.class_cond:
        args.num_classes += 1
    
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )

    if 'vit' in args.model_name or 'ldm_f8' in args.model_name:
        model = build_model(**vars(args))

    pretrained_obj = dist_util.load_state_dict(args.model_path, dist_type='pytorch', map_location="cpu")
    # import pdb; pdb.set_trace()
    # hack for non-consistent time embedding length
    if 'time_embedding.weight' in model.state_dict() and \
        model.state_dict()['time_embedding.weight'].shape[0] != pretrained_obj['time_embedding.weight'].shape[0]:
        
        tmp_time_embed = torch.zeros(size=(model.state_dict()['time_embedding.weight'].shape))
        useful_length = min(pretrained_obj['time_embedding.weight'].shape[0], model.state_dict()['time_embedding.weight'].shape[0])
        tmp_time_embed[:useful_length] = pretrained_obj['time_embedding.weight'][:useful_length]
        pretrained_obj['time_embedding.weight'] = tmp_time_embed

    # import pdb; pdb.set_trace()
    model.load_state_dict(
        pretrained_obj,
        strict=True,
    )
    model.eval()

    classifier = None
    classifier_kwargs = {}
    if args.guidance_type == "classifier":
        classifier = create_classifier(**args_to_dict(args, classifier_defaults().keys()))
        ckpt = torch.load(args.classifier_path, map_location="cpu")
        classifier.load_state_dict(ckpt)
        classifier.cuda()
        classifier.eval()

    net = Net(model=model, img_channels=args.in_chans, 
              img_resolution=args.image_size, 
              pred_x0=args.predict_xstart, 
              noise_schedule=args.noise_schedule).cuda()

    _iter = 0
    while _iter * args.batch_size * dist.get_world_size() < args.num_samples:
        
        class_labels = None
        if args.class_cond:
            y_cond = torch.randint(0, 1000, (args.batch_size,)).cuda()
            if args.guidance_scale > 0:
                y_uncond = torch.randint(1000, 1001, (args.batch_size,)).cuda()
                class_labels = torch.cat((y_cond, y_uncond), dim=0)
            else:
                y_uncond = None
                class_labels = y_cond
        
        z = torch.randn([args.batch_size, net.img_channels, net.img_resolution, net.img_resolution], device="cuda")
        if args.sample_name == "edm":
            if args.guidance_scale > 0:
                z = torch.cat((z, z), dim=0)
            samples = ablation_sampler(
                net, latents=z, 
                num_steps=args.steps, solver=args.edm_solver,
                class_labels=class_labels,
                guidance_scale=args.guidance_scale,
            )

        else:
            raise ValueError(f"Such sampler {args.sample_name} is not supported")

        if args.in_chans == 4:
            if args.guidance_scale > 0 and args.sample_name == "edm":
                samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
            samples = vae.decode(samples.float() / 0.18215).sample
        
        for _j in range(args.batch_size):
            save_image(samples[_j], fp=os.path.join(
                    os.environ.get('OPENAI_LOGDIR', 'exp'), 
                    f'{_iter}_{_j}_rank{dist.get_rank()}.png'), 
                normalize=True, value_range=(-1, 1))
        _iter += 1
        dist.print0(f"Sampled {(_iter * args.batch_size * dist.get_world_size()):06d} imgs")

    torch.distributed.barrier()


def create_argparser():
    defaults = dict(
        data_dir="",
        schedule_sampler="uniform",
        classifier_scale=1.0,
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        num_samples=10000,
        batch_size=16,
        use_ddim=False,
        classifier_depth=4,
        classifier_path="",
        model_path="",
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=10,
        save_interval=10000,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        exp_name="debug",
        use_wandb=True,
        model_name='vit_base_patch4_64',
        # vit related settings
        drop_path_rate=0.0,
        use_conv_last=False,
        use_shared_rel_pos_bias=False,
        depth=12,
        warmup_steps=0,
        lr_final=1e-5,
        in_chans=3,
        num_classes=1000,

        # sampler settings
        sample_name="edm", # ["dpm", "edm"]
        steps=35,
        edm_solver="heun",
        guidance_scale=0.0,
        guidance_type="classifier-free",

        use_rel_pos_bias=False,
        seed=2022,
    )
    defaults.update(model_and_diffusion_defaults())
    defaults.update(classifier_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == '__main__':
    main()
