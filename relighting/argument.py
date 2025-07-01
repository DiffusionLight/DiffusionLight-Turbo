import argparse
import torch
import numpy as np
from diffusers import DDIMScheduler, DDPMScheduler, UniPCMultistepScheduler

class CustomUniPCMultistepScheduler(UniPCMultistepScheduler):
    def set_full_timesteps(self):
        num_inference_steps = 999

        if self.config.timestep_spacing == "linspace":
            timesteps = (
                np.linspace(0, self.config.num_train_timesteps - 1, num_inference_steps + 1)
                .round()[::-1][:-1]
                .copy()
                .astype(np.int64)
            )
        elif self.config.timestep_spacing == "leading":
            step_ratio = self.config.num_train_timesteps // (num_inference_steps + 1)
            # creates integer timesteps by multiplying by ratio
            # casting to int to avoid issues when num_inference_step is power of 3
            timesteps = (np.arange(0, num_inference_steps + 1) * step_ratio).round()[::-1][:-1].copy().astype(np.int64)
            timesteps += self.config.steps_offset
        elif self.config.timestep_spacing == "trailing":
            step_ratio = self.config.num_train_timesteps / num_inference_steps
            # creates integer timesteps by multiplying by ratio
            # casting to int to avoid issues when num_inference_step is power of 3
            timesteps = np.arange(self.config.num_train_timesteps, 0, -step_ratio).round().copy().astype(np.int64)
            timesteps -= 1
        else:
            raise ValueError(
                f"{self.config.timestep_spacing} is not supported. Please make sure to choose one of 'linspace', 'leading' or 'trailing'."
            )

        sigmas = np.array(((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5)
        if self.config.use_karras_sigmas:
            log_sigmas = np.log(sigmas)
            sigmas = np.flip(sigmas).copy()
            sigmas = self._convert_to_karras(in_sigmas=sigmas, num_inference_steps=num_inference_steps)
            timesteps = np.array([self._sigma_to_t(sigma, log_sigmas) for sigma in sigmas]).round()
            sigmas = np.concatenate([sigmas, sigmas[-1:]]).astype(np.float32)
        else:
            sigmas = np.interp(timesteps, np.arange(0, len(sigmas)), sigmas)
            sigma_last = ((1 - self.alphas_cumprod[0]) / self.alphas_cumprod[0]) ** 0.5
            sigmas = np.concatenate([sigmas, [sigma_last]]).astype(np.float32)

        self.full_sigmas = torch.from_numpy(sigmas)
        self.full_timesteps = torch.from_numpy(timesteps).to(device="cuda", dtype=torch.int64)

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.IntTensor,
    ) -> torch.Tensor:
        
        self.set_full_timesteps()

        sigmas = self.full_sigmas.to(device=original_samples.device, dtype=original_samples.dtype)
        if original_samples.device.type == "mps" and torch.is_floating_point(timesteps):
            # mps does not support float64
            schedule_timesteps = self.full_timesteps.to(original_samples.device, dtype=torch.float32)
            timesteps = timesteps.to(original_samples.device, dtype=torch.float32)
        else:
            schedule_timesteps = self.full_timesteps.to(original_samples.device)
            timesteps = timesteps.to(original_samples.device)

        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < len(original_samples.shape):
            sigma = sigma.unsqueeze(-1)

        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma)
        noisy_samples = alpha_t * original_samples + sigma_t * noise
        return noisy_samples

def get_control_signal_type(controlnet):
    if "normal" in controlnet:
        return "normal"
    elif "depth" in controlnet:
        return "depth"
    else:
        raise NotImplementedError

SD_MODELS = {
    "sd15_old": "runwayml/stable-diffusion-inpainting",
    "sd15_new": "runwayml/stable-diffusion-inpainting",
    "sd21": "stabilityai/stable-diffusion-2-inpainting",
    "sdxl": "stabilityai/stable-diffusion-xl-base-1.0",
    "sdxl_turbo": "stabilityai/sdxl-turbo",
    "sdxl_lighting_4steps": ["ByteDance/SDXL-Lightning", "sdxl_lightning_4step_unet.safetensors"],
    "sdxl_lighting_8steps": ["ByteDance/SDXL-Lightning", "sdxl_lightning_8step_unet.safetensors"],
    "hypersd_8steps": ["ByteDance/Hyper-SD", "Hyper-SDXL-8steps-CFG-lora.safetensors"],
    "sd15_depth": "runwayml/stable-diffusion-inpainting",
}

VAE_MODELS = {
    "sdxl": "madebyollin/sdxl-vae-fp16-fix",
    "sdxl_turbo": "madebyollin/sdxl-vae-fp16-fix",
}

CONTROLNET_MODELS = {
    "sd15_old": "fusing/stable-diffusion-v1-5-controlnet-normal",
    "sd15_new": "lllyasviel/control_v11p_sd15_normalbae",
    "sd21": "thibaud/controlnet-sd21-normalbae-diffusers",
    "sdxl": "diffusers/controlnet-depth-sdxl-1.0",
    "sdxl_turbo": "diffusers/controlnet-depth-sdxl-1.0",
    "sdxl_lighting_4steps": "diffusers/controlnet-depth-sdxl-1.0",
    "sdxl_lighting_8steps": "diffusers/controlnet-depth-sdxl-1.0",
    "hypersd_8steps": "diffusers/controlnet-depth-sdxl-1.0",
    "sd15_depth": "lllyasviel/control_v11f1p_sd15_depth",
}

SAMPLERS = {
    "ddim": DDIMScheduler,
    "ddpm": DDPMScheduler,
    "unipc": CustomUniPCMultistepScheduler,
}

DEPTH_ESTIMATOR = "Intel/dpt-hybrid-midas"