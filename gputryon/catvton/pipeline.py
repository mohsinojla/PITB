# Adapted from CatVTON (https://github.com/Zheng-Chong/CatVTON), by Zheng
# Chong et al., licensed CC BY-NC-SA 4.0. See ../README.md for attribution
# and license details -- most of this file is not original work. Changes
# from the upstream model/pipeline.py:
#   - import paths made package-relative (catvton.*, not model.*)
#   - the NSFW safety checker was dropped entirely (it needed an extra
#     ~600MB download of its own weights plus a bundled placeholder image
#     from the original repo we don't vendor here; this tool is for trying
#     your own photos, not a public-facing app, so we accepted that
#     tradeoff -- see ../README.md)
#   - auto_attn_ckpt_load now passes allow_patterns to snapshot_download so
#     it only fetches the one checkpoint variant actually being used
#     (mix-48k-1024 / vitonhd-16k-512 / dresscode-16k-512), not all three
import inspect
import os
from typing import Union

import PIL
import torch
import tqdm
from accelerate import load_checkpoint_in_model
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
from diffusers.utils.torch_utils import randn_tensor
from huggingface_hub import snapshot_download

from .attn_processor import SkipAttnProcessor
from .model_utils import get_trainable_module, init_adapter
from .utils import (compute_vae_encodings, numpy_to_pil, prepare_image,
                     prepare_mask_image, resize_and_crop, resize_and_padding)


class CatVTONPipeline:
    def __init__(
        self,
        base_ckpt,
        attn_ckpt,
        attn_ckpt_version="mix",
        weight_dtype=torch.float32,
        device='cuda',
        compile=False,
        use_tf32=True,
    ):
        self.device = device
        self.weight_dtype = weight_dtype

        self.noise_scheduler = DDIMScheduler.from_pretrained(base_ckpt, subfolder="scheduler")
        # The VAE is always kept in float32, even when weight_dtype is fp16.
        # SD's VAE decoder is well documented to overflow to NaN/Inf in fp16
        # (its GroupNorm/attention layers produce activations outside fp16
        # range for some inputs) -- confirmed here too: fp16 produced a
        # fully black output with a "invalid value encountered in cast"
        # warning right before saving. The VAE is a small fraction of the
        # pipeline's total weights, so running just it in fp32 costs very
        # little extra memory for a real fix (vs. e.g. giving up on fp16
        # for the whole pipeline, which we can't afford on a 4GB card).
        self.vae = AutoencoderKL.from_pretrained(
            "stabilityai/sd-vae-ft-mse", torch_dtype=torch.float32
        ).to(device)
        # Slicing/tiling trade a little speed for a lot less VAE memory --
        # worth it by default here since this project targets small (<=8GB)
        # consumer/workstation GPUs, not a dedicated inference server.
        self.vae.enable_slicing()
        self.vae.enable_tiling()

        self.unet = UNet2DConditionModel.from_pretrained(
            base_ckpt, subfolder="unet", torch_dtype=weight_dtype
        ).to(device)
        init_adapter(self.unet, cross_attn_cls=SkipAttnProcessor)  # Skip Cross-Attention (no text conditioning)
        self.attn_modules = get_trainable_module(self.unet, "attention")
        self.auto_attn_ckpt_load(attn_ckpt, attn_ckpt_version)

        if compile:
            self.unet = torch.compile(self.unet)
            self.vae = torch.compile(self.vae, mode="reduce-overhead")

        if use_tf32:
            torch.set_float32_matmul_precision("high")
            torch.backends.cuda.matmul.allow_tf32 = True

    def auto_attn_ckpt_load(self, attn_ckpt, version):
        sub_folder = {
            "mix": "mix-48k-1024",
            "vitonhd": "vitonhd-16k-512",
            "dresscode": "dresscode-16k-512",
        }[version]
        if os.path.exists(attn_ckpt):
            load_checkpoint_in_model(self.attn_modules, os.path.join(attn_ckpt, sub_folder, 'attention'))
        else:
            # allow_patterns restricts the download to the one checkpoint
            # variant we're actually going to use, instead of all three.
            repo_path = snapshot_download(repo_id=attn_ckpt, allow_patterns=[f"{sub_folder}/*"])
            load_checkpoint_in_model(self.attn_modules, os.path.join(repo_path, sub_folder, 'attention'))

    def check_inputs(self, image, condition_image, mask, width, height):
        if isinstance(image, torch.Tensor) and isinstance(condition_image, torch.Tensor) and isinstance(mask, torch.Tensor):
            return image, condition_image, mask
        assert image.size == mask.size, "Image and mask must have the same size"
        image = resize_and_crop(image, (width, height))
        mask = resize_and_crop(mask, (width, height))
        condition_image = resize_and_padding(condition_image, (width, height))
        return image, condition_image, mask

    def prepare_extra_step_kwargs(self, generator, eta):
        accepts_eta = "eta" in set(
            inspect.signature(self.noise_scheduler.step).parameters.keys()
        )
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta
        accepts_generator = "generator" in set(
            inspect.signature(self.noise_scheduler.step).parameters.keys()
        )
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    @torch.no_grad()
    def __call__(
        self,
        image: Union[PIL.Image.Image, torch.Tensor],
        condition_image: Union[PIL.Image.Image, torch.Tensor],
        mask: Union[PIL.Image.Image, torch.Tensor],
        num_inference_steps: int = 50,
        guidance_scale: float = 2.5,
        height: int = 1024,
        width: int = 768,
        generator=None,
        eta=1.0,
        **kwargs
    ):
        concat_dim = -2  # FIXME: y axis concat
        image, condition_image, mask = self.check_inputs(image, condition_image, mask, width, height)
        image = prepare_image(image).to(self.device, dtype=self.weight_dtype)
        condition_image = prepare_image(condition_image).to(self.device, dtype=self.weight_dtype)
        mask = prepare_mask_image(mask).to(self.device, dtype=self.weight_dtype)
        masked_image = image * (mask < 0.5)
        # compute_vae_encodings casts its input to self.vae's dtype
        # internally (float32 -- see __init__), so these come back float32
        # regardless of weight_dtype; cast back to weight_dtype right away
        # so everything downstream of the VAE (the UNet, the noise/mask
        # latents) stays in one consistent dtype.
        masked_latent = compute_vae_encodings(masked_image, self.vae).to(dtype=self.weight_dtype)
        condition_latent = compute_vae_encodings(condition_image, self.vae).to(dtype=self.weight_dtype)
        mask_latent = torch.nn.functional.interpolate(mask, size=masked_latent.shape[-2:], mode="nearest")
        del image, mask, condition_image
        masked_latent_concat = torch.cat([masked_latent, condition_latent], dim=concat_dim)
        mask_latent_concat = torch.cat([mask_latent, torch.zeros_like(mask_latent)], dim=concat_dim)
        latents = randn_tensor(
            masked_latent_concat.shape,
            generator=generator,
            device=masked_latent_concat.device,
            dtype=self.weight_dtype,
        )
        self.noise_scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = self.noise_scheduler.timesteps
        latents = latents * self.noise_scheduler.init_noise_sigma
        if do_classifier_free_guidance := (guidance_scale > 1.0):
            masked_latent_concat = torch.cat(
                [
                    torch.cat([masked_latent, torch.zeros_like(condition_latent)], dim=concat_dim),
                    masked_latent_concat,
                ]
            )
            mask_latent_concat = torch.cat([mask_latent_concat] * 2)

        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
        num_warmup_steps = (len(timesteps) - num_inference_steps * self.noise_scheduler.order)
        with tqdm.tqdm(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                non_inpainting_latent_model_input = (torch.cat([latents] * 2) if do_classifier_free_guidance else latents)
                non_inpainting_latent_model_input = self.noise_scheduler.scale_model_input(non_inpainting_latent_model_input, t)
                inpainting_latent_model_input = torch.cat([non_inpainting_latent_model_input, mask_latent_concat, masked_latent_concat], dim=1)
                noise_pred = self.unet(
                    inpainting_latent_model_input,
                    t.to(self.device),
                    encoder_hidden_states=None,
                    return_dict=False,
                )[0]
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )
                latents = self.noise_scheduler.step(
                    noise_pred, t, latents, **extra_step_kwargs
                ).prev_sample
                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps
                    and (i + 1) % self.noise_scheduler.order == 0
                ):
                    progress_bar.update()

        latents = latents.split(latents.shape[concat_dim] // 2, dim=concat_dim)[0]
        latents = 1 / self.vae.config.scaling_factor * latents
        # Decode in the VAE's own dtype (float32 -- see __init__), not
        # weight_dtype, for the same NaN-in-fp16 reason noted there.
        image = self.vae.decode(latents.to(self.device, dtype=self.vae.dtype)).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        return numpy_to_pil(image)
