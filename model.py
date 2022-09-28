import typing as t
from logging import INFO, getLogger

import torch
import numpy as np
from numpy import typing as npt
from diffusers import StableDiffusionPipeline


_DEFAULT_LOGGER = getLogger(__name__)
_DEFAULT_LOGGER.setLevel(INFO)


def get_stable_diffusion_pipeline(auth_token: str):
    """Returns pre-trained Stable Diffusion pipeline.

    Args:
        auth_token (str): authentication token.
    """
    model_id = "CompVis/stable-diffusion-v1-4"
    return StableDiffusionPipeline.from_pretrained(model_id, use_auth_token=auth_token)  # type: ignore


def generate_images_from_sd_pipeline(
    sd_pipeline, 
    prompts: t.Sequence[str],
    n_inference_steps: int = 50,
    device: torch.device = torch.device("cpu")
) -> npt.NDArray[np.float32]:
    """Generates images matching provided list of prompts.

    Args:
        sd_pipeline: Stable Diffusion pipeline to use for generating images.
        prompts (t.Iterable[str]): a list of prompts to generate images for.
        n_inference_steps (int, optional): number of inference steps. Defaults to 50.
        device(torch.device, optional): device to use for pipeline.
    """
    if device.type == "mps":
        _DEFAULT_LOGGER.info("Running on MPS device can result in un-reproducible results.")
    
    sd_pipeline.to(device)
    n_prompts = len(prompts)
    img_shape = (512, 512, 3)
    images = np.zeros((n_prompts, *img_shape))
    for prompt_index, prompt in enumerate(prompts):
        print(f"Working on prompt {prompt_index} / {len(prompts)}.")
        image = sd_pipeline(prompt, num_inference_steps=n_inference_steps).images[0]
        images[prompt_index] = np.asarray(image)
    
    return images.astype(np.float32)
        