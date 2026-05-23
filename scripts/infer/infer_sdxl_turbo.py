import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from torchvision import transforms
from torchvision.transforms import ToPILImage
import torch.nn.functional as F

import csv
import os
from pathlib import Path
from PIL import Image
import numpy as np

from diffusers import StableDiffusionXLPipeline
from transformers import AutoModel, CLIPModel, CLIPProcessor, AutoModelForImageSegmentation

import inspect
from typing import Optional, Union, List, Tuple

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_DIR = Path(os.environ.get("RMLER_CHECKPOINT_DIR", str(PROJECT_ROOT / "checkpoints")))
OUTPUT_DIR = Path(os.environ.get("RMLER_OUTPUT_DIR", str(PROJECT_ROOT / "outputs")))
SDXL_TURBO_MODEL_PATH = os.environ.get(
    "RMLER_SDXL_TURBO_MODEL",
    str(CHECKPOINT_DIR / "sdxl-turbo"),
)

def denormalize(images: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    return (images / 2 + 0.5).clamp(0, 1)

def numpy_to_pil(images: np.ndarray) -> List[Image.Image]:
    if images.ndim == 3:
        images = images[None, ...]

    images = (images * 255).round().astype("uint8")
    if images.shape[-1] == 1:
        pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
    else:
        pil_images = [Image.fromarray(image) for image in images]

    return pil_images

def preprocess(pil_images):
    if isinstance(pil_images, Image.Image):
        pil_images = [pil_images]
    images = [np.array(img).astype(np.float16) / 255.0 for img in pil_images]
    tensors = [torch.tensor(img).permute(2, 0, 1) for img in images]
    tensors = [(tensor * 2 - 1).clamp(-1, 1) for tensor in tensors]
    batch_tensor = torch.stack(tensors)
    return batch_tensor

def xl_postprocess(image):
    if not isinstance(image, torch.Tensor):
        raise ValueError(
            f"Input for postprocessing is in incorrect format: {type(image)}. We only support pytorch tensor"
        )
    do_denormalize = [True] * image.shape[0]
    image = torch.stack(
        [denormalize(image[i]) if do_denormalize[i] else image[i] for i in range(image.shape[0])]
        )
    image = image.detach().cpu().permute(0, 2, 3, 1).float().numpy()
    return numpy_to_pil(image)

def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps

def self_pipe(latents, pipeline, device, prompt_embeds, pooled_prompt_embeds):
    do_classifier_free_guidance = False
    height = 512    
    width = 512
    num_inference_steps = 4
    guidance_scale = 0
    clip_skip = None
    joint_attention_kwargs = None
    _interrupt = False
    num_images_per_prompt = 1
    batch_size = 1

    if pipeline is not None:
        pipe = pipeline
    timesteps, num_inference_steps = retrieve_timesteps(pipe.scheduler, num_inference_steps, device, timesteps=None)

    num_channels_latents = pipe.unet.config.in_channels

    if latents is None:
        latents = pipe.prepare_latents(
                    batch_size * num_images_per_prompt,
                    num_channels_latents,
                    height,
                    width,
                    prompt_embeds.dtype,
                    device,
                    None,
                    None,
                )
    extra_step_kwargs = pipe.prepare_extra_step_kwargs(None, 0.0)

    add_text_embeds = pooled_prompt_embeds

    text_encoder_projection_dim = pipe.text_encoder_2.config.projection_dim

    add_time_ids = pipe._get_add_time_ids(
                (height, width),
                (0, 0),
                (height, width),
                dtype=prompt_embeds.dtype,
                text_encoder_projection_dim=text_encoder_projection_dim,
            )
    negative_add_time_ids = add_time_ids

    prompt_embeds = prompt_embeds.to(device)
    add_text_embeds = add_text_embeds.to(device)
    add_time_ids = add_time_ids.to(device).repeat(batch_size * num_images_per_prompt, 1)

    num_warmup_steps = max(len(timesteps) - num_inference_steps * pipe.scheduler.order, 0)

    for i, t in enumerate(timesteps):
        latent_model_input = latents
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)

        added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}

        noise_pred = pipe.unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=prompt_embeds,
                        timestep_cond=None,
                        cross_attention_kwargs=None,
                        added_cond_kwargs=added_cond_kwargs,
                        return_dict=False,
                    )[0]

        latents_dtype = latents.dtype
        
        latents = pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

    needs_upcasting = True
    if needs_upcasting:
        pipe.upcast_vae()
        latents = latents.to(next(iter(pipe.vae.post_quant_conv.parameters())).dtype)
    elif latents.dtype != pipe.vae.dtype:
        if torch.backends.mps.is_available():
            pipe.vae = pipe.vae.to(latents.dtype)

    return latents

# 策略网络定义
class PolicyNet(nn.Module):
    def __init__(self, embedding_dim=2048, hidden_dim=4096):
        super().__init__()
        self.mu_layer = nn.Sequential(
            nn.Linear(embedding_dim * 77, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
            nn.Sigmoid()
        )
        self.log_std_layer = nn.Sequential(
            nn.Linear(embedding_dim * 77, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
            nn.Softplus()
        )
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        mu = self.mu_layer(x)
        log_std = self.log_std_layer(x)
        std = torch.exp(log_std * 0.1)
        dist = Normal(mu, std)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob, dist

# 主函数
def main(latents, file_path, prompts1, prompts2, prompts1_, prompts2_, weights_path):
    # 初始化模型
    inference_dtype = torch.float16
    device1 = "cuda:0"
    device2 = "cuda:1"
    
    actor = PolicyNet(embedding_dim=2048, hidden_dim=2048*2).to(dtype=inference_dtype, device=device1)
    
    # 加载预训练权重
    if weights_path and os.path.exists(weights_path):
        actor.load_state_dict(torch.load(weights_path))
        print(f"Loaded weights from {weights_path}")
    else:
        print("Warning: No weights loaded, using random initialization")
    
    # 初始化SDXL pipeline
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        SDXL_TURBO_MODEL_PATH, 
        revision="main", 
        torch_dtype=torch.float16
    )
    pipeline.to(device2)
    pipeline.safety_checker = None
    
    # 准备输入
    with torch.no_grad():
        # 获取文本嵌入
        c11_dev2, _, c12_dev2, _ = pipeline.encode_prompt(
            prompt=prompts1, 
            num_images_per_prompt=1, 
            do_classifier_free_guidance=False, 
            device=device2
        )
        c21_dev2, _, c22_dev2, _ = pipeline.encode_prompt(
            prompt=prompts2, 
            num_images_per_prompt=1, 
            do_classifier_free_guidance=False, 
            device=device2
        )
        
        # 移动到device1
        c11_dev1 = c11_dev2.to(device1)
        c21_dev1 = c21_dev2.to(device1)
        c12_dev1 = c12_dev2.to(device1)
        c22_dev1 = c22_dev2.to(device1)
        
        current_state = c11_dev1 * 0.5 + c21_dev1 * 0.5
        
        # 推理循环
        for step in tqdm(range(50)):  # 生成10个样本
            current_state = current_state.to(device1, dtype=inference_dtype)
            
            # 生成动作
            action_raw, _, _ = actor(current_state)
            
            # 准备扩散输入
            action_for_mixing = action_raw.unsqueeze(1).expand(-1, 77, -1)
            c11_dev2 = c11_dev1.to(device2, dtype=inference_dtype)
            c21_dev2 = c21_dev1.to(device2, dtype=inference_dtype)
            action_for_mixing_dev2 = action_for_mixing.to(device2, dtype=inference_dtype)
            
            prompt_embeds_for_diffusion = c11_dev2 * action_for_mixing_dev2 + c21_dev2 * (1 - action_for_mixing_dev2)
            pooled_prompt_embeds_for_diffusion = (c12_dev1 * 0.5 + c22_dev1 * 0.5).to(device2, dtype=inference_dtype)
            
            # 生成图像
            latent0 = self_pipe(
                latents, 
                pipeline, 
                device=device2, 
                prompt_embeds=prompt_embeds_for_diffusion, 
                pooled_prompt_embeds=pooled_prompt_embeds_for_diffusion
            )
            image_tensor = pipeline.vae.decode(latent0 / pipeline.vae.config.scaling_factor)[0].to(device1)
            image_good = xl_postprocess(image_tensor)[0]
            
            # 保存结果
            image_good.save(os.path.join(file_path, f"infer_{step}.png"))
            
            # 更新状态
            current_state = prompt_embeds_for_diffusion.detach().to(device1)

if __name__ == "__main__":
    # 参数设置
    output_path = OUTPUT_DIR / "inference_results"
    weights_path = os.environ.get(
        "RMLER_POLICY_WEIGHTS",
        str(OUTPUT_DIR / "res_ppo_522_seg" / "2-baseloss-save" / "zebra&rabbit" / "policy_weights_update_8.pt"),
    )
    
    # 创建输出目录
    os.makedirs(output_path, exist_ok=True)
    
    # 示例prompt
    # style = "anime-style illustration, clean lineart, flat shading, soft pastel colors, simple background"
    # pose = "facing right, neutral expression"

    # prompts1_ = "white shark, smooth skin texture, underwater lighting, streamlined body, dorsal fin clearly visible, " + pose + ", " + style
    # prompts2_ = "kit fox, soft fur texture, desert terrain, large ears, bushy tail, " + pose + ", " + style


    style = "anime-style illustration, clean lineart, flat shading, soft pastel colors, simple background"
    # style = "cyberpunk aesthetic, neon lighting, high contrast colors, glowing elements, futuristic cityscape background, gritty texture, moody atmosphere"
    # style = "Van Gogh style painting, expressive brushstrokes, swirling textures, vivid and saturated colors, thick impasto oil paint, impressionist background"
    # style = "cubism, abstract forms, geometric distortions, flat perspective, bold outlines"

    pose = "facing right, neutral expression, standing pose, centered composition"

    prompts1_ = "cat" # "cauliflower, " + pose + ", " + style
    prompts2_ = "rabbit" # "eagle, " + pose + ", " + style


    prompts1 = "A photo of a full-body " + prompts1_
    prompts2 = "A photo of a full-body " + prompts2_
    
    
    # 准备latents
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        SDXL_TURBO_MODEL_PATH, 
        revision="main", 
        torch_dtype=torch.float16
    )
    latents = pipeline.prepare_latents(
        1, 4, 512, 512,
        dtype=torch.float16,
        device="cuda:1",
        generator=None,
        latents=None,
    )
    
    # 运行推理
    main(latents, output_path, prompts1, prompts2, prompts1_, prompts2_, weights_path)
