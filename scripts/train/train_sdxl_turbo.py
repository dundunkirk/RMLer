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
DATASET_DIR = Path(os.environ.get("RMLER_DATASET_DIR", str(PROJECT_ROOT / "dataset")))
OUTPUT_DIR = Path(os.environ.get("RMLER_OUTPUT_DIR", str(PROJECT_ROOT / "outputs")))
CLIP_MODEL_PATH = os.environ.get(
    "RMLER_CLIP_MODEL",
    str(CHECKPOINT_DIR / "CLIP-ViT-H-14-laion2B-s32B-b79K"),
)
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

to_pil_image = ToPILImage()
model_name = 'facebook/dinov2-base'
dinov2_transforms = transforms.Compose([
    transforms.Resize(size=224, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(size=(224, 224)),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])
to_tensor = transforms.ToTensor() 

clip_model_name = CLIP_MODEL_PATH
clip_model = CLIPModel.from_pretrained(clip_model_name)
clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
cliptransforms_origin = transforms.Compose([
        transforms.Resize((224, 224)),
    ])

clip_model.to("cuda:0")
clip_model.requires_grad_(False)

seg_model = AutoModelForImageSegmentation.from_pretrained('briaai/RMBG-2.0', trust_remote_code=True)
seg_model.to("cuda:0")
seg_model.eval()
    
def segment_image(image):
    if isinstance(image, torch.Tensor):
        if image.ndim == 4:
            image = image.squeeze(0)
        
        if image.ndim == 3:
            image = to_pil_image(image.cpu().float().clamp(0, 1))
        else:
            raise ValueError(f"Unexpected tensor dimensions: {image.shape}")
    elif not isinstance(image, Image.Image):
        try:
            image = Image.fromarray(np.uint8(image * 255))
        except:
            raise ValueError(f"Unsupported image type: {type(image)}")
    
    seg_transform = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    input_image = seg_transform(image).unsqueeze(0).to("cuda:0")
    
    with torch.no_grad():
        pred = seg_model(input_image)[-1].sigmoid().cpu()
    pred = pred[0].squeeze()
    pred_pil = transforms.ToPILImage()(pred)
    mask = pred_pil.resize(image.size)
    image.putalpha(mask)

    return image

def get_image_features_clip(image):
    segmented = segment_image(image)
    segmented.save("demo.png")
    if segmented.mode == 'RGBA':
        background = Image.new('RGB', segmented.size, (255, 255, 255))
        composited = Image.alpha_composite(background.convert('RGBA'), segmented).convert('RGB')
        
        segmented = composited
    inputs = clip_processor(images=segmented, return_tensors="pt").to("cuda:0")
    with torch.no_grad():
        image_features = clip_model.get_image_features(**inputs)
    
    return image_features

def get_text_features(t1, t2):
    with torch.no_grad():
        text_inputs = clip_processor(text=t1, return_tensors="pt", padding=True).to("cuda:0")
        text_features_1 = clip_model.get_text_features(**text_inputs)
        text_inputs = clip_processor(text=t2, return_tensors="pt", padding=True).to("cuda:0")
        text_features_2 = clip_model.get_text_features(**text_inputs)
        return text_features_1, text_features_2

def creative_reward_fn():
    def sim(f1, f2):
        return torch.nn.functional.cosine_similarity(f1, f2, dim=-1)

    def reward_fn(image_good, f1, f2, t1, t2):
        f_good = get_image_features_clip(image_good)

        sim1 = sim(f_good, f1)
        sim2 = sim(f_good, f2)
        
        reward = (sim1 + sim2)  - torch.abs(sim1 - sim2) * 5
        
        return reward, sim1, sim2

    return reward_fn

def compute_gae(rewards, values, next_value, gamma=0.99, lam=0.95):
    advantages = []
    gae = 0
    rewards = [r.item() if torch.is_tensor(r) else r for r in rewards]
    values = [v.item() if torch.is_tensor(v) else v for v in values]
    next_value = next_value.item() if torch.is_tensor(next_value) else next_value

    num_steps = len(rewards)
    values = values + [next_value]

    for t in reversed(range(num_steps)):
        delta = rewards[t] + gamma * values[t + 1] - values[t]
        gae = delta + gamma * lam * gae
        advantages.insert(0, gae)

    return advantages

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


def ppo_update(actor, actor_optimizer, states, actions_raw, logp_old, rewards, clip_eps=0.2, epochs=1, device="cuda:0", csv_writer=None, save_path_prefix=None):

    states = states.to(device)
    actions_raw = actions_raw.to(device)
    logp_old = logp_old.to(device)
    rewards = rewards.to(device)

    print(f"Collected Actions (before update): Mean={actions_raw.mean().item():.8f} (after sigmoid), Std={actions_raw.std().item():.8f} (after sigmoid)")
    print(f"Rewards: Mean={rewards.mean().item():.8f}, Std={rewards.std().item():.8f}")

    adv = rewards

    for _ in range(epochs):
        action_pred, logp_new_raw, dist_new = actor(states)
        logp = dist_new.log_prob(actions_raw).sum(dim=-1)
        ratio = torch.exp(torch.clamp(logp - logp_old, -10., 10.))

        surr1 = ratio * adv
        surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
        actor_loss = -torch.min(surr1, surr2).mean()

        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()

    if csv_writer is not None:
        actions_mean = actions_raw.mean().item()
        actions_std = actions_raw.std().item()
        rewards_mean = rewards.mean().item()
        rewards_std = rewards.std().item()
        csv_writer.writerow([actions_mean, actions_std, rewards_mean, rewards_std])
        
    if hasattr(ppo_update, 'update_count'):
        ppo_update.update_count += 1
    else:
        ppo_update.update_count = 1
        
    if save_path_prefix:
        torch.save(actor.state_dict(), f"{save_path_prefix}policy_weights_update_{ppo_update.update_count}.pt")

torch.manual_seed(42)
inference_dtype = torch.float16
device1 = "cuda:0"
device2 = "cuda:1"
pipeline = StableDiffusionXLPipeline.from_pretrained(SDXL_TURBO_MODEL_PATH, revision="main", torch_dtype=torch.float16)

pipeline.vae.requires_grad_(False)
pipeline.text_encoder.requires_grad_(False)
pipeline.text_encoder_2.requires_grad_(False)
pipeline.unet.requires_grad_(False)

pipeline.vae.to(device2, dtype=inference_dtype)
pipeline.text_encoder.to(device2, dtype=inference_dtype)
pipeline.text_encoder_2.to(device2, dtype=inference_dtype)
pipeline.unet.to(device2, dtype=inference_dtype)    
pipeline.to(device2)

pipeline.safety_checker = None

def main(latents, file_path, prompts1, prompts2, prompts1_, prompts2_):
    actor = PolicyNet(embedding_dim=2048, hidden_dim=2048*2).to(dtype=inference_dtype, device=device1)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=3e-6, eps=1e-4)

    csv_file = open(file_path + 'training_log.csv', 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['actions_mean', 'actions_std', 'rewards_mean', 'rewards_std'])

    reward_fn = creative_reward_fn()

    states_buffer = []
    actions_raw_buffer = []
    log_probs_old_buffer = []
    rewards_buffer = []

    max_steps = 128
    update_interval = 16

    image1 = Image.open(DATASET_DIR / prompts1_ / "output_0.png")
    image2 = Image.open(DATASET_DIR / prompts2_ / "output_0.png")
    f1 = get_image_features_clip(image1).to(device1)
    f2 = get_image_features_clip(image2).to(device1)
    t1, t2 = get_text_features(prompts1, prompts2)

    c11, c12, c21, c22 = None, None, None, None
    with torch.no_grad():
        c11_dev2, _, c12_dev2, _ = pipeline.encode_prompt(prompt=prompts1, num_images_per_prompt=1, do_classifier_free_guidance=False, device=device2)
        c21_dev2, _, c22_dev2, _ = pipeline.encode_prompt(prompt=prompts2, num_images_per_prompt=1, do_classifier_free_guidance=False, device=device2)

        c11_dev1 = c11_dev2.to(device1)
        c21_dev1 = c21_dev2.to(device1)
        c12_dev1 = c12_dev2.to(device1)
        c22_dev1 = c22_dev2.to(device1)

    current_state = c11_dev1 * 0.5 + c21_dev1 * 0.5

    print(f"Generate {prompts1_} & {prompts2_} image")
    for step in tqdm(range(max_steps)):
        current_state = current_state.to(device1, dtype=inference_dtype)

        with torch.no_grad():
            action_raw, log_prob_raw, dist = actor(current_state)

        states_buffer.append(current_state.cpu())
        actions_raw_buffer.append(action_raw.cpu())
        log_probs_old_buffer.append(log_prob_raw.cpu())

        action_for_mixing = action_raw.unsqueeze(1).expand(-1, 77, -1)

        c11_dev2 = c11_dev1.to(device2, dtype=inference_dtype)
        c21_dev2 = c21_dev1.to(device2, dtype=inference_dtype)
        action_for_mixing_dev2 = action_for_mixing.to(device2, dtype=inference_dtype)

        prompt_embeds_for_diffusion = c11_dev2 * action_for_mixing_dev2 + c21_dev2 * (1 - action_for_mixing_dev2)
        pooled_prompt_embeds_for_diffusion = (c12_dev1 * 0.5 + c22_dev1 * 0.5).to(device2, dtype=inference_dtype)

        latent0 = self_pipe(latents, pipeline, device=device2, prompt_embeds=prompt_embeds_for_diffusion, pooled_prompt_embeds=pooled_prompt_embeds_for_diffusion)
        image_tensor = pipeline.vae.decode(latent0 / pipeline.vae.config.scaling_factor)[0].to(device1)
        image_good = xl_postprocess(image_tensor)[0]
        im0 = image_good.copy()

        reward, sim1, sim2 = reward_fn(image_good, f1, f2, t1, t2)
        rewards_buffer.append(reward.cpu())

        current_state = prompt_embeds_for_diffusion.detach().to(device1)

        print(f"\nStep {step+1}: Reward={reward.item():.4f}, Sim1={sim1.item():.4f}, Sim2={sim2.item():.4f}")
        im0.save(file_path + f"{step+1:04d}_{reward.item():.4f}_{sim1.item():.4f}_{sim2.item():.4f}.png")

        if len(rewards_buffer) >= update_interval:
            print(f"\nPerforming PPO update after {len(rewards_buffer)} steps...")
            states_batch = torch.stack(states_buffer)
            actions_raw_batch = torch.stack(actions_raw_buffer)
            log_probs_old_batch = torch.stack(log_probs_old_buffer)
            rewards_batch = torch.stack(rewards_buffer)


            ppo_update(
                actor,
                actor_optimizer,
                states_batch,
                actions_raw_batch,
                log_probs_old_batch,
                rewards_batch,
                device=device1,
                csv_writer=csv_writer,
                save_path_prefix=file_path
            )

            states_buffer = []
            actions_raw_buffer = []
            log_probs_old_buffer = []
            rewards_buffer = []
            print("PPO update complete. Buffers cleared.")

    if len(rewards_buffer) > 0:
         print(f"\nPerforming final PPO update with {len(rewards_buffer)} remaining steps...")
         states_batch = torch.stack(states_buffer)
         actions_raw_batch = torch.stack(actions_raw_buffer)
         log_probs_old_batch = torch.stack(log_probs_old_buffer)
         rewards_batch = torch.stack(rewards_buffer)

         ppo_update(
             actor,
             actor_optimizer,
             states_batch,
             actions_raw_batch,
             log_probs_old_batch,
             rewards_batch,
             device=device1,
             csv_writer=csv_writer,
             save_path_prefix=file_path
         )
         print("Final PPO update complete.")


    torch.cuda.empty_cache()
    csv_file.close()

if __name__ == "__main__":
    dataset_path = DATASET_DIR
    floder_path = OUTPUT_DIR / "res_ppo_522_seg" / "2-baseloss-save"
    folders = [folder for folder in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, folder))]

    list1 = ["zebra", "frog", "owl", "dinosaur", "giraffe", "cat", "zebra", "giraffe", "cauliflower", "komodo dragon"]
    list2 = ["rabbit", "cauliflower", "tiger", "strawberry", "snail", "armadillo", "rabbit", "cock", "eagle", "howler monkey"]
    
    for i in range(1):
        prompts1_, prompts2_ = random.sample(folders, 2)
        prompts1_ = list1[i]
        prompts2_ = list2[i]
        prompts1 = "A photo of a full-body " + prompts1_
        prompts2 = "A photo of a full-body " + prompts2_
        
        if os.path.exists(floder_path / f"{prompts1_}&{prompts2_}"):
            continue
        
        os.makedirs(floder_path, exist_ok=True)
        os.makedirs(floder_path / f"{prompts1_}&{prompts2_}", exist_ok=True)
        file_path = str(floder_path / f"{prompts1_}&{prompts2_}") + os.sep

        latents = pipeline.prepare_latents(
            1,
            4,
            512,
            512,
            dtype=inference_dtype,
            device=device2,
            generator=None,
            latents=None,
        )
        
        main(latents, file_path, prompts1, prompts2, prompts1_, prompts2_)
