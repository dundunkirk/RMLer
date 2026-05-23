# RMLer Technical Implementation

This branch focuses on the technical implementation of RMLer. It describes the code organization, path configuration, training/inference scripts, and the reinforcement mixing pipeline.

## Method Pipeline

RMLer formulates cross-category concept fusion as a reinforcement learning problem for text-to-image generation.

1. Encode two source prompts with a Stable Diffusion text encoder.
2. Build the current state from mixed text embeddings.
3. Use an MLP policy network to predict adaptive mixing coefficients.
4. Generate a fused image from the mixed prompt embeddings.
5. Evaluate the image with visual rewards based on source-concept similarity and balance.
6. Optimize the policy network with PPO.
7. Select high-reward generated samples during inference or post-processing.

## Core Components

### Policy Network

The policy network is implemented as an MLP that maps flattened prompt embeddings to a mixing action. The action is expanded across token dimensions and used to combine the two source prompt embeddings:

```python
prompt_embeds = embed_a * action + embed_b * (1 - action)
```

The policy samples actions from a Normal distribution and stores log probabilities for PPO updates.

### Reward Function

The reward compares the generated image feature with the two source concept features. The current implementation uses CLIP image/text features and a balance penalty so that the generated object is encouraged to preserve both concepts instead of collapsing to only one.

### PPO Update

Collected states, actions, old log probabilities, and rewards are buffered during generation. After a fixed interval, PPO updates the policy network with clipped policy ratios.

### Diffusion Backbones

The repository includes training scripts for multiple diffusion backbones:

- SDXL-Turbo
- Stable Diffusion v1.4
- Stable Diffusion v1.5
- Stable Diffusion v2.1

## Project Structure

```text
RMLer/
|-- checkpoints/
|-- data/
|   |-- dataset/
|   `-- metadata/
|       |-- Cretok.txt
|       `-- ImageNet-200.txt
|-- docs/
|-- outputs/
|-- prompts/
|-- rmler/
|   |-- __init__.py
|   |-- metric/
|   `-- pipeline/
|-- scripts/
|   |-- train/
|   |   |-- save.py
|   |   |-- pipeline-ppo-sd-v1-4.py
|   |   |-- pipeline-ppo-sd-v1-5.py
|   |   `-- pipeline-ppo-sd-v2-1.py
|   |-- infer/
|   |   `-- save_infer.py
|   |-- eval/
|   |   |-- 0-param-hpsv2.py
|   |   `-- 0-param-vqascore.py
|   `-- postprocess/
|       `-- 0-select_top_k_0801.py
`-- README.md
```

## Path Configuration

Scripts resolve paths relative to the repository root by default.

Default locations:

- `checkpoints/`: local model checkpoints
- `data/dataset/`: source concept images
- `prompts/prompt.txt`: prompt pairs
- `outputs/`: generated images, logs, policy weights, and selected samples

Environment variables can override these defaults:

| Variable | Purpose |
| --- | --- |
| `RMLER_CHECKPOINT_DIR` | Base directory for local checkpoints |
| `RMLER_DATASET_DIR` | Dataset directory |
| `RMLER_OUTPUT_DIR` | Output directory |
| `RMLER_PROMPT_FILE` | Prompt-pair file |
| `RMLER_CLIP_MODEL` | CLIP model path |
| `RMLER_SDXL_TURBO_MODEL` | SDXL-Turbo model path |
| `RMLER_SD14_MODEL` | Stable Diffusion v1.4 model path |
| `RMLER_SD15_MODEL` | Stable Diffusion v1.5 model path |
| `RMLER_SD21_MODEL` | Stable Diffusion v2.1 model path |
| `RMLER_POLICY_WEIGHTS` | Policy checkpoint used for inference |

## Script Entry Points

### Training

```bash
python scripts/train/save.py
python scripts/train/pipeline-ppo-sd-v1-4.py
python scripts/train/pipeline-ppo-sd-v1-5.py
python scripts/train/pipeline-ppo-sd-v2-1.py
```

### Inference

```bash
python scripts/infer/save_infer.py
```

### Evaluation

```bash
python scripts/eval/0-param-hpsv2.py
python scripts/eval/0-param-vqascore.py
```

### Post-processing

```bash
python scripts/postprocess/0-select_top_k_0801.py
```

## Expected Input Layout

For reward computation, source concept images are expected in:

```text
data/dataset/
|-- concept_a/
|   `-- output_0.png
`-- concept_b/
    `-- output_0.png
```

For SD v1.4, v1.5, and v2.1 scripts, prompt pairs are expected in:

```text
prompts/prompt.txt
```

Each line should contain one pair separated by `&`, for example:

```text
zebra&rabbit
cat&frog
```

## Output Layout

Generated samples and PPO artifacts are written under `outputs/` by default.

Typical generated files include:

- `training_log.csv`
- `policy_weights_update_*.pt`
- generated images named with step, reward, and similarity values

## Notes

- The scripts assume CUDA devices are available and currently use `cuda:0` and `cuda:1` in several places.
- Large model checkpoints are not included in this repository.
- Some hyperparameters, prompt lists, and concept pairs are still script-level settings and can be adjusted directly in the corresponding entry-point files.
