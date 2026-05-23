## RMLer: Synthesizing Novel Objects Across Diverse Categories via Reinforcement Mixing Learning (AAAI 2026, Oral)

The official implementation of our work: **RMLer**.

[![AAAI 2026](https://img.shields.io/badge/AAAI%202026-Oral-red)](https://ojs.aaai.org/index.php/AAAI/article/view/37552)
[![arXiv](https://img.shields.io/badge/arXiv-2512.19300-b31b1b.svg)](https://arxiv.org/abs/2512.19300)
[![Paper](https://img.shields.io/badge/Paper-AAAI-blue)](https://ojs.aaai.org/index.php/AAAI/article/view/37552)

Jun Li<sup>1*</sup>, Zikun Chen<sup>1</sup>, Haibo Chen<sup>1*</sup>, Shuo Chen<sup>2</sup>, Jian Yang<sup>2</sup>

<sup>1</sup> Nanjing University of Science and Technology, China  
<sup>2</sup> Nanjing University, China

<sup>*</sup> Corresponding authors

## Contents

- [News](#news)
- [Abstract](#abstract)
- [Overview](#overview)
- [Project Structure](#project-structure)
- [Main Script Entry Points](#main-script-entry-points)
- [Citation](#citation)
- [Notes](#notes)

## News

- 2026-03: RMLer was published in the Proceedings of the AAAI Conference on Artificial Intelligence.
- 2026-02: RMLer was accepted by AAAI 2026 as an Oral paper.
- 2025-12: The arXiv version was released.

## Abstract

RMLer targets the problem of generating a single novel object from two semantically different concepts. Instead of relying on fixed prompt interpolation or simple visual juxtaposition, RMLer learns how to mix cross-category text embeddings through reinforcement learning. A policy network predicts adaptive mixing coefficients, while rewards encourage the generated object to preserve both source concepts in a balanced and semantically meaningful way. This design helps produce coherent fused objects with stronger concept integration and better visual quality.

## Overview

RMLer is a reinforcement learning framework for novel object synthesis in text-to-image generation. Given two concepts from different categories, the goal is to synthesize a single coherent object that meaningfully blends both concepts instead of producing an imbalanced, superficial, or side-by-side composition.

Our method formulates cross-category concept fusion as a reinforcement learning problem:

- **States** are mixed text-embedding features.
- **Actions** are dynamic mixing coefficients predicted by an MLP policy network.
- **Rewards** evaluate the generated visual result using semantic similarity and compositional balance with respect to the source concepts.

The policy is optimized with proximal policy optimization (PPO). At inference time, RMLer uses reward-based selection to keep high-quality fused objects. The framework is designed for synthesizing coherent, high-fidelity novel visual concepts, with applications in creative design, games, film, and digital content creation.

## Highlights

- Reinforcement learning formulation for cross-category concept fusion.
- Dynamic text-embedding mixing through an MLP policy network.
- Reward design that considers both semantic relevance and balanced concept composition.
- Support for SDXL-Turbo, Stable Diffusion v1.4, v1.5, and v2.1 training scripts.
- Evaluation utilities for HPSv2 and VQAScore.


## Project Structure

```text
RMLer/
|-- data/
|   |-- dataset/
|   `-- metadata/
|       |-- Cretok.txt
|       `-- ImageNet-200.txt
|-- checkpoints/
|-- docs/
|-- outputs/
|-- prompts/
|-- rmler/
|   |-- __init__.py
|   |-- metric/
|   `-- pipeline/
|-- scripts/
|   |-- train/
|   |   |-- pipeline-ppo-sd-v1-4.py
|   |   |-- pipeline-ppo-sd-v1-5.py
|   |   |-- pipeline-ppo-sd-v2-1.py
|   |   `-- save.py
|   |-- infer/
|   |   `-- save_infer.py
|   |-- eval/
|   |   |-- 0-param-hpsv2.py
|   |   `-- 0-param-vqascore.py
|   `-- postprocess/
|       `-- 0-select_top_k_0801.py
`-- README.md
```

## Main Script Entry Points

- Training
  - `python scripts/train/save.py`
  - `python scripts/train/pipeline-ppo-sd-v1-4.py`
  - `python scripts/train/pipeline-ppo-sd-v1-5.py`
  - `python scripts/train/pipeline-ppo-sd-v2-1.py`
- Inference
  - `python scripts/infer/save_infer.py`
- Evaluation
  - `python scripts/eval/0-param-hpsv2.py`
  - `python scripts/eval/0-param-vqascore.py`
- Post-processing
  - `python scripts/postprocess/0-select_top_k_0801.py`

## Citation

If you find this work useful, please consider citing:

```bibtex
@inproceedings{li2026rmler,
  title={RMLer: Synthesizing Novel Objects Across Diverse Categories via Reinforcement Mixing Learning},
  author={Li, Jun and Chen, Zikun and Chen, Haibo and Chen, Shuo and Yang, Jian},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={40},
  number={8},
  pages={6262--6270},
  year={2026},
  doi={10.1609/aaai.v40i8.37552}
}
```

## Notes

- Scripts now resolve paths relative to the repository root by default.
- Put local model checkpoints under `checkpoints/`, datasets under `data/dataset/`, prompts under `prompts/`, and generated results under `outputs/`.
- You can override these defaults with environment variables such as `RMLER_CHECKPOINT_DIR`, `RMLER_DATASET_DIR`, `RMLER_OUTPUT_DIR`, `RMLER_PROMPT_FILE`, `RMLER_CLIP_MODEL`, `RMLER_SDXL_TURBO_MODEL`, `RMLER_SD14_MODEL`, `RMLER_SD15_MODEL`, `RMLER_SD21_MODEL`, and `RMLER_POLICY_WEIGHTS`.
