## RMLer: Synthesizing Novel Objects Across Diverse Categories via Reinforcement Mixing Learning (AAAI 2026, Oral)

本文是 **RMLer** 的官方代码实现。

[![AAAI 2026](https://img.shields.io/badge/AAAI%202026-Oral-red)](https://ojs.aaai.org/index.php/AAAI/article/view/37552)
[![arXiv](https://img.shields.io/badge/arXiv-2512.19300-b31b1b.svg)](https://arxiv.org/abs/2512.19300)
[![Paper](https://img.shields.io/badge/Paper-AAAI-blue)](https://ojs.aaai.org/index.php/AAAI/article/view/37552)

Jun Li<sup>1*</sup>, Zikun Chen<sup>1</sup>, Haibo Chen<sup>1*</sup>, Shuo Chen<sup>2</sup>, Jian Yang<sup>2</sup>

<sup>1</sup> 南京理工大学  
<sup>2</sup> 南京大学

<sup>*</sup> 通讯作者

## 目录

- [最新动态](#最新动态)
- [摘要](#摘要)
- [工作简介](#工作简介)
- [项目结构](#项目结构)
- [主要脚本入口](#主要脚本入口)
- [引用](#引用)
- [说明](#说明)

## 最新动态

- 2026-03: RMLer 论文发表在 AAAI Conference on Artificial Intelligence 会议论文集中。
- 2026-02: RMLer 被 AAAI 2026 接收为 Oral 论文。
- 2025-12: arXiv 版本发布。

## 摘要

RMLer 面向“由两个语义差异较大的概念生成一个新颖物体”的任务。不同于固定 prompt 插值或简单的视觉并置，RMLer 通过强化学习学习跨类别文本嵌入的混合方式。策略网络预测自适应混合系数，奖励函数则鼓励生成结果同时保留两个源概念，并保持语义平衡与视觉一致性。通过这种设计，RMLer 能够生成概念融合更充分、结构更连贯、视觉质量更高的新颖物体。

## 工作简介

RMLer 是一个面向文本到图像生成的新颖物体合成框架。给定来自不同类别的两个概念，RMLer 的目标是生成一个语义连贯、视觉自然、同时融合两者特征的新物体，而不是简单拼接、偏向单一概念或生成表面化组合。

我们将跨类别概念融合建模为一个强化学习问题：

- **状态**：由两个概念的文本嵌入混合得到的特征表示。
- **动作**：由 MLP 策略网络预测的动态混合系数。
- **奖励**：根据生成图像与源概念之间的语义相似度和组合平衡性来评估结果质量。

策略网络通过 PPO 进行优化。在推理阶段，RMLer 使用基于奖励的筛选机制保留高质量融合结果。该方法适用于创意设计、游戏、影视和数字内容创作等需要新颖视觉概念生成的场景。

## 方法亮点

- 将跨类别概念融合建模为强化学习问题。
- 使用 MLP 策略网络动态混合文本嵌入。
- 奖励函数同时考虑语义相关性与概念组合平衡性。
- 提供 SDXL-Turbo、Stable Diffusion v1.4、v1.5 和 v2.1 的训练脚本。
- 提供 HPSv2 和 VQAScore 评估脚本。

## 项目结构

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

## 主要脚本入口

- 训练
  - `python scripts/train/save.py`
  - `python scripts/train/pipeline-ppo-sd-v1-4.py`
  - `python scripts/train/pipeline-ppo-sd-v1-5.py`
  - `python scripts/train/pipeline-ppo-sd-v2-1.py`
- 推理
  - `python scripts/infer/save_infer.py`
- 评估
  - `python scripts/eval/0-param-hpsv2.py`
  - `python scripts/eval/0-param-vqascore.py`
- 后处理
  - `python scripts/postprocess/0-select_top_k_0801.py`

## 引用

如果本工作对你的研究有帮助，请考虑引用：

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

## 说明

- 脚本现在默认基于仓库根目录解析路径。
- 本地模型权重可放在 `checkpoints/`，数据集可放在 `data/dataset/`，prompt 文件可放在 `prompts/`，生成结果默认写入 `outputs/`。
- 如需使用服务器上的其他位置，可以通过环境变量覆盖默认路径，例如 `RMLER_CHECKPOINT_DIR`、`RMLER_DATASET_DIR`、`RMLER_OUTPUT_DIR`、`RMLER_PROMPT_FILE`、`RMLER_CLIP_MODEL`、`RMLER_SDXL_TURBO_MODEL`、`RMLER_SD14_MODEL`、`RMLER_SD15_MODEL`、`RMLER_SD21_MODEL` 和 `RMLER_POLICY_WEIGHTS`。
