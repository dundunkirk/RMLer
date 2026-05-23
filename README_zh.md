## RMLer

论文: [arXiv:2512.19300](https://arxiv.org/pdf/2512.19300)

当前仓库已为开源发布做了轻量级整理（仅目录结构与 README 调整，脚本逻辑未改动）。

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

## 说明

- 脚本现在默认基于仓库根目录解析路径。
- 本地模型权重可放在 `checkpoints/`，数据集可放在 `data/dataset/`，prompt 文件可放在 `prompts/`，生成结果默认写入 `outputs/`。
- 如需使用服务器上的其他位置，可以通过环境变量覆盖默认路径，例如 `RMLER_CHECKPOINT_DIR`、`RMLER_DATASET_DIR`、`RMLER_OUTPUT_DIR`、`RMLER_PROMPT_FILE`、`RMLER_CLIP_MODEL`、`RMLER_SDXL_TURBO_MODEL`、`RMLER_SD14_MODEL`、`RMLER_SD15_MODEL`、`RMLER_SD21_MODEL` 和 `RMLER_POLICY_WEIGHTS`。
