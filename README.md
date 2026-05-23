## RMLer

Paper: [arXiv:2512.19300](https://arxiv.org/pdf/2512.19300)

This repository has been lightly reorganized for open-source release (directory layout and README only; script logic is unchanged).

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

## Notes

- Scripts now resolve paths relative to the repository root by default.
- Put local model checkpoints under `checkpoints/`, datasets under `data/dataset/`, prompts under `prompts/`, and generated results under `outputs/`.
- You can override these defaults with environment variables such as `RMLER_CHECKPOINT_DIR`, `RMLER_DATASET_DIR`, `RMLER_OUTPUT_DIR`, `RMLER_PROMPT_FILE`, `RMLER_CLIP_MODEL`, `RMLER_SDXL_TURBO_MODEL`, `RMLER_SD14_MODEL`, `RMLER_SD15_MODEL`, `RMLER_SD21_MODEL`, and `RMLER_POLICY_WEIGHTS`.
