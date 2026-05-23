import os
import shutil
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = Path(os.environ.get("RMLER_OUTPUT_DIR", str(PROJECT_ROOT / "outputs")))

def select_and_copy_top_k_per_class(root_dir, dest_dir, top_k=10):
    """
    对每个类别（即每个子目录）内，选择符合条件的 reward 前 top_k 张图像复制到目标目录。
    文件名格式需为: reward_num1_num2_num3.png
    条件：|num1 - num2| < 0.05 且 num1 > 0.6 且 num2 > 0.6
    """
    for subdir, _, files in os.walk(root_dir):
        candidates = []
        for file in files:
            if file.endswith(".png"):
                match = re.match(r"^([\d.]+)_([\d.]+)_([\d.]+)_([\d.]+)\.png$", file)
                if match:
                    reward = float(match.group(2))
                    num1 = float(match.group(3))
                    num2 = float(match.group(4))

                    if abs(num1 - num2) < 0.05: # and num1 > 0.6 and num2 > 0.6:
                        src_path = os.path.join(subdir, file)
                        candidates.append((reward, src_path, file))

        # 如果有合格图像，则处理
        if candidates:
            top_candidates = sorted(candidates, key=lambda x: x[0], reverse=True)[:top_k]

            # 目标子目录结构
            relative_path = os.path.relpath(subdir, root_dir)
            dest_subdir = os.path.join(dest_dir, relative_path)
            os.makedirs(dest_subdir, exist_ok=True)

            for reward, src_path, file in top_candidates:
                dest_path = os.path.join(dest_subdir, file)
                shutil.copy2(src_path, dest_path)
                print(f"Copied: {file} (reward={reward:.3f}) → {dest_subdir}")

    print(f"Top-{top_k} images per class have been copied.")

if __name__ == "__main__":
    source_directory = Path(os.environ.get("RMLER_SELECT_SOURCE_DIR", str(OUTPUT_DIR / "grpo-canjie")))
    destination_directory = Path(os.environ.get("RMLER_SELECT_DEST_DIR", str(OUTPUT_DIR / "selected" / "grpo-canjie")))
    select_and_copy_top_k_per_class(source_directory, destination_directory, top_k=10)
    print("Image selection and copying complete.")
