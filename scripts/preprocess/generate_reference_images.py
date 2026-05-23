import argparse
import os
from pathlib import Path

import torch
from diffusers import StableDiffusionXLPipeline
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_DIR = Path(os.environ.get("RMLER_CHECKPOINT_DIR", str(PROJECT_ROOT / "checkpoints")))
DEFAULT_PROMPT_FILE = Path(os.environ.get("RMLER_PROMPT_FILE", str(PROJECT_ROOT / "prompts" / "Cretok.txt")))
DEFAULT_DATASET_DIR = Path(os.environ.get("RMLER_DATASET_DIR", str(PROJECT_ROOT / "dataset")))
DEFAULT_MODEL_PATH = os.environ.get("RMLER_SDXL_TURBO_MODEL", str(CHECKPOINT_DIR / "sdxl-turbo"))


def parse_concepts(prompt_file: Path) -> list[str]:
    concepts = set()
    with prompt_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "&" not in line:
                continue
            left, right = [part.strip() for part in line.split("&", 1)]
            if left:
                concepts.add(left)
            if right:
                concepts.add(right)
    return sorted(concepts)


def build_prompt(concept: str, template: str) -> str:
    return template.format(concept=concept)


def generate_reference_images(args: argparse.Namespace) -> None:
    prompt_file = Path(args.prompt_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    concepts = parse_concepts(prompt_file)
    if args.limit is not None:
        concepts = concepts[: args.limit]

    if not concepts:
        raise ValueError(f"No prompt pairs found in {prompt_file}")

    pipe = StableDiffusionXLPipeline.from_pretrained(
        args.model_path,
        revision=args.revision,
        torch_dtype=torch.float16 if args.dtype == "float16" else torch.float32,
    )
    pipe.to(args.device)
    pipe.safety_checker = None

    generator = None
    if args.seed is not None:
        generator = torch.Generator(device=args.device).manual_seed(args.seed)

    for concept in tqdm(concepts, desc="Generating reference images"):
        concept_dir = output_dir / concept
        image_path = concept_dir / "output_0.png"
        if image_path.exists() and not args.overwrite:
            continue

        concept_dir.mkdir(parents=True, exist_ok=True)
        prompt = build_prompt(concept, args.prompt_template)
        image = pipe(
            prompt=prompt,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            generator=generator,
        ).images[0]
        image.save(image_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate reference images from RMLer prompt-pair files.")
    parser.add_argument("--prompt-file", default=str(DEFAULT_PROMPT_FILE), help="Prompt-pair file, e.g. prompts/Cretok.txt")
    parser.add_argument("--output-dir", default=str(DEFAULT_DATASET_DIR), help="Output dataset directory")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH, help="SDXL-Turbo model path")
    parser.add_argument("--revision", default="main", help="Model revision")
    parser.add_argument("--device", default="cuda:0", help="Torch device")
    parser.add_argument("--dtype", choices=["float16", "float32"], default="float16", help="Pipeline dtype")
    parser.add_argument("--prompt-template", default="A photo of a full-body {concept}", help="Template for single-concept prompts")
    parser.add_argument("--num-inference-steps", type=int, default=4, help="Diffusion inference steps")
    parser.add_argument("--guidance-scale", type=float, default=0.0, help="Classifier-free guidance scale")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--limit", type=int, default=None, help="Generate only the first N unique concepts")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output_0.png files")
    args = parser.parse_args()

    generate_reference_images(args)


if __name__ == "__main__":
    main()
