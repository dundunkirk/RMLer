import t2v_metrics.t2v_metrics as t2v_metrics
import os
import statistics
import glob
from pathlib import Path
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = Path(os.environ.get("RMLER_OUTPUT_DIR", str(PROJECT_ROOT / "outputs")))

clip_flant5_score = t2v_metrics.VQAScore(model='clip-flant5-xxl') # our recommended scoring model

def extract_concepts(folder_name):
    """Extract concepts from folder name (e.g., 'Alpaca&Lion' or 'Alpaca_Lion')"""
    if '&' in folder_name:
        return folder_name.split("&")
    elif '_' in folder_name:
        return folder_name.split("_")
    else:
        return [folder_name, ""] # fallback case

def create_prompt(concepts):
    """Create a text prompt from concepts"""
    a, b = concepts
    return f"A photo of one full-body hybrid of {a} and {b}"

def calculate_score(folder_path):
    """Calculate score for the last 10 images in a folder"""
    # Extract folder name from path
    folder_name = os.path.basename(folder_path)
    
    # Extract concepts and create prompt
    concepts = extract_concepts(folder_name)
    text = create_prompt(concepts)
    
    # Find the images in the folder
    image_extensions = ['*.jpg', '*.jpeg', '*.png']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(folder_path, ext)))
    
    # Sort images by filename and select the last 10
    image_files = sorted(image_files)[-10:]
    
    if not image_files:
        print(f"No images found in {folder_path}")
        return None, text
    
    scores = []
    for image in image_files:
        try:
            score = clip_flant5_score(images=[image], texts=[text])
            scores.append(float(score))
        except Exception as e:
            print(f"Error scoring {image}: {str(e)}")
    
    # Calculate the average score for the last 10 images
    if scores:
        average_score = round(statistics.mean(scores), 4)
    else:
        average_score = None
    
    return average_score, text

# Main execution
base_dir = Path(os.environ.get("RMLER_EVAL_DIR", str(OUTPUT_DIR / "res_ppo_1011_canjie")))
results = {}
all_scores = []

# Check if the directory exists
if not os.path.exists(base_dir):
    print(f"Directory not found: {base_dir}")
else:
    # Get all subdirectories
    subdirs = [os.path.join(base_dir, d) for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    
    for subdir in tqdm(subdirs, desc="Processing folders with VQA"):
        score, prompt = calculate_score(subdir)
        folder_name = os.path.basename(subdir)
        results[folder_name] = {
            "prompt": prompt,
            "score": score
        }
        
        if score is not None:
            all_scores.append(score)
    
    # Calculate the average score across all folders
    average_overall_score = statistics.mean(all_scores) if all_scores else None
    
    # Print results
    print("\nVQA Results:")
    print("=" * 50)
    for folder, data in results.items():
        print(f"Folder: {folder}")
        print(f"Prompt: {data['prompt']}")
        print(f"VQA Score: {data['score']}")
        print("-" * 50)
    print(f"\nPath:{base_dir}")
    print(f"\nOverall Average VQA Score: {round(average_overall_score, 4) if average_overall_score is not None else 'N/A'}")

######################### canjie
# conceptlab 0.3444
# gpt 0.5185
# our 0.4287
# sdxl 0.2439
# TP2O 0.3069

# Path: outputs/critic-canjie
# Overall Average VQA Score: 0.4155
# our 0.4287
# Path: outputs/grpo-canjie
# Overall Average VQA Score: 0.4073

########################### Our dataset

# conceptlab   0.2671
# gpt 0.4729
# out 0.3301
# sdxl 0.1875
# tp2o 0.3055
