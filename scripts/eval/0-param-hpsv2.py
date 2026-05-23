import hpsv2
import os
import statistics
import glob
from pathlib import Path
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = Path(os.environ.get("RMLER_OUTPUT_DIR", str(PROJECT_ROOT / "outputs")))

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
    prompt = create_prompt(concepts)
    
    # Find the images in the folder
    image_extensions = ['*.jpg', '*.jpeg', '*.png']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(folder_path, ext)))
    
    # Sort images by filename and select the last 10
    image_files = sorted(image_files)[-10:]
    
    if not image_files:
        print(f"No images found in {folder_path}")
        return None, prompt
    
    scores = []
    for image in image_files:
        try:
            result = hpsv2.score(image, prompt, hps_version="v2.0")
            
            # Handle result based on its type
            if isinstance(result, dict):
                score = result['score']
            elif isinstance(result, list) and len(result) > 0:
                if isinstance(result[0], dict) and 'score' in result[0]:
                    score = result[0]['score']
                else:
                    score = result[0]
            else:
                print(f"Unexpected result format for {image}: {result}")
                continue
            
            scores.append(float(score))
        except Exception as e:
            print(f"Error scoring {image}: {str(e)}")
            print(f"Result was: {result}")
    
    # Calculate the average score for the last 10 images
    if scores:
        average_score = round(statistics.mean(scores), 4)
    else:
        average_score = None
    
    return average_score, prompt

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
    
    for subdir in tqdm(subdirs, desc="Processing folders with HPSv2"):
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
    print("\nHPSv2 Results:")
    print("=" * 50)
    for folder, data in results.items():
        print(f"Folder: {folder}")
        print(f"Prompt: {data['prompt']}")
        print(f"HPSv2 Score: {data['score']}")
        print("-" * 50)
    
    print(f"\nPath:{base_dir}")
    print(f"\nOverall Average HPSv2 Score: {round(average_overall_score, 4) if average_overall_score is not None else 'N/A'}")


######################### hpsv2.0
# con 0.2714
# gpt 0.2971
# our 0.2774
# sdxl 0.2886
# TP2O 0.2750


##############
# Path: outputs/critic-canjie
# Overall Average HPSv2 Score: 0.2746

# our 0.2774

# Path: outputs/grpo-canjie

# Overall Average HPSv2 Score: 0.2744

'''
hpsv2.1

our 2.44
tp20 0.2579


hpsv2.0
our 0.2737
tp2o 0.2756


'''
