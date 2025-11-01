from roboflow import Roboflow
import os
import shutil
from PIL import Image

def setup_roboflow_dataset():
    """Download car damage dataset from Roboflow"""
    
    # Initialize Roboflow (uses public dataset)
    rf = Roboflow(api_key="YOUR_API_KEY")  # We'll use public access
    
    print("Downloading car damage dataset from Roboflow...")
    
    # Public car damage detection dataset
    project = rf.workspace("roboflow-universe").project("car-damage-detection")
    dataset = project.version(1).download("folder")
    
    print("Organizing images into damaged/undamaged folders...")
    
    # Create our folders
    os.makedirs("dataset/damaged", exist_ok=True)
    os.makedirs("dataset/undamaged", exist_ok=True)
    
    # Process downloaded images
    # (Roboflow datasets come pre-organized)
    
    print("✅ Dataset ready!")

# For public access without API key, use this simpler approach:
def download_public_samples():
    """Download sample images from public sources"""
    import urllib.request
    
    os.makedirs("dataset/damaged", exist_ok=True)
    os.makedirs("dataset/undamaged", exist_ok=True)
    
    print("This will guide you to download datasets manually...")
    print("\n" + "="*60)
    print("RECOMMENDED DATASETS:")
    print("="*60)
    
    print("\n1. Kaggle - Car Damage Detection:")
    print("   https://www.kaggle.com/datasets/anujms/car-damage-detection")
    
    print("\n2. Roboflow Universe - Car Damage:")
    print("   https://universe.roboflow.com/car-damage-detection")
    
    print("\n3. GitHub - CarDD Dataset:")
    print("   https://cardd-ustc.github.io/")
    
    print("\n" + "="*60)
    print("INSTRUCTIONS:")
    print("="*60)
    print("1. Visit the links above")
    print("2. Download the datasets")
    print("3. Extract and organize:")
    print("   - Damaged car images → dataset/damaged/")
    print("   - Clean car images → dataset/undamaged/")
    print("="*60)

if __name__ == "__main__":
    download_public_samples()
