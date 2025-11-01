import os
import zipfile
import shutil
from pathlib import Path

def download_and_setup_dataset():
    """Download car damage dataset from Kaggle"""
    
    print("📥 Downloading car damage dataset from Kaggle...")
    
    # Download dataset
    os.system("kaggle datasets download -d anujms/car-damage-detection")
    
    print("📦 Extracting dataset...")
    
    # Extract zip
    with zipfile.ZipFile("car-damage-detection.zip", 'r') as zip_ref:
        zip_ref.extractall("temp_dataset")
    
    # Create our folder structure
    os.makedirs("dataset/damaged", exist_ok=True)
    os.makedirs("dataset/undamaged", exist_ok=True)
    
    print("📁 Organizing images...")
    
    # Move damaged images
    damaged_source = Path("temp_dataset/data/training/damaged")
    if damaged_source.exists():
        for img in damaged_source.glob("*.jpg"):
            shutil.copy(img, "dataset/damaged/")
        for img in damaged_source.glob("*.png"):
            shutil.copy(img, "dataset/damaged/")
    
    # Move undamaged images
    whole_source = Path("temp_dataset/data/training/whole")
    if whole_source.exists():
        for img in whole_source.glob("*.jpg"):
            shutil.copy(img, "dataset/undamaged/")
        for img in whole_source.glob("*.png"):
            shutil.copy(img, "dataset/undamaged/")
    
    # Clean up
    shutil.rmtree("temp_dataset")
    os.remove("car-damage-detection.zip")
    
    # Count images
    damaged_count = len(list(Path("dataset/damaged").glob("*.*")))
    undamaged_count = len(list(Path("dataset/undamaged").glob("*.*")))
    
    print("\n✅ Dataset setup complete!")
    print(f"   Damaged images: {damaged_count}")
    print(f"   Undamaged images: {undamaged_count}")
    print("\nReady to train! Run: python3 train_damage_model.py")

if __name__ == "__main__":
    download_and_setup_dataset()
