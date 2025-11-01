import zipfile
import shutil
from pathlib import Path
import os

downloads = Path.home() / "Downloads"

print("🔍 Looking for dataset in Downloads...")

# Find the dataset ZIP
zip_files = (
    list(downloads.glob("*damage*.zip")) + 
    list(downloads.glob("archive*.zip")) +
    list(downloads.glob("car-damage*.zip"))
)

if not zip_files:
    print("\n❌ No dataset ZIP found in Downloads")
    print("\n📥 Please download from:")
    print("   https://www.kaggle.com/datasets/anujms/car-damage-detection")
    exit()

zip_file = zip_files[0]
print(f"✅ Found: {zip_file.name}")
print("📦 Extracting...")

# Extract
with zipfile.ZipFile(zip_file, 'r') as zip_ref:
    zip_ref.extractall("temp_data")

print("📁 Organizing images...")

# Create folders
Path("dataset/damaged").mkdir(parents=True, exist_ok=True)
Path("dataset/undamaged").mkdir(parents=True, exist_ok=True)

# Find and copy files
damaged_count = 0
undamaged_count = 0

for root, dirs, files in os.walk("temp_data"):
    root_lower = root.lower()
    
    for f in files:
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            src = Path(root) / f
            
            # Check if in damaged folder
            if 'damage' in root_lower or '00-damage' in root_lower:
                shutil.copy(src, "dataset/damaged/")
                damaged_count += 1
            # Check if in whole/undamaged folder
            elif 'whole' in root_lower or '01-whole' in root_lower or 'undamaged' in root_lower:
                shutil.copy(src, "dataset/undamaged/")
                undamaged_count += 1

# Clean up
shutil.rmtree("temp_data")

print("\n" + "="*60)
print("✅ DATASET SETUP COMPLETE!")
print("="*60)
print(f"   Damaged images: {damaged_count}")
print(f"   Undamaged images: {undamaged_count}")

if damaged_count > 0 and undamaged_count > 0:
    print("\n🚀 Ready to train! Run:")
    print("   python3 train_damage_model.py")
else:
    print("\n⚠️  Warning: Check if images were copied correctly")

print("="*60)
