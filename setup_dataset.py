cat > setup_damage_dataset.py << 'EOF'
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
    print("\
