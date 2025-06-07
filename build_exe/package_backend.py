import zipfile
import os
import json
from pathlib import Path
import shutil

def package_backend():
    """Package backend code for distribution"""
    
    print("📦 Packaging backend...")
    
    # Files and directories to include
    backend_items = [
        "dev_server.py",
        "config.py",
        "core/",
        "requirements_.txt",
        # Add any other files/folders your backend needs
    ]
    
    # Create backend package
    with zipfile.ZipFile('backend_latest.zip', 'w', zipfile.ZIP_DEFLATED) as zipf:
        for item in backend_items:
            item_path = Path(item)
            if item_path.is_dir():
                for root, dirs, files in os.walk(item_path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.join('backend', os.path.relpath(file_path))
                        zipf.write(file_path, arcname)
            elif item_path.exists():
                arcname = os.path.join('backend', item)
                zipf.write(item_path, arcname)
    
    print("✅ Backend packaged: backend_latest.zip")
    print(f"📊 Size: {os.path.getsize('backend_latest.zip') / 1024 / 1024:.1f} MB")

def create_versions_file():
    """Create versions.json for update checking"""
    
    # Increment version number (you can automate this)
    versions = {
        "backend": "1.0.1",  # Update this each time
        "frontend": "1.0.0"  # Update when frontend changes
    }
    
    with open('versions.json', 'w') as f:
        json.dump(versions, f, indent=2)
    
    print("✅ Version file created: versions.json")

if __name__ == "__main__":
    package_backend()
    create_versions_file()
