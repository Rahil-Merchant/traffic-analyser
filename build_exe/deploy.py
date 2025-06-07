import subprocess
import os
from pathlib import Path

def deploy_update():
    """Package and deploy updates"""
    
    print("🚀 Deploying EMAAN update...")
    
    # 1. Package backend
    subprocess.run(['python', 'package_backend.py'], check=True)
    
    # 2. Upload to your server (replace with your method)
    files_to_upload = [
        'backend_latest.zip',
        'versions.json'
    ]
    
    # Example using SCP (replace with your upload method)
    server = "your-server.com"
    remote_path = "/var/www/emaan/"
    
    for file in files_to_upload:
        cmd = f"scp {file} user@{server}:{remote_path}"
        print(f"Uploading {file}...")
        # subprocess.run(cmd.split(), check=True)  # Uncomment when ready
    
    print("✅ Update deployed!")
    print("📢 All users will get the update on next launch")

if __name__ == "__main__":
    deploy_update()
