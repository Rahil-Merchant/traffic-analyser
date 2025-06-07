import subprocess
import os
from pathlib import Path

def build_launcher():
    """Build launcher into executable"""
    
    print("🔨 Building launcher executable...")
    
    # PyInstaller command
    cmd = [
        'pyinstaller',
        '--onefile',
        '--windowed',
        '--name=EMAAN_Setup',
        '--icon=icon.ico',  # Add your icon file
        'launcher.py'
    ]
    
    subprocess.run(cmd, check=True)
    
    print("✅ Launcher built: dist/EMAAN_Setup.exe")
    print("📊 This is what you distribute to users!")

if __name__ == "__main__":
    build_launcher()
