import subprocess
import zipfile
import shutil
from pathlib import Path
import os
import sys

def get_python_info():
    """Get current Python environment information"""
    python_exe = Path(sys.executable)
    python_dir = python_exe.parent
    
    print(f"🐍 Python executable: {python_exe}")
    print(f"📁 Python directory: {python_dir}")
    
    return python_exe, python_dir

def get_installed_packages():
    """Get list of installed packages"""
    try:
        result = subprocess.run([sys.executable, '-m', 'pip', 'freeze'], 
                               capture_output=True, text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error getting packages: {e}")
        return ""

def package_current_environment():
    """Package your current Python environment for distribution"""
    
    print("📦 Packaging current Python environment...")
    
    python_exe, python_dir = get_python_info()
    
    # Create portable environment directory
    portable_dir = Path("emaan_portable_env")
    if portable_dir.exists():
        shutil.rmtree(portable_dir)
    portable_dir.mkdir()
    
    print("📋 Getting installed packages...")
    packages = get_installed_packages()
    
    # Save requirements.txt
    with open(portable_dir / "requirements.txt", "w") as f:
        f.write(packages)
    
    # Essential directories to copy
    essential_dirs = ['Scripts', 'Lib', 'DLLs', 'Include']
    
    print("📂 Copying Python directories...")
    for dir_name in essential_dirs:
        src_dir = python_dir / dir_name
        if src_dir.exists():
            dst_dir = portable_dir / dir_name
            print(f"  Copying {dir_name}...")
            try:
                shutil.copytree(src_dir, dst_dir, ignore=shutil.ignore_patterns('*.pyc', '__pycache__'))
            except Exception as e:
                print(f"  Warning: Could not copy {dir_name}: {e}")
    
    # Essential files to copy
    essential_files = [
        'python.exe', 
        'pythonw.exe', 
        'python310.dll',
        'python3.dll',
        'vcruntime140.dll',
        'vcruntime140_1.dll'
    ]
    
    print("📄 Copying Python files...")
    for file_name in essential_files:
        src_file = python_dir / file_name
        if src_file.exists():
            print(f"  Copying {file_name}...")
            shutil.copy2(src_file, portable_dir / file_name)
    
    # Also check parent directory for DLLs (common in conda envs)
    parent_dir = python_dir.parent
    for file_name in essential_files:
        src_file = parent_dir / file_name
        if src_file.exists() and not (portable_dir / file_name).exists():
            print(f"  Copying {file_name} from parent...")
            shutil.copy2(src_file, portable_dir / file_name)
    
    # Create zip file
    print("🗜️ Creating python_env.zip...")
    with zipfile.ZipFile('python_env.zip', 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(portable_dir):
            for file in files:
                file_path = os.path.join(root, file)
                # Create archive path starting with 'python_env'
                arcname = os.path.join('python_env', os.path.relpath(file_path, portable_dir))
                zipf.write(file_path, arcname)
                
                # Show progress for large operations
                if len(files) > 1000 and files.index(file) % 500 == 0:
                    print(f"    Added {files.index(file)} files...")
    
    # Cleanup
    shutil.rmtree(portable_dir)
    
    file_size = os.path.getsize('python_env.zip') / 1024 / 1024
    print(f"✅ Environment packaged: python_env.zip")
    print(f"📊 Size: {file_size:.1f} MB")
    
    if file_size > 500:
        print("⚠️  Warning: Package is quite large. Consider cleaning up unnecessary packages.")
    
    return True

def create_requirements_only():
    """Lighter alternative: just create requirements.txt"""
    print("📝 Creating requirements.txt only...")
    
    packages = get_installed_packages()
    with open("requirements.txt", "w") as f:
        f.write(packages)
    
    print("✅ Requirements saved to requirements.txt")
    print("📢 You can use this for pip-based deployment instead")

if __name__ == "__main__":
    print("Choose packaging method:")
    print("1. Full environment (recommended for complex setups)")
    print("2. Requirements.txt only (lighter, requires pip install)")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "2":
        create_requirements_only()
    else:
        try:
            package_current_environment()
        except Exception as e:
            print(f"❌ Error packaging environment: {e}")
            print("\n💡 Trying requirements.txt method instead...")
            create_requirements_only()
