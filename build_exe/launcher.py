import os
import sys
import subprocess
import requests
import zipfile
import json
from pathlib import Path
import time
import threading
from tkinter import Tk, Label, ProgressBar, StringVar
import tkinter as tk
from tkinter import ttk

class EmaanLauncher:
    def __init__(self):
        self.app_dir = Path.home() / "EMAAN_Urban_Reconnaissance"
        self.backend_dir = self.app_dir / "backend"
        self.env_dir = self.app_dir / "python_env"
        self.frontend_dir = self.app_dir / "frontend"
        self.config_file = self.app_dir / "launcher_config.json"
        
        # Your server URLs (replace with your actual hosting)
        self.base_url = "https://your-server.com/emaan"  # Replace this
        self.env_url = f"{self.base_url}/python_env.zip"
        self.backend_url = f"{self.base_url}/backend_latest.zip"
        self.frontend_url = f"{self.base_url}/frontend_latest.zip"
        self.version_url = f"{self.base_url}/versions.json"
        
        # GUI
        self.root = None
        self.progress_var = None
        self.status_var = None
        self.progress_bar = None
        
    def create_gui(self):
        """Create a simple progress GUI"""
        self.root = Tk()
        self.root.title("EMAAN Urban Reconnaissance")
        self.root.geometry("500x200")
        self.root.resizable(False, False)
        
        # Center window
        self.root.eval('tk::PlaceWindow . center')
        
        # Styling
        style = ttk.Style()
        style.theme_use('clam')
        
        # Logo/Title
        title_label = Label(self.root, text="EMAAN Urban Reconnaissance", 
                           font=("Arial", 16, "bold"), fg="#2563eb")
        title_label.pack(pady=20)
        
        # Status
        self.status_var = StringVar(value="Initializing...")
        status_label = Label(self.root, textvariable=self.status_var, 
                           font=("Arial", 10))
        status_label.pack(pady=10)
        
        # Progress bar
        self.progress_var = StringVar()
        self.progress_bar = ttk.Progressbar(self.root, mode='determinate', 
                                          length=400)
        self.progress_bar.pack(pady=10)
        
        # Progress percentage
        progress_label = Label(self.root, textvariable=self.progress_var, 
                             font=("Arial", 9), fg="#666")
        progress_label.pack()
        
    def update_progress(self, value, status):
        """Update GUI progress"""
        if self.root:
            self.progress_bar['value'] = value
            self.status_var.set(status)
            self.progress_var.set(f"{int(value)}%")
            self.root.update_idletasks()
            
    def load_config(self):
        """Load launcher configuration"""
        if self.config_file.exists():
            with open(self.config_file, 'r') as f:
                return json.load(f)
        return {"backend_version": "0.0.0", "frontend_version": "0.0.0"}
        
    def save_config(self, config):
        """Save launcher configuration"""
        self.app_dir.mkdir(exist_ok=True)
        with open(self.config_file, 'w') as f:
            json.dump(config, f)
            
    def download_file(self, url, filepath, description):
        """Download a file with progress"""
        try:
            response = requests.get(url, stream=True)
            total_size = int(response.headers.get('content-length', 0))
            
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            with open(filepath, 'wb') as f:
                downloaded = 0
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            progress = (downloaded / total_size) * 100
                            self.update_progress(progress, f"Downloading {description}...")
                            
            return True
        except Exception as e:
            print(f"Error downloading {description}: {e}")
            return False
            
    def extract_zip(self, zip_path, extract_to, description):
        """Extract zip file with progress"""
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                files = zip_ref.namelist()
                total_files = len(files)
                
                for i, file in enumerate(files):
                    zip_ref.extract(file, extract_to)
                    progress = ((i + 1) / total_files) * 100
                    self.update_progress(progress, f"Extracting {description}...")
                    
            zip_path.unlink()  # Delete zip file
            return True
        except Exception as e:
            print(f"Error extracting {description}: {e}")
            return False
            
    def check_for_updates(self):
        """Check if updates are available"""
        try:
            response = requests.get(self.version_url, timeout=10)
            remote_versions = response.json()
            local_config = self.load_config()
            
            updates_needed = {
                'backend': remote_versions['backend'] != local_config.get('backend_version', '0.0.0'),
                'frontend': remote_versions['frontend'] != local_config.get('frontend_version', '0.0.0'),
                'environment': not self.env_dir.exists()
            }
            
            return updates_needed, remote_versions
        except:
            # If can't check updates, assume no updates needed
            return {'backend': False, 'frontend': False, 'environment': False}, {}
            
    def setup_environment(self):
        """Set up Python environment"""
        self.update_progress(10, "Setting up Python environment...")
        
        if self.env_dir.exists():
            import shutil
            shutil.rmtree(self.env_dir)
            
        # Download portable Python environment
        env_zip = self.app_dir / "python_env.zip"
        if self.download_file(self.env_url, env_zip, "Python Environment"):
            self.extract_zip(env_zip, self.app_dir, "Python Environment")
            return True
        return False
        
    def update_backend(self):
        """Download and update backend"""
        self.update_progress(40, "Updating backend...")
        
        if self.backend_dir.exists():
            import shutil
            shutil.rmtree(self.backend_dir)
            
        backend_zip = self.app_dir / "backend.zip"
        if self.download_file(self.backend_url, backend_zip, "Backend"):
            return self.extract_zip(backend_zip, self.backend_dir, "Backend")
        return False
        
    def update_frontend(self):
        """Download and update frontend"""
        self.update_progress(70, "Updating interface...")
        
        if self.frontend_dir.exists():
            import shutil
            shutil.rmtree(self.frontend_dir)
            
        frontend_zip = self.app_dir / "frontend.zip"
        if self.download_file(self.frontend_url, frontend_zip, "Interface"):
            return self.extract_zip(frontend_zip, self.frontend_dir, "Interface")
        return False
        
    def launch_backend(self):
        """Start the backend server"""
        python_exe = self.env_dir / "Scripts" / "python.exe"
        backend_script = self.backend_dir / "dev_server.py"
        
        if not python_exe.exists() or not backend_script.exists():
            raise Exception("Backend components not found")
            
        # Start backend in background
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startupinfo.wShowWindow = subprocess.SW_HIDE
        
        backend_process = subprocess.Popen(
            [str(python_exe), str(backend_script)],
            startupinfo=startupinfo,
            cwd=str(self.backend_dir)
        )
        
        # Wait a moment for backend to start
        time.sleep(3)
        return backend_process
        
    def launch_frontend(self):
        """Start the frontend application"""
        frontend_exe = self.frontend_dir / "EMAAN Urban Reconnaissance.exe"
        
        if not frontend_exe.exists():
            raise Exception("Frontend application not found")
            
        frontend_process = subprocess.Popen([str(frontend_exe)])
        return frontend_process
        
    def run(self):
        """Main launcher logic"""
        try:
            self.create_gui()
            self.update_progress(0, "Checking for updates...")
            
            # Check what needs updating
            updates_needed, remote_versions = self.check_for_updates()
            
            # Update components as needed
            if updates_needed['environment']:
                if not self.setup_environment():
                    raise Exception("Failed to setup environment")
                    
            if updates_needed['backend']:
                if not self.update_backend():
                    raise Exception("Failed to update backend")
                    
            if updates_needed['frontend']:
                if not self.update_frontend():
                    raise Exception("Failed to update frontend")
                    
            # Save new versions
            if remote_versions:
                self.save_config({
                    'backend_version': remote_versions.get('backend', '0.0.0'),
                    'frontend_version': remote_versions.get('frontend', '0.0.0')
                })
                
            self.update_progress(90, "Starting application...")
            
            # Launch backend
            backend_process = self.launch_backend()
            
            self.update_progress(95, "Opening interface...")
            
            # Close progress window
            self.root.withdraw()
            
            # Launch frontend
            frontend_process = self.launch_frontend()
            
            # Wait for frontend to close
            frontend_process.wait()
            
            # Cleanup backend
            backend_process.terminate()
            
        except Exception as e:
            if self.root:
                self.status_var.set(f"Error: {e}")
                self.root.after(5000, self.root.destroy)
                self.root.mainloop()
            else:
                print(f"Error: {e}")
                input("Press Enter to exit...")

if __name__ == "__main__":
    launcher = EmaanLauncher()
    launcher.run()
