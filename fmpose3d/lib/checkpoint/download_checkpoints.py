"""
Auto-download checkpoint files from Google Drive if they don't exist.
Downloads to user cache directory for better compatibility.
"""
import os
import sys
import platform

# Required checkpoint files
REQUIRED_FILES = ['yolov3.weights', 'pose_hrnet_w48_384x288.pth']

# Google Drive folder URL
GDRIVE_FOLDER_URL = "https://drive.google.com/drive/folders/1_ENAMOsPM7FXmdYRbkwbFHgzQq_B_NQA"

def get_cache_dir():
    """Get the platform-specific cache directory for fmpose checkpoints."""
    system = platform.system()
    
    if system == "Windows":
        # Windows: %LOCALAPPDATA%\fmpose\checkpoints
        base = os.environ.get("LOCALAPPDATA", os.path.expanduser("~"))
        cache_dir = os.path.join(base, "fmpose", "checkpoints")
    elif system == "Darwin":
        # macOS: ~/Library/Caches/fmpose/checkpoints
        cache_dir = os.path.join(os.path.expanduser("~"), "Library", "Caches", "fmpose", "checkpoints")
    else:
        # Linux and others: ~/.cache/fmpose/checkpoints
        xdg_cache = os.environ.get("XDG_CACHE_HOME", os.path.join(os.path.expanduser("~"), ".cache"))
        cache_dir = os.path.join(xdg_cache, "fmpose", "checkpoints")
    
    return cache_dir

def get_checkpoint_dir():
    """Get the checkpoint directory path (creates if not exists)."""
    cache_dir = get_cache_dir()
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir

def get_checkpoint_path(filename):
    """Get the full path to a checkpoint file."""
    return os.path.join(get_checkpoint_dir(), filename)

def install_gdown():
    """Install gdown if not available."""
    try:
        import gdown
        return gdown
    except ImportError:
        print("Installing gdown for Google Drive downloads...")
        os.system(f"{sys.executable} -m pip install gdown -q")
        import gdown
        return gdown

def download_folder(output_dir):
    """Download all files from the Google Drive folder."""
    gdown = install_gdown()
    
    print(f"Downloading checkpoint files from Google Drive folder...")
    print(f"URL: {GDRIVE_FOLDER_URL}")
    print(f"Saving to: {output_dir}\n")
    
    try:
        gdown.download_folder(GDRIVE_FOLDER_URL, output=output_dir, quiet=False)
        return True
    except Exception as e:
        print(f"Folder download failed: {e}")
        return False

def ensure_checkpoints():
    """Check and download missing checkpoint files."""
    checkpoint_dir = get_checkpoint_dir()
    
    # Check which files are missing
    missing_files = []
    for filename in REQUIRED_FILES:
        filepath = os.path.join(checkpoint_dir, filename)
        if not os.path.exists(filepath):
            missing_files.append(filename)
    
    if not missing_files:
        return True
    
    print(f"\n{'='*60}")
    print("Missing checkpoint files detected:")
    for f in missing_files:
        print(f"  - {f}")
    print(f"{'='*60}")
    print("Starting auto-download from Google Drive...")
    print(f"Cache directory: {checkpoint_dir}")
    print(f"{'='*60}\n")
    
    # Try to download the folder
    success = download_folder(checkpoint_dir)
    
    # Verify all files are present
    still_missing = []
    for filename in REQUIRED_FILES:
        filepath = os.path.join(checkpoint_dir, filename)
        if not os.path.exists(filepath):
            still_missing.append(filename)
    
    if not still_missing:
        print(f"\n{'='*60}")
        print("All checkpoint files downloaded successfully!")
        print(f"{'='*60}\n")
        return True
    else:
        print(f"\n{'='*60}")
        print("WARNING: Some files are still missing:")
        for f in still_missing:
            print(f"  - {f}")
        print(f"\nPlease download manually from:")
        print(GDRIVE_FOLDER_URL)
        print(f"And place them in: {checkpoint_dir}")
        print(f"{'='*60}\n")
        return False

if __name__ == "__main__":
    ensure_checkpoints()
    print(f"\nCheckpoint directory: {get_checkpoint_dir()}")
