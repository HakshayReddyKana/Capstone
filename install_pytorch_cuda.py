# PyTorch CUDA Installation Script
# Run this to install PyTorch with CUDA support

import subprocess
import sys
import os

def run_command(cmd):
    """Run command and return success status"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=600)
        if result.returncode == 0:
            print(f"✓ Success: {cmd}")
            return True
        else:
            print(f"✗ Failed: {cmd}")
            print(f"Error: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print(f"✗ Timeout: {cmd}")
        return False
    except Exception as e:
        print(f"✗ Error: {cmd} - {e}")
        return False

def main():
    print("PyTorch CUDA Installation Script")
    print("=" * 40)

    # Check current PyTorch
    try:
        import torch
        print(f"Current PyTorch: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print("CUDA already working! No installation needed.")
            return
    except ImportError:
        print("PyTorch not installed")

    # Uninstall current version
    print("\n1. Uninstalling current PyTorch...")
    run_command("pip uninstall torch torchvision torchaudio -y")

    # Install CUDA version
    print("\n2. Installing PyTorch with CUDA support...")
    print("This may take several minutes...")

    # Try different CUDA versions
    cuda_versions = [
        "cu121",  # CUDA 12.1 (works with CUDA 12.6)
        "cu118",  # CUDA 11.8
        "cu117",  # CUDA 11.7
    ]

    for cuda_ver in cuda_versions:
        print(f"\nTrying CUDA {cuda_ver}...")
        cmd = f"pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/{cuda_ver} --no-cache-dir --timeout 600"
        if run_command(cmd):
            break
    else:
        print("\n❌ All installation attempts failed.")
        print("Try manual installation:")
        print("1. Visit: https://pytorch.org/get-started/locally/")
        print("2. Select: PyTorch, Windows, Pip, Python, CUDA 12.1")
        print("3. Run the provided pip command")
        return

    # Verify installation
    print("\n3. Verifying installation...")
    try:
        import torch
        print(f"✓ PyTorch installed: {torch.__version__}")
        print(f"✓ CUDA available: {torch.cuda.is_available()}")

        if torch.cuda.is_available():
            print(f"✓ CUDA version: {torch.version.cuda}")
            print(f"✓ GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"✓ GPU {i}: {torch.cuda.get_device_name(i)}")

            # Test CUDA
            test_tensor = torch.randn(1, 3, 224, 224).cuda()
            print("✓ CUDA tensor creation successful")
            del test_tensor
            torch.cuda.empty_cache()
            print("\n🎉 CUDA installation successful!")
        else:
            print("❌ CUDA not available after installation")

    except Exception as e:
        print(f"❌ Installation verification failed: {e}")

if __name__ == "__main__":
    main()