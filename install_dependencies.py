import subprocess
import sys
import platform

def install_package(package):
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--user", package])
        print(f"✓ Berhasil install {package}")
        return True
    except subprocess.CalledProcessError:
        print(f"✗ Gagal install {package}")
        return False

def install_pytorch():
    """Install PyTorch berdasarkan sistem operasi"""
    system = platform.system().lower()
    
    # Default PyTorch install command
    torch_package = "torch torchvision torchaudio"
    
    if system == "windows":
        # Untuk Windows, gunakan PyTorch dengan CUDA 11.8 atau CPU
        try:
            # Coba install dengan CUDA support terlebih dahulu
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", 
                "torch", "torchvision", "torchaudio", 
                "--index-url", "https://download.pytorch.org/whl/cu118"
            ])
            print("✓ Berhasil install PyTorch dengan CUDA support")
            return True
        except subprocess.CalledProcessError:
            try:
                # Fallback ke CPU-only version
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install",
                    "torch", "torchvision", "torchaudio", 
                    "--index-url", "https://download.pytorch.org/whl/cpu"
                ])
                print("✓ Berhasil install PyTorch (CPU-only)")
                return True
            except subprocess.CalledProcessError:
                print("✗ Gagal install PyTorch")
                return False
    else:
        # Untuk Linux/Mac, gunakan pip biasa
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--user", "torch", "torchvision", "torchaudio"])
            print("✓ Berhasil install PyTorch")
            return True
        except subprocess.CalledProcessError:
            print("✗ Gagal install PyTorch")
            return False

# Dependencies yang diperlukan
basic_packages = [
    "numpy",
    "opencv-python", 
    "scikit-learn",
    "matplotlib",
    "seaborn",
    "pandas",
    "Pillow",
    "joblib",
    "scipy",
    "tqdm"
]

print("=" * 60)
print("🔄 PULMONB-SCAN - INSTALLASI DEPENDENCIES")
print("=" * 60)

print("\n📦 Menginstall dependencies dasar...")
success_count = 0

for package in basic_packages:
    if install_package(package):
        success_count += 1

print(f"\n✅ Dependencies dasar: {success_count}/{len(basic_packages)} berhasil diinstall")

print("\n🧠 Menginstall PyTorch untuk CNN Transfer Learning...")
pytorch_success = install_pytorch()

print("\n" + "=" * 60)
print("📊 HASIL INSTALLASI")
print("=" * 60)

if success_count == len(basic_packages) and pytorch_success:
    print("🎉 SEMUA DEPENDENCIES BERHASIL DIINSTALL!")
    print("\nModel yang tersedia:")
    print("• ✅ Naive Bayes dengan Feature Extraction")
    print("• ✅ CNN dengan Transfer Learning (PyTorch)")
else:
    print("⚠ Beberapa dependencies gagal diinstall:")
    if success_count < len(basic_packages):
        print(f"  - {len(basic_packages) - success_count} packages dasar gagal")
    if not pytorch_success:
        print("  - PyTorch gagal diinstall")
    
    print("\n💡 Solusi:")
    print("1. Coba jalankan dengan administrator/root")
    print("2. Gunakan: pip install --user package_name")
    print("3. Untuk PyTorch, kunjungi: https://pytorch.org/")

print("\n" + "=" * 60)

# Verifikasi installasi
print("\n🔍 Verifikasi installasi...")
try:
    import numpy as np
    print("✓ NumPy:", np.__version__)
except ImportError:
    print("✗ NumPy tidak terinstall")

try:
    import cv2
    print("✓ OpenCV:", cv2.__version__)
except ImportError:
    print("✗ OpenCV tidak terinstall")

try:
    import sklearn
    print("✓ Scikit-learn:", sklearn.__version__)
except ImportError:
    print("✗ Scikit-learn tidak terinstall")

try:
    import torch
    print("✓ PyTorch:", torch.__version__)
    print("✓ CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("✓ GPU device:", torch.cuda.get_device_name(0))
except ImportError:
    print("✗ PyTorch tidak terinstall")

try:
    import torchvision
    print("✓ TorchVision:", torchvision.__version__)
except ImportError:
    print("✗ TorchVision tidak terinstall")

print("\n" + "=" * 60)
print("🚀 Siap menjalankan PulmoNB-Scan!")
print("=" * 60)