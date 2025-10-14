#!/usr/bin/env python3
"""
Diagnostic test for HER2 Attention Classifier project dependencies.
Checks CUDA, NVIDIA drivers, and system dependencies.
"""

import os
import sys
import subprocess
import importlib
from pathlib import Path

def run_command(cmd, capture_output=True):
    """Run shell command and return result."""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=capture_output, text=True)
        return result.returncode == 0, result.stdout.strip(), result.stderr.strip()
    except Exception as e:
        return False, "", str(e)

def check_nvidia_driver():
    """Check NVIDIA driver availability."""
    print("Looking for NVIDIA Driver...")
    success, stdout, stderr = run_command("nvidia-smi")
    if success:
        print("NVIDIA Driver: Available")
        # Extract driver version
        lines = stdout.split('\n')
        for line in lines:
            if "Driver Version:" in line:
                version = line.split("Driver Version:")[1].split()[0]
                print(f"   Version: {version}")
                break
        return True
    else:
        print("NVIDIA Driver: Not found or not working")
        print(f"   Error: {stderr}")
        return False

def check_cuda():
    """Check CUDA availability."""
    print("\n Looking for CUDA...")
    
    # Check nvcc compiler
    success, stdout, stderr = run_command("nvcc --version")
    if success:
        print("CUDA Compiler (nvcc): Available")
        # Extract CUDA version
        for line in stdout.split('\n'):
            if "release" in line.lower():
                version = line.split("release")[1].split(",")[0].strip()
                print(f"   Version: {version}")
                break
        cuda_available = True
    else:
        print("CUDA Compiler (nvcc): Not found")
        cuda_available = False
    
    # Check CUDA runtime
    try:
        import torch
        if torch.cuda.is_available():
            print("CUDA Runtime: Available via PyTorch")
            print(f"CUDA Version: {torch.version.cuda}")
            print(f"Available GPUs: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                print(f"   GPU {i}: {gpu_name}")
        else:
            print("CUDA Runtime: Not available via PyTorch")
            cuda_available = False
    except ImportError:
        print("PyTorch not installed, cannot check CUDA runtime")
        
    return cuda_available

def check_system_libraries():
    """Check required system libraries."""
    print("\nChecking System Libraries...")
    
    required_libs = {
        'libgcc_s.so.1': 'GCC runtime library',
        'libgomp.so.1': 'OpenMP runtime library',
        'libbz2.so.1': 'Bzip2 compression library',
        'libssl.so.3': 'OpenSSL library',
        'libffi.so.8': 'Foreign Function Interface library'
    }
    
    all_found = True
    for lib, description in required_libs.items():
        success, _, _ = run_command(f"ldconfig -p | grep {lib}")
        if success:
            print(f"{description}: Found")
        else:
            print(f"{description}: Not found ({lib})")
            all_found = False
    
    return all_found

def check_python_packages():
    """Check required Python packages."""
    print("\n🔍 Checking Python Packages...")
    
    required_packages = [
        'torch', 'torchvision', 'numpy', 'matplotlib', 
        'PIL', 'cv2', 'sklearn', 'pandas', 'jupyter'
    ]
    
    all_found = True
    for package in required_packages:
        try:
            if package == 'PIL':
                importlib.import_module('PIL')
            elif package == 'cv2':
                importlib.import_module('cv2')
            elif package == 'sklearn':
                importlib.import_module('sklearn')
            else:
                importlib.import_module(package)
            print(f"{package}: Available")
        except ImportError:
            print(f"{package}: Not installed")
            all_found = False
    
    return all_found

def check_environment():
    """Check conda/python environment."""
    print("\Check Environment...")
    
    # Python version
    print(f"Python Version: {sys.version.split()[0]}")
    
    # Conda environment
    conda_env = os.environ.get('CONDA_DEFAULT_ENV', 'Not in conda environment')
    print(f"Conda Environment: {conda_env}")
    
    # Check if in correct environment
    if 'her2-class' in conda_env:
        print("Correct conda environment activated")
        return True
    else:
        print("Expected 'her2-class' conda environment")
        return False

def run_diagnostic():
    """Run complete diagnostic test."""
    print("Running Diagnostic")
    
    results = {
        'nvidia_driver': check_nvidia_driver(),
        'cuda': check_cuda(),
        'system_libs': check_system_libraries(),
        'python_packages': check_python_packages(),
        'environment': check_environment()
    }
    
    print("\nSummary:")
    print("=" * 50)
    
    all_pass = True
    for component, status in results.items():
        status_icon = "True" if status else "False"
        component_name = component.replace('_', ' ').title()
        print(f"{status_icon} {component_name}: {'PASS' if status else 'FAIL'}")
        if not status:
            all_pass = False
    
    print("\nOverall State:", " - + READY" if all_pass else " - - NEEDS ATTENTION")
    
    if not all_pass:
        print("\n Recommendations:")
        if not results['nvidia_driver']:
            print("   - Install NVIDIA drivers: sudo apt update && sudo apt install nvidia-driver-xxx")
        if not results['cuda']:
            print("   - Install CUDA toolkit or ensure PyTorch with CUDA support")
        if not results['system_libs']:
            print("   - Install missing system libraries via conda or apt")
        if not results['python_packages']:
            print("   - Install missing packages: pip install torch torchvision numpy matplotlib...")
        if not results['environment']:
            print("   - Activate correct environment: conda activate her2-class")
    
    return all_pass

if __name__ == "__main__":
    # Make script executable standalone
    success = run_diagnostic()
    sys.exit(0 if success else 1)
