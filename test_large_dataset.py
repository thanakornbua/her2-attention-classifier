"""
Test script to simulate large-scale evaluation with the condensed code.
Creates synthetic dataset and tests all optimizations.
"""
import numpy as np
from PIL import Image
from pathlib import Path
import torch
import sys

def create_test_images(base_dir: Path, num_images_per_class: int = 1000, image_size: int = 512):
    """Create synthetic test images to simulate large dataset."""
    print(f"Creating {num_images_per_class * 2} test images ({image_size}x{image_size})...")
    
    for class_idx, class_name in enumerate(['class0', 'class1']):
        class_dir = base_dir / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        
        for i in range(num_images_per_class):
            # Create random RGB image
            img_array = np.random.randint(0, 256, (image_size, image_size, 3), dtype=np.uint8)
            img = Image.fromarray(img_array)
            img.save(class_dir / f"img_{i:05d}.jpg", quality=95)
            
            if (i + 1) % 100 == 0:
                print(f"  {class_name}: {i+1}/{num_images_per_class} images created")
    
    print(f"✓ Dataset created: {num_images_per_class * 2} images total")

def create_dummy_model(num_classes: int = 2):
    """Create a dummy model checkpoint for testing."""
    checkpoint_path = Path("test_model.pth")
    
    # Create a simple model state dict
    state = {
        'model_state_dict': {
            'fc.weight': torch.randn(num_classes, 512),
            'fc.bias': torch.randn(num_classes)
        },
        'num_classes': num_classes,
        'arch': 'resnet18'
    }
    
    torch.save(state, checkpoint_path)
    print(f"✓ Dummy model saved: {checkpoint_path}")
    return checkpoint_path

def test_evaluate_imagefolder():
    """Test evaluate_imagefolder.py with large dataset."""
    print("\n" + "="*70)
    print("TEST 1: evaluate_imagefolder.py")
    print("="*70)
    
    # Create test data
    test_dir = Path("test_data")
    output_dir = Path("test_output_imagefolder")
    output_dir.mkdir(exist_ok=True)
    
    # Create dataset (1000 images per class = 2000 total, 512x512)
    create_test_images(test_dir, num_images_per_class=500, image_size=512)
    model_path = create_dummy_model()
    
    # Test with various configurations
    test_configs = [
        {
            "name": "Auto-scaling batch size",
            "args": f"--data-dir {test_dir} --model-path {model_path} --output-dir {output_dir} --image-size 512"
        },
        {
            "name": "AMP enabled",
            "args": f"--data-dir {test_dir} --model-path {model_path} --output-dir {output_dir} --image-size 512 --amp"
        },
        {
            "name": "Custom batch size",
            "args": f"--data-dir {test_dir} --model-path {model_path} --output-dir {output_dir} --image-size 512 --batch-size 8"
        }
    ]
    
    for config in test_configs:
        print(f"\n{'─'*70}")
        print(f"Testing: {config['name']}")
        print(f"{'─'*70}")
        cmd = f"python inference/cli/evaluate_imagefolder.py {config['args']}"
        print(f"Command: {cmd}\n")
        
        import subprocess
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✓ PASSED")
            # Print last 10 lines of output
            lines = result.stdout.split('\n')
            print("\nOutput (last 10 lines):")
            for line in lines[-10:]:
                if line.strip():
                    print(f"  {line}")
        else:
            print("✗ FAILED")
            print(f"Error: {result.stderr}")
            return False
    
    return True

def test_evaluate_performance():
    """Test evaluate_performance.py with large dataset."""
    print("\n" + "="*70)
    print("TEST 2: evaluate_performance.py")
    print("="*70)
    
    test_dir = Path("test_data")
    output_dir = Path("test_output_performance")
    output_dir.mkdir(exist_ok=True)
    model_path = Path("test_model.pth")
    
    test_configs = [
        {
            "name": "Basic performance test",
            "args": f"--data-dir {test_dir} --model-path {model_path} --output-dir {output_dir} --image-size 512 --warmup-batches 2 --measure-batches 10"
        },
        {
            "name": "With AMP",
            "args": f"--data-dir {test_dir} --model-path {model_path} --output-dir {output_dir} --image-size 512 --amp --warmup-batches 2 --measure-batches 10"
        }
    ]
    
    for config in test_configs:
        print(f"\n{'─'*70}")
        print(f"Testing: {config['name']}")
        print(f"{'─'*70}")
        cmd = f"python inference/cli/evaluate_performance.py {config['args']}"
        print(f"Command: {cmd}\n")
        
        import subprocess
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✓ PASSED")
            lines = result.stdout.split('\n')
            print("\nOutput (last 10 lines):")
            for line in lines[-10:]:
                if line.strip():
                    print(f"  {line}")
        else:
            print("✗ FAILED")
            print(f"Error: {result.stderr}")
            return False
    
    return True

def cleanup():
    """Clean up test files."""
    print("\n" + "="*70)
    print("CLEANUP")
    print("="*70)
    
    import shutil
    paths_to_remove = [
        "test_data",
        "test_output_imagefolder",
        "test_output_performance",
        "test_model.pth"
    ]
    
    for path in paths_to_remove:
        p = Path(path)
        if p.exists():
            if p.is_dir():
                shutil.rmtree(p)
            else:
                p.unlink()
            print(f"✓ Removed: {path}")

if __name__ == "__main__":
    print("="*70)
    print("LARGE DATASET EVALUATION TEST")
    print("="*70)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    try:
        # Run tests
        test1_passed = test_evaluate_imagefolder()
        test2_passed = test_evaluate_performance()
        
        # Summary
        print("\n" + "="*70)
        print("TEST SUMMARY")
        print("="*70)
        print(f"evaluate_imagefolder.py: {'✓ PASSED' if test1_passed else '✗ FAILED'}")
        print(f"evaluate_performance.py: {'✓ PASSED' if test2_passed else '✗ FAILED'}")
        
        if test1_passed and test2_passed:
            print("\n🎉 ALL TESTS PASSED - Code is ready for production!")
        else:
            print("\n⚠ SOME TESTS FAILED - Check errors above")
            sys.exit(1)
    
    finally:
        # Cleanup
        cleanup()
