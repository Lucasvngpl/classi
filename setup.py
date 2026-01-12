#!/usr/bin/env python3
"""
Setup and test script for Photo Organizer.
Verifies installation and creates sample directory structure.
"""

import sys
import subprocess
import os
from pathlib import Path


def check_python_version():
    """Check if Python version is 3.8 or higher."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python 3.8+ required. You have {version.major}.{version.minor}")
        return False
    print(f"✓ Python {version.major}.{version.minor}.{version.micro}")
    return True


def install_dependencies():
    """Install required dependencies."""
    print("\nInstalling dependencies...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ])
        print("✓ Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install dependencies")
        return False


def verify_imports():
    """Verify that all required packages can be imported."""
    print("\nVerifying installations...")
    packages = {
        'tensorflow': 'TensorFlow',
        'PIL': 'Pillow',
        'numpy': 'NumPy'
    }
    
    all_ok = True
    for module, name in packages.items():
        try:
            __import__(module)
            print(f"✓ {name}")
        except ImportError:
            print(f"❌ {name} not found")
            all_ok = False
    
    return all_ok


def create_sample_structure():
    """Create sample directory structure for testing."""
    print("\nCreating sample directory structure...")
    
    # Create test directories
    test_dir = Path("test_photos")
    test_dir.mkdir(exist_ok=True)
    
    print(f"✓ Created '{test_dir}/' directory")
    print(f"\n📁 Place your test images in '{test_dir}/' directory")
    print(f"   Then run: python organizer.py {test_dir}")
    
    return True


def main():
    """Main setup function."""
    print("="*60)
    print("Photo Organizer - Setup & Verification")
    print("="*60)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Ask user if they want to install dependencies
    response = input("\nInstall dependencies? (y/n): ").lower().strip()
    if response == 'y':
        if not install_dependencies():
            sys.exit(1)
        
        if not verify_imports():
            print("\n❌ Some packages failed to import. Please check the errors above.")
            sys.exit(1)
    else:
        print("\nSkipping dependency installation.")
        print("Run manually: pip install -r requirements.txt")
    
    # Create sample structure
    create_sample_structure()
    
    print("\n" + "="*60)
    print("✓ Setup complete!")
    print("="*60)
    print("\nNext steps:")
    print("1. Place test images in 'test_photos/' directory")
    print("2. Run: python organizer.py test_photos --dry-run")
    print("3. If results look good, run without --dry-run flag")
    print("\nFor single image test:")
    print("  python classifier.py path/to/image.jpg")
    print("="*60)


if __name__ == "__main__":
    main()
