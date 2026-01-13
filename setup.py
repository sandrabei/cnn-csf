"""Setup configuration for CNN-CSF package."""
from pathlib import Path
from setuptools import setup, find_packages

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
long_description = ""
if readme_file.exists():
    long_description = readme_file.read_text(encoding="utf-8")

setup(
    name="cnn-csf",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A library for 3D medical image point detection using U-Net",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sandrabei/cnn-csf",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Medical Image Processing",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.19.0",
        "torch>=1.9.0",
        "torchvision>=0.10.0",
        "pandas>=1.2.0",
        "tqdm>=4.60.0",
        "scikit-learn>=0.24.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=21.0",
            "flake8>=3.9",
        ],
    },
    entry_points={
        "console_scripts": [
            "cnn-csf-train=cnn_csf.scripts.train:main",
            "cnn-csf-eval=cnn_csf.scripts.eval:main",
        ],
    },
)
