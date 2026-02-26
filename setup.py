from setuptools import setup, find_packages

setup(
    name="airfoil-generator",
    version="1.0.0",
    description="AI-Powered Airfoil Shape Generator — Inverse Design using CVAE",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    python_requires=">=3.9,<3.12",
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scipy>=1.7.0",
        "torch>=1.12.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.5.0",
        "h5py>=3.7.0",
        "tqdm>=4.64.0",
        "pyyaml>=6.0",
    ],
)