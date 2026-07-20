from setuptools import setup, find_packages

setup(
    name="equivcompiler",
    version="0.2.0",
    description="Representation-compiled equivariant probabilistic structured prediction",
    packages=find_packages(exclude=["tests", "tests.*"]),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "pandas>=1.5.0",
        "scikit-learn>=1.2.0",
        "torch-geometric>=2.3.0",
        "torchmetrics>=0.11.0",
        "e3nn>=0.5.7",
        "h5py>=3.8.0",
        "ase>=3.22.0",
        "pymatgen>=2023.0.0",
        "tqdm>=4.65.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
    ],
    extras_require={
        "test": ["pytest>=7.0.0"],
    },
    entry_points={
        "console_scripts": [
            "equiv-compiler=equivcompiler.cli:main",
        ],
    },
)
