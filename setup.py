# setup.py
from __future__ import annotations

from pathlib import Path
from setuptools import setup, find_packages

README = Path(__file__).with_name("README.md")
long_description = README.read_text(encoding="utf-8") if README.exists() else ""

install_requires = [
    "numpy>=1.24",
    "pandas>=2.0",
    "scikit-learn>=1.3",
    "typer>=0.9",
    "appdirs>=1.4.4",
    "huggingface-hub>=0.20",
]

extras_require = {
    "embeddings": [
        "torch>=2.1",
        "transformers>=4.40",
        "accelerate>=0.29",
        "safetensors>=0.4",
        "sentencepiece>=0.1.99",
        "esm>=3.0; platform_system!='Windows'",
    ],
    "reductions": [
        "umap-learn>=0.5.5",
        "clustpy>=0.0.2",
        "tslearn>=0.6.0",
    ],
    "cli": [
        "rich>=13.0",
        "joblib>=1.3",
    ],
    "bio": [
        "biopython>=1.80",
        "biotite>=0.38.0",
        "igraph>=0.11.0",
    ],
    "dev": [
        "pytest>=7.4",
        "pytest-cov>=4.1",
        "mypy>=1.10",
        "ruff>=0.4",
    ],
}

setup(
    name="protein-representation",                 
    version="0.1.0",
    author="Kren AI Lab",
    author_email="krenai@umag.cl",
    description="Protein sequence representation: encoders, embeddings, and reductions.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ProteinEngineering-PESB2/protein_representations",  
    project_urls={
        "Source": "https://github.com/ProteinEngineering-PESB2/protein_representations",
        "Issues": "https://github.com/ProteinEngineering-PESB2/protein_representations/issues",
    },
    license="GPL-3.0-only",
    packages=find_packages(exclude=("tests", "tests.*", "docs", "examples")),
    include_package_data=True,
    python_requires=">=3.10",
    install_requires=install_requires,
    extras_require=extras_require,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: GNU General Public License v2 (GPLv2)",
        "Operating System :: OS Independent",
    ],
    keywords=[
        "protein", "bioinformatics", "embeddings", "transformers",
        "ESM", "ProtT5", "ProtBERT", "k-mers", "AAindex", "dimension-reduction",
    ],
    entry_points={
        "console_scripts": [
            "protein-representation=protein_representation.cli.main:app",
        ],
    },

    zip_safe=False,
)
