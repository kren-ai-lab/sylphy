# setup.py
from __future__ import annotations

from pathlib import Path
from setuptools import setup, find_packages
from setuptools.command.install import install
from setuptools.command.develop import develop
import os

README = Path(__file__).with_name("README.md")
long_description = README.read_text(encoding="utf-8") if README.exists() else ""
PACKAGE_NAME = "sylphy"

def _resolve_cache_dir(cli_value: str | None) -> str:
    """Decide cache dir during installation time."""
    if cli_value:
        return str(Path(cli_value).expanduser())
    env = os.getenv("SYLPHY_CACHE_DIR")
    if env:
        return str(Path(env).expanduser())
    # fallback: platformdirs
    try:
        from platformdirs import user_cache_dir
        base = Path(user_cache_dir(PACKAGE_NAME))
    except Exception:
        base = Path.home() / ".cache" / PACKAGE_NAME
    return str((base).expanduser())

def _write_siteconfig(target_root: Path, cache_dir: str) -> None:

    pkg_dir = target_root / PACKAGE_NAME
    pkg_dir.mkdir(parents=True, exist_ok=True)
    siteconfig = pkg_dir / "_siteconfig.py"
    siteconfig.write_text(
        f'# Auto-generated at install time\nCACHE_DIR = r"{cache_dir}"\n',
        encoding="utf-8"
    )

class SylphyInstall(install):
    user_options = install.user_options + [
        ('sylphy-cache-dir=', None, 'Custom cache directory for Sylphy'),
    ]

    def initialize_options(self):
        super().initialize_options()
        self.sylphy_cache_dir = None

    def finalize_options(self):
        super().finalize_options()

    def run(self):
        super().run()
        cache_dir = _resolve_cache_dir(self.sylphy_cache_dir)
        target_root = Path(self.install_lib)
        _write_siteconfig(target_root, cache_dir)
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        self.announce(f"[Sylphy] Cache dir set to: {cache_dir}", level=2)

class SylphyDevelop(develop):
    user_options = develop.user_options + [
        ('sylphy-cache-dir=', None, 'Custom cache directory for Sylphy (editable install)'),
    ]

    def initialize_options(self):
        super().initialize_options()
        self.sylphy_cache_dir = None

    def finalize_options(self):
        super().finalize_options()

    def run(self):
        super().run()
        cache_dir = _resolve_cache_dir(self.sylphy_cache_dir)
        project_root = Path(__file__).resolve().parent
        target_root = project_root  
        _write_siteconfig(target_root, cache_dir)
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        self.announce(f"[Sylphy] (editable) Cache dir set to: {cache_dir}", level=2)

install_requires = [
    "numpy>=1.24",
    "pandas>=2.0",
    "scikit-learn>=1.3",

    "typer>=0.9",
    "rich>=13.0",
    "appdirs>=1.4.4",
    "joblib>=1.3",

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
    "parquet": [
        "pyarrow>=14.0.0; platform_python_implementation!='PyPy'",
        "fastparquet>=2024.5.0",
    ],
    "bio": [
        "biopython>=1.80",
        "biotite>=0.38.0",
    ],
    "graph": [
        "igraph>=0.11.0",
    ],
    "cli": [
        "rich>=13.0",
    ],
    "tests": [
        "pytest>=8.4.1",
        "pytest-cov>=5.0.0",
    ],
    "dev": [
        "black>=24.3.0",
        "ruff>=0.4.0",
        "mypy>=1.8.0",
        "build>=1.0.0",
        "twine>=5.0.0",
    ],
}

# convenience meta-groups
extras_require["all"] = sorted({
    dep
    for group in ("embeddings", "reductions", "parquet", "bio", "graph", "cli")
    for dep in extras_require[group]
})

setup(
    name=PACKAGE_NAME,
    version="0.1.0",
    author="Kren AI Lab",
    author_email="krenai@umag.cl",
    description="Protein sequence representation: encoders, embeddings, and reductions.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/KREN-AI-Lab/sylphy",
    project_urls={
        "Source": "https://github.com/KREN-AI-Lab/sylphy",
        "Issues": "https://github.com/KREN-AI-Lab/sylphy/issues",
        "Documentation": "https://github.com/KREN-AI-Lab/sylphy#readme",
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
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    keywords=[
        "protein", "bioinformatics", "embeddings", "transformers",
        "ESM", "ProtT5", "ProtBERT", "k-mers", "AAindex", "dimension-reduction",
    ],
    entry_points={
        "console_scripts": [
            "sylphy=sylphy.cli.main:app",
        ],
    },
    zip_safe=False,
    cmdclass={
        "install": SylphyInstall,
        "develop": SylphyDevelop,
    },
)
