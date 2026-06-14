"""Package configuration for the Beyond-Accuracy reproducibility repository.

The evaluation driver (``run_all.py``) depends on the vendored ``basics_cdss``
package shipped under ``src/`` so that the repository is fully self-contained:
installing it with ``pip install -e .`` makes the experiment runnable without
any external project checkout.
"""

from setuptools import find_packages, setup

setup(
    name="beyond-accuracy",
    version="1.0.0",
    author="Chatchai Tritham, Chakkrit Snae Namahoot",
    author_email="chatchait66@nu.ac.th, chakkrits@nu.ac.th",
    description=(
        "Simulation-based, harm-aware and calibration-focused pre-deployment "
        "safety evaluation framework for clinical decision support systems"
    ),
    url="https://github.com/ChatchaiTritham/Beyond-Accuracy",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "matplotlib",
        "pyyaml",
        "tqdm",
        "pydantic>=2",
        "shap>=0.42.0",
    ],
    classifiers=[
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
)
