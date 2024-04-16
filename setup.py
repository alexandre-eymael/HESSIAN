from setuptools import find_packages
from setuptools import setup
import setuptools

from distutils.command.build import build as _build
import subprocess

import pathlib

PACKAGES = ["models"]

REQUIRED_PACKAGES = [
    #"torch", # Not needed because it's already installed on Vertex AI
    "torchvision",
    "numpy",
    "pandas",
    "Pillow",
    "scikit-learn",
    "wandb",
    "tqdm",
    "datasets",
    "cloudml-hypertune",
    "kaggle",
    "google-cloud-secret-manager"
]

setup(
    name='hessian_trainer',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    packages=PACKAGES,
    include_package_data=True,
    description='Hessian Training on Vertex AI',
)