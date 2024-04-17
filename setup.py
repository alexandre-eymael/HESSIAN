from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = [
    "torch",
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
    packages=find_packages(),
    include_package_data=True,
    description='Hessian Training on Vertex AI',
)