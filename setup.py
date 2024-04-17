from setuptools import find_packages
from setuptools import setup
import pathlib

REQUIRED_PACKAGES = pathlib.Path("requirements/train_requirements.txt").read_text().split("\n")

setup(
    name='hessian_trainer',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='Hessian Training on Vertex AI',
)