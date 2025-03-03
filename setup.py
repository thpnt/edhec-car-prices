from setuptools import setup, find_packages

# Read dependencies from requirements.txt
def read_requirements():
    with open("requirements.txt") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="car-price-prediction",
    version="0.1",
    packages=find_packages(),
    install_requires=read_requirements(),  # Dynamically load dependencies
)