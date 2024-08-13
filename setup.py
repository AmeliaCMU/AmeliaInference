from setuptools import setup, find_packages


setup(
    name='amelia_inference',
    packages=find_packages(['./amelia_inference/*']),
    version='1.0.0',
    description="Tool running amelia's framework",
    url='https://github.com/AmeliaCMU/AmeliaInference',
    install_requires=[
        'python-dateutil==2.9.0',
        'numpy==1.21.2',
        'matplotlib==3.7.1',
        'imageio==2.34.0',
        'torch==2.4.0',
        'tqdm==4.65.0',
        'hydra-core==1.3.2',
        'hydra_colorlog==1.2.0',
        'contextily==1.3.0',
        'easydict==1.10',
        'opencv-python==4.7.0.72',
    ],
)
