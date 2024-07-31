from gettext import install
from setuptools import setup, find_packages
import os

setup(
    name='amelia_inference',
    packages=find_packages(['./tools/*'], exclude=['test*']),
    version='1.0.0',
    description="Tool runing amelia's framework.",
    install_requires=[
        # "src @ ../amelia_tf",
    ],
)
