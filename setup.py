# setup.py

from setuptools import setup, find_packages

setup(
    name='AdvancedImageStitching',
    version='1.0.0',
    packages=find_packages(),
    install_requires=[
        'opencv-python',
        'opencv-contrib-python',
        'torch',
        'torchvision',
        'matplotlib',
        'scikit-image',
        'Pillow'
    ],
    author='Austin Kuo and Joshua Lee',
    description='Implementing and Comparing Image Filters for Enhanced Image Stitching',
    url='https://github.com/yourusername/AdvancedImageStitching',  # Replace with your repository URL
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
)
