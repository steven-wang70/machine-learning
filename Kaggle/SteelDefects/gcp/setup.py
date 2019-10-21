from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = ['Keras==2.3.1', 
					 'matplotlib', 
					 'opencv-python', 
					 'image-classifiers==0.2.0', 
					 'segmentation_models==0.2.1',
					 'gcsfs']

setup(
    name='trainer',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='Keras trainer application'
)
