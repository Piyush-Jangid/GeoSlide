from setuptools import setup, find_packages

with open("README.md","r") as f:
    description = f.read() 

setup(
    name='GeoSlide',
    version='0.0.1',
    description='A package for determining factor of safety for critical slip circle using stability charts',
    packages=find_packages(),
    include_package_data=True,
    package_data={
        'GeoSlide': ['Data/*.xlsx'],  # Adjust the pattern to match your files
    },
    install_requires=[
        'numpy',
        'pandas',
        'openpyxl',
        'matplotlib',
        # Add other dependencies here
    ],
    long_description=description,
    long_description_content_type="text/markdown",
)
