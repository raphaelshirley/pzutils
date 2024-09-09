from setuptools import setup, find_packages

setup(
    name="pzutils",  # Replace with your package name
    version="0.1.0",
    author="Raphael Shirley",
    author_email="rshirley@mpe.mpg.de",
    description="pzutils basic photometric redshift utilities.",
    packages=find_packages(where="src"),  # Specify 'src' as the source folder
    package_dir={"": "src"},  # Specify 'src' as the package root
    install_requires=[  # List your package dependencies here
        # 'numpy',  # Example dependency
        # 'requests',
    ],
    python_requires=">=3.6",  # Specify the minimum Python version
)
