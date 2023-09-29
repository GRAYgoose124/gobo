from setuptools import setup, find_packages

setup(
    name="gogogo",
    version="0.1.0",
    description="Homebrew Go AI",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Grayson Miller",
    author_email="grayson.miller124@gmail.com",
    packages=find_packages(),
    python_requires=">=3.9, <3.13",
    install_requires=[
        "tensorflow==2.13.0",
        "pysgf==0.8.0",
        "numpy==1.25.2",
        "matplotlib==3.16.2",
    ],
    entry_points={
        "console_scripts": [
            "gobo=gogogo.__main__:main",
        ],
    },
)
