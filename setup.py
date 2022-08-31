import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("protpretrain/version.py") as infile:
    exec(infile.read())

setuptools.setup(
    name="ProtPreTrain",
    version=version,
    author="Ilya Senatorov",
    author_email="ilya.senatorov@helmholtz-hips.de",
    description="Protein structure pretraining",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ilsenatorov/ProtPreTrain",
    packages=setuptools.find_packages(),
    keywords=[
        "deep-learning",
        "pytorch",
        "foundation",
        "self-supervised",
        "residue-interaction-network",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.8",
        "Development Status :: 3 - Alpha",
    ],
    python_requires=">=3.8",
)
