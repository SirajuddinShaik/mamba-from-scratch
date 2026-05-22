from setuptools import setup, find_packages

setup(
    name="mamba",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "triton>=2.1.0",
        "einops>=0.7.0",
        "causal-conv1d>=1.4.0",
    ],
    python_requires=">=3.8",
    author="Mamba Implementation",
    description="Clean implementation of Mamba: Linear-Time Sequence Modeling with Selective State Spaces",
)
