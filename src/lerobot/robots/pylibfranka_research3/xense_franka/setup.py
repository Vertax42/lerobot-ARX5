from setuptools import setup, find_packages

setup(
    name="xense_franka",
    version="0.1.0",
    packages=find_packages(),

    install_requires=[
        "pylibfranka",
        "ruckig",
        "numpy",
        "scipy",
        "tqdm",
        "requests",
    ],

    author="dyl",
    description="Xense Franka asyncio control SDK",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",

    url="https://github.com/xensedyl/xense_franka",

    python_requires=">=3.8",
)
