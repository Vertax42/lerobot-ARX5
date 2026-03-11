from setuptools import setup, find_packages

setup(
    name="pyarx",
    version="0.1.0",
    description="Python bindings for the ARX5 robot arm SDK",
    packages=find_packages(),
    package_data={"pyarx": ["*.so", "*.pyd", "*.pyi"]},
    python_requires=">=3.10",
    install_requires=["pybind11", "numpy"],
)
