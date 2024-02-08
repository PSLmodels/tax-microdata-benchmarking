from setuptools import setup, find_packages

setup(
    name="tax_microdata_benchmarking",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "policyengine_us",
        "taxcalc",
        "paramtools",
        "pytest",
        "black",
    ],
)