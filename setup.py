from setuptools import setup, find_packages

setup(
    name="tax_microdata_benchmarking",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "policyengine_us==0.648.0",
        "taxcalc==3.4.1",
        "paramtools==0.18.1",
        "pytest",
        "black",
    ],
)
