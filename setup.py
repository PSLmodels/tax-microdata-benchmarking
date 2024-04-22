from setuptools import setup, find_packages

setup(
    name="tax_microdata_benchmarking",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "policyengine_us>=0.744",
        "taxcalc>=3.5.0",
        "paramtools==0.18.1",
        "pytest",
        "black",
    ],
    extras_require={"reweight": ["torch"]},
)
