from setuptools import setup, find_packages

setup(
    name="tax_microdata_benchmarking",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        # "git+https://github.com/policyengine/policyengine_us@nikhilwoodruff/issue4410",
        "taxcalc>=3.5.0",
        "paramtools==0.18.1",
        "pytest",
        "black",
    ],
    # torch in dev deps
    extras_require={"reweight": ["torch"]},
)
