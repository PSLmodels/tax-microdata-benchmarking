from setuptools import setup, find_packages

setup(
    name="tax_microdata_benchmarking",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "policyengine_us==0.794",
        "taxcalc>=4.1.0",
        "pytest",
        "black>=24.4.2",
        "torch",
        "tensorboard",
        "jupyter-book",
        "furo",
        "scikit-learn",
    ],
)
