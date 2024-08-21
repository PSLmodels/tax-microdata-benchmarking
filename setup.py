from setuptools import setup, find_packages

setup(
    name="tax_microdata_benchmarking",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.9,<3.13",
    install_requires=[
        "policyengine_us>=1.50.0",
        "taxcalc>=4.2.1",
        "black>=24.4.2",
        "tables",
        "pytest",
        "torch",
        "tensorboard",
        "jupyter-book",
        "furo",
        "scikit-learn",
    ],
)
