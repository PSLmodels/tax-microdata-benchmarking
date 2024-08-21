from setuptools import setup, find_packages

setup(
    name="tax_microdata_benchmarking",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.10,<3.13",
    install_requires=[
        "policyengine_us==1.45.0",
        "marshmallow<=3.19.0",  # to work around paramtools bug
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
