from setuptools import setup, find_packages

setup(
    name="tax_microdata_benchmarking",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.10,<3.13",
    install_requires=[
        "policyengine_us==1.55.0",
        "marshmallow<3.22",  # to work around paramtools bug
        "taxcalc>=4.2.1",  # requires paramtools
        "black>=24.4.2",
        "tables",  # required by policyengine_us
        "pytest",
        "pytest-xdist",
        "torch",
        "tensorboard",
        "jupyter-book",
        "furo",
        "scikit-learn",
    ],
)
