from setuptools import setup, find_packages

setup(
    name="tmd",
    version="1.2.0",
    packages=find_packages(),
    python_requires=">=3.10,<3.13",
    install_requires=[
        "policyengine_us==1.55.0",
        "tables",  # required by policyengine_us
        "taxcalc>=6.4.0",
        "behresp",  # required by taxcalc
        "clarabel",
        "scikit-learn",
        "torch",
        "tensorboard",
        "scipy",
        "jax",
        "black>=26.1.0",
        "pycodestyle>=2.14.0",
        "pylint>=3.3.8",
        "pytest",
        "pytest-xdist",
        "jupyter-book",
    ],
)
