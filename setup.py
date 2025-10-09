from setuptools import setup, find_packages

setup(
    name="tmd",
    version="1.0.0",
    packages=find_packages(),
    python_requires=">=3.10,<3.13",
    install_requires=[
        "policyengine_us==1.55.0",
        "tables",  # required by policyengine_us
        "taxcalc>=6.0.0",
        "scikit-learn",
        "torch",
        "tensorboard",
        "scipy",
        "jax",
        "black>=24.4.2",
        "pytest",
        "pytest-xdist",
        "jupyter-book",
    ],
)
