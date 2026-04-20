from setuptools import setup, find_packages

setup(
    name="tmd",
    version="2.0.0",
    packages=find_packages(),
    python_requires=">=3.11,<3.14",
    install_requires=[
        "taxcalc>=6.5.0",
        "clarabel",
        "scikit-learn",
        "scipy",
        "xlrd",
        "openpyxl",
        "black>=26.1.0",
        "pycodestyle>=2.14.0",
        "pylint>=3.3.8",
        "pytest",
        "pytest-xdist",
    ],
)
