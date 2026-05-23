from setuptools import setup, find_packages

setup(
    name="tmd",
    version="2.1.1",
    packages=find_packages(),
    python_requires=">=3.11,<3.14",
    install_requires=[
        "taxcalc==6.6.1",
        "numpy",
        "pandas>=3.0.2",
        "clarabel",
        "scikit-learn",
        "scipy",
        "xlrd",
        "openpyxl",
        "tqdm",
        "tables",
        "requests",
        "PyYAML",
        "black>=26.1.0",
        "pycodestyle>=2.14.0",
        "pylint>=3.3.8",
        "pytest",
        "pytest-xdist",
    ],
)
