from setuptools import setup, find_packages

setup(
    name="tmd",
    version="2.2.0",
    packages=find_packages(),
    python_requires=">=3.12,<3.15",
    install_requires=[
        "taxcalc>=6.8.1",
        "numpy",
        "pandas>=3.0.2",
        "clarabel",
        "scikit-learn>=1.9.0",
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
    extras_require={
        # Needed only by the annual growth-factors source-data refresh
        # scripts in tmd/growfactors/update, which are never run by
        # `make data`.  See tmd/growfactors/docs/annual_update.md.
        "growfactors-update": [
            "requests_html",
            "jinja2",
        ],
    },
)
