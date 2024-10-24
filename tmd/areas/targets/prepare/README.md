# areas / targets / prepare

Contains code and data used to generate sub-national area targets files.

The code creates:

- A local website that documents the source data and the code used to create targets files. The local website is created in the _targetprep subfolder of the "prepare" folder. The website also contains details on how to selectively create target files for individual Congressional districts. A recent uploaded version of the website can be found [here](https://tmd-area-targets.netlify.app/).
- Target files for individual Congressional Districts, which may be used in conjunction with `make_all.py` to create files with area-specific weights for Tax-Calculator

Prerequisites for creating target files and the local website:

- You will need to have R and all packages used in the project installed on your computer. RStudio is a good choice for an IDE for R.
- You will need a [quarto](https://quarto.org/docs/get-started/) version 1.5.57 or later installed on your computer.

The target files and website can be created by opening a terminal in the "prepare" folder and running `quarto render`.

