# areas / weights / examine

Contains an R project with code used in sub-national area weights quality examination.

The R project:

1.  constructs summary data needed to analyze quality,

2.  prepares summary tables comparing weighted area sums to targets and potential targets (if an item was not targeted), and

3.  creates local web pages for Congressional Districts and states (in \_cd and \_state folders that it creates).

The user can deploy these pages to the web if desired. Online versions as of December 31, 2024 are at these links: [Congressional Districts](https://tmd-examine-cds.netlify.app/) and [states](https://tmd-examine-states.netlify.app/).

## How to run the R project to examine updated results

After cloning the tax-microdata-benchmarking repository to your local computer and creating new weights:

-   For each area type (cd or state) that you want to examine, copy `<area>_targets.csv` files, `<area>.log` files, and `<area>_tmd_weights.csv.gz` files to a folder on your computer.

-   Make sure you have opened the RStudio project and created the necessary project environment with `renv::restore()`.

-   Edit `functions_constant.R`:

    -   Define `WEIGHTS_DIR` for Congressional Districts and `WEIGHTS_DIR` for states to point to the folders for target files, log files and weights files that you established in the first step.

    -   If you want to change the areas for which individual reports are created, modify the `AREAS` vector for Congressional Districts or for states, or both.

-   Source the `render_all.R` file

-   In the console, enter `render_all(eval_data = TRUE, deploy = FALSE)`.

This will create the two local web pages. It will take a while, especially for Congressional Districts, because it must (1) create a version of the TMD file with one weight for every area, for every tax unit, and (2) for each area, for each of approximately 10 TMD variables, calculate by filer status, marital status, and AGi range the weighted sum and weighted number of returns with nonzero, positive, and negative values for the variable. After the first run, you can set `eval_data = FALSE` and it will use a previously saved summary file.
