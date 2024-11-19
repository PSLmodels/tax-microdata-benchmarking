
# https://docs.posit.co/ide/user/ide/guide/environments/r/renv.html

# workflow:

# renv::init() to initialize a new project-local environment with a private R library

# Work in the project as normal, installing and removing new R packages as they are needed in the project,

# renv::snapshot() to save the state of the project library to the lockfile (called renv.lock),

# Continue working on your project, installing and updating R packages as needed.

# renv::snapshot() again to save the state of your project library if your attempts to update R packages were successful, or call renv::restore() to revert to the previous state as encoded in the lockfile if your attempts to update packages introduced some new problems.