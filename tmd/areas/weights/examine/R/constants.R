
CDICUTS <- c(-Inf, 1, 10e3, 25e3, 50e3, 75e3, 100e3, 200e3, 500e3, Inf)

# \\wsl.localhost\Ubuntu\home\donboyd5\Documents\python_projects\tax-microdata-benchmarking\tmd\storage\output
TMDDIR <- here::here("..", "..", "..", "storage", "output")
# list.files(TMDDIR)

TARGETSDIR <- here::here("..", "..", "targets")
TARGETSPREPDIR <- here::here(TARGETSDIR, "prepare")
CDDIR <- fs::path(TARGETSPREPDIR, "cds")
CDINTERMEDIATE <- fs::path(CDDIR, "intermediate")
# dir_ls(CDINTERMEDIATE)


# older stuff below ----
# WEIGHTSDIR <- here::here("..")
# list.files(TARGETSDIR)
# list.files(WEIGHTSDIR)

# CDZIPURL <- "https://www.irs.gov/pub/irs-soi/congressional2021.zip"
# CDDOCURL <- "https://www.irs.gov/pub/irs-soi/21incddocguide.docx"

# \\wsl.localhost\Ubuntu\home\donboyd5\Documents\python_projects\tax-microdata-benchmarking\tmd\areas\weights\examine
# \\wsl.localhost\Ubuntu\home\donboyd5\Documents\python_projects\tax-microdata-benchmarking\tmd\areas\targets\prepare
# TARGETSPREPDIR <- here::here("..", "..", "targets", "prepare")
# print(TARGETSPREPDIR)  # Should print the absolute path to the folder
# list.files(TARGETSPREPDIR)

# CDDIR <- here::here("cds")
# CDDIR <- fs::path(TARGETSPREPDIR, "cds")
# CDRAW <- fs::path(CDDIR, "raw_data")
# CDINTERMEDIATE <- fs::path(CDDIR, "intermediate")
# CDFINAL <- fs::path(CDDIR, "final")
# list.files(CDFINAL)
# CDDOCEXTRACT <- "cd_documentation_extracted_from_21incddocguide.docx.xlsx"
