
state_stubs <- read_csv(
  'agistub,agilo,agihi,agilabel
0,-9e99,9e99,Total
1,-9e99,1,Under $1
2,1,10000,"$1 under $10,000"
3,10000,25000,"$10,000 under $25,000"
4,25000,50000,"$25,000 under $50,000"
5,50000,75000,"$50,000 under $75,000"
6,75000,100000,"$75,000 under $100,000"
7,100000,200000,"$100,000 under $200,000"
8,200000,500000,"$200,000 under $500,000"
9,500000,1000000,"$500,000 under $1,000,000"
10,1000000,9e99,"$1,000,000 or more
')
# state_stubs
state_cuts <- c(state_stubs$agilo[-1], Inf)

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
