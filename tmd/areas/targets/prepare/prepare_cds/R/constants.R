
CDZIPURL <- "https://www.irs.gov/pub/irs-soi/congressional2021.zip"
CDDOCURL <- "https://www.irs.gov/pub/irs-soi/21incddocguide.docx"

CDDATA <- here::here("data")
CDRAW <- fs::path(CDDATA, "data_raw")
CDINTERMEDIATE <- fs::path(CDDATA, "intermediate")

CDDOCEXTRACT <- "cd_documentation_extracted_from_21incddocguide.docx.xlsx"

TMDHOME <- fs::path(here::here(), "..", "..", "..", "..", "..")
# normalizePath(TMDHOME)
TMDDATA <- fs::path(TMDHOME, "tmd", "storage", "output")
# normalizePath(TMDDATA)

CDAGICUTS <- c(-Inf, 1, 10e3, 25e3, 50e3, 75e3, 100e3, 200e3, 500e3, Inf)
