
CDZIPURL <- "https://www.irs.gov/pub/irs-soi/congressional2021.zip"
CDDOCURL <- "https://www.irs.gov/pub/irs-soi/21incddocguide.docx"

CDDIR <- here::here("cds")
CDRAW <- fs::path(CDDIR, "raw_data")
CDINTERMEDIATE <- fs::path(CDDIR, "intermediate")
CDFINAL <- fs::path(CDDIR, "final")

CDDOCEXTRACT <- "cd_documentation_extracted_from_21incddocguide.docx.xlsx"

TMDHOME <- fs::path(here::here(), "..", "..", "..", "..", "..")
# normalizePath(TMDHOME)
TMDDATA <- fs::path(TMDHOME, "tmd", "storage", "output")
# normalizePath(TMDDATA)
