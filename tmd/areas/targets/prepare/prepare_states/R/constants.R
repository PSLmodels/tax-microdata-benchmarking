# NJ, NM, VA, AK, MN
PHASE6_STATES <- c("AK", "MN", "NJ", "NM", "VA")

DRAW <- here::here("data", "data_raw")
DINTERMEDIATE <- here::here("data", "intermediate")

TMDHOME <- fs::path(here::here(), "..", "..", "..", "..", "..")
# normalizePath(TMDHOME)
TMDDATA <- fs::path(TMDHOME, "tmd", "storage", "output")
# normalizePath(TMDDATA)

CDAGICUTS <- c(-Inf, 1, 10e3, 25e3, 50e3, 75e3, 100e3, 200e3, 500e3, Inf)

