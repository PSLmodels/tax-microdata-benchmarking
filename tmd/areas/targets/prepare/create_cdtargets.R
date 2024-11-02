
# source(here::here("R", "libraries.R"))
# source(here::here("R", "constants.R"))

# library(tidyverse)
# library(dplyr)
# library(purrr)
# library(jsonlite)

# Rscript test.r > output.log 2>&1


# startup -----------------------------------------------------------------
# Get command-line arguments
args <- commandArgs(trailingOnly = TRUE)

if (length(args) > 0) {
  fnrecipe <- args[1]
} else {
  fnrecipe <- "cdrecipe.json"
}

suppressPackageStartupMessages({
  library(rlang)
  library(tidyverse)
  library(here)
  library(fs)
  library(jsonlite)
})


# constants ---------------------------------------------------------------

CDDIR <- here::here("cds")
CDINTERMEDIATE <- fs::path(CDDIR, "intermediate")
CDFINAL <- fs::path(CDDIR, "final")

CDRECIPES <- fs::path("cdrecipes") 
CDTARGETS <- fs::path("cdtargets") # output files go here

# phase4cds <- c("AK00", "DE00", "ID01", "ID02", "ME02", "MT00", "ND00", "PA08", "SD00", "WY00")

# MARS mappings let us get counts by filing status by agi range
vmap <- read_csv(file="
varname, basevname
XTOT, XTOT
c00100, v00100
e00200, v00200
e00300, v00300
e01700, v01700
e26270, v26270
", show_col_types = FALSE)


# load targets data -------------------------------------------------------
# saveRDS(stack, fs::path(CDINTERMEDIATE, "cdbasefile_sessions.rds"))
# system.time(stack <- readRDS(fs::path(CDINTERMEDIATE, "cdbasefile_sessions.rds")))

stack <- read_csv(fs::path(CDINTERMEDIATE, "cdbasefile_sessions.csv"), show_col_types = FALSE)


# get target recipes ------------------------------------------------------

fpath <- here::here(CDRECIPES, fnrecipe)

cdrecipe <- read_json(fpath) 

cdlist <- unlist(cdrecipe$cdlist)
# quit(save="no", status=1, runLast=FALSE)

# create targets "recipe" tibble to merge against targets data-----------------

f <- function(target){
  # for later -- a first step in adding income ranges as a possibility
  # if(!"agilo" %in% names(target)) target$agilo <- -9e99
  # if(!"agihi" %in% names(target)) target$agihi <- 9e99
  as_tibble(target)
}

targets_tibble <- cdrecipe$targets |> 
  purrr::map(f) |> 
  purrr::list_rbind() |> 
  left_join(vmap,
            by = join_by(varname)) |> 
  mutate(basevname = case_when(fstatus == 1 ~ "MARS1",
                               fstatus == 2 ~ "MARS2",
                               fstatus == 4 ~ "MARS4",
                               .default = basevname))


# create mapped targets tibble --------------------------------------------

if(length(cdlist) > 1){
  cdfilter <- expr(statecd %in% cdlist)
} else if(length(cdlist) == 1 & cdlist == "all") {
  cdfilter <- TRUE
} else stop('cdlist must be "all" or a list of valid cd codes')

mapped <- targets_tibble |> 
  left_join(stack |>
              filter(!!cdfilter,
                     session %in% cdrecipe$session,
                     !(agistub == 0 & basevname !="XTOT")),
            by = join_by(basevname, scope, count, fstatus),
            relationship = "many-to-many") |> 
  mutate(group = case_when(basevname=="XTOT" & scope==0 & count==0 & fstatus==0 ~ 1,
                           .default = 2)) |> 
  arrange(statecd, group, scope, fstatus, varname, count, agistub) |> 
  mutate(sort=row_number(), .by=group)

# write targets -----------------------------------------------------------

f <- function(data, group){
  cd <- group$statecd |> 
    str_to_lower()
  fname <- paste0(cd, "_targets.csv")
  fpath <- fs::path(CDTARGETS, fname)
  print(fpath)
  write_csv(data, fpath)
}

print("writing targets files...")
mapped |> 
  select(statecd, varname, count, scope, agilo, agihi, fstatus, target) |> 
  group_by(statecd) |> 
  group_walk(~f(.x, .y))

print("all done!")