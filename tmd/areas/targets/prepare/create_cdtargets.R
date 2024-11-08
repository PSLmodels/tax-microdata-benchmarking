
# run from terminal with:
#   Rscript create_cdtargets.R phase5.json

# Rscript test.r > output.log 2>&1

# load packages quietly -----------------------------------------------------------------

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

# Check command-line arguments --------------------------------------------
print("checking arguments and getting data needed for target files...")

args <- commandArgs(trailingOnly = TRUE)

# Check if the correct number of arguments is provided
if (length(args) < 1) {
  stop("Error: No JSON file specified. Please provide the name of a JSON file in cdrecipes as an argument.")
}

# Assign the first argument as the file path
fnrecipe <- args[1]

# ALTERNATIVE for testing: hardcode a file name -------------------------------------------
# uncomment a line below for interactive testing
# fnrecipe <- "temp.json"
# fnrecipe <- "phase5_salt.json"

# Check if the specified file exists
fpath <- here::here(CDRECIPES, fnrecipe)
if (!file.exists(fpath)) {
  stop("Error: The specified file does not exist: ", fpath)
}

# get target recipes and validate ------------------------------------------------------

cdrecipe <- read_json(fpath) 
# names(cdrecipe)

# Check and set defaults for suffix
if (is.null(cdrecipe$suffix)) {
  message("Suffix value is missing. Defaulting to an empty string.")
  cdrecipe$suffix <- ""
} else if (!cdrecipe$suffix %in% c("", LETTERS)) {
  stop("Invalid suffix value: ", cdrecipe$suffix, ". Valid values are an empty string or a single capital letter (A-Z).")
}

# Check and set defaults for session
if (is.null(cdrecipe$session)) {
  message("Session value is missing. Defaulting to 118.")
  cdrecipe$session <- 118
} else if (!(cdrecipe$session %in% c(117, 118))) {
  stop("Invalid session value: ", cdrecipe$session, ". Valid values are 117, 118.")
}

cdlist <- unlist(cdrecipe$cdlist)

# Print updated cdrecipe list
print(cdrecipe)


# define variable mappings ------------------------------------------------
# allowable target variables are those maped below
# MARS mappings let us get counts by filing status by agi range

vmap <- read_csv(file="
varname, basevname, description
XTOT, XTOT, population
c00100, v00100, agi
e00200, v00200, wages
e00300, v00300, interest income
e01700, v01700, pensions and annuities (taxable amount)
e26270, v26270, partnership and S corporation net income
e18400, v18425, state and local income or sales taxes allocated by S and L income taxes
e18500, v18500, state and local real estate taxes
", show_col_types = FALSE)

# TODO: check whether target names are in vmap


# prepare target rules ----------------------------------------------------

# general rules, before exceptions
target_rules <- cdrecipe$targets |> 
  purrr::map(as_tibble) |> 
  purrr::list_rbind()

# combine with agi ranges, before excluding any ranges
target_stubs <- target_rules |> 
  select(varname, scope, count, fstatus) |> 
  distinct() |> 
  cross_join(tibble(agistub=1:9))

# update target_stubs to drop any agi ranges that are named for exclusion
if("agi_exclude" %in% names(target_rules)){
  target_drops <- target_rules |> 
    unnest(cols=agi_exclude)
  
  target_stubs <- target_stubs |> 
    anti_join(target_drops |> 
                rename(agistub=agi_exclude),
              join_by(varname, scope, count, fstatus, agistub))
}
# target_stubs
  
# create a dataframe to match against the stack data for targets
targets_matchframe <- target_stubs |>
  mutate(sort=row_number() + 1) |> 
  rows_insert(tibble(varname="XTOT", scope=0, count=0, fstatus=0, agistub=0, sort=1),
              by="varname") |> 
  left_join(vmap, by = join_by(varname)) |> 
  mutate(basevname = case_when(fstatus == 1 ~ "MARS1",
                               fstatus == 2 ~ "MARS2",
                               fstatus == 4 ~ "MARS4",
                               .default = basevname)) |> 
  relocate(sort) |> 
  arrange(sort)


# load targets data -------------------------------------------------------
stack <- read_csv(fs::path(CDINTERMEDIATE, "cdbasefile_enhanced.csv"), show_col_types = FALSE)

# create mapped targets tibble --------------------------------------------

if(length(cdlist) > 1){
  cdfilter <- expr(statecd %in% cdlist)
} else if(length(cdlist) == 1 & cdlist == "all") {
  cdfilter <- TRUE
} else stop('cdlist must be "all" or a list of valid cd codes')

mapped <- targets_matchframe |>
  left_join(stack |>
              filter(!!cdfilter,
                     session %in% cdrecipe$session) |> 
              rename(label=description),
            by = join_by(basevname, scope, count, fstatus, agistub),
            relationship = "many-to-many") |> 
  arrange(statecd, sort)

# summary(mapped)
# skim(mapped)
# count(mapped, statecd)
stop("done with check")

# write targets -----------------------------------------------------------

f <- function(data, group, suffix=""){
  cd <- group$statecd |> 
    str_to_lower() |> 
    paste0(suffix)
  fname <- paste0(cd, "_targets.csv")
  fpath <- fs::path(CDTARGETS, fname)
  print(fpath)
  write_csv(data, fpath)
}

print("writing targets files...")
mapped |> 
  select(statecd, varname, count, scope, agilo, agihi, fstatus, target) |> 
  group_by(statecd) |> 
  group_walk(~f(.x, .y, suffix))

print("all done!")


# f <- function(target){
#   # for later -- a first step in adding income ranges as a possibility
#   # if(!"agilo" %in% names(target)) target$agilo <- -9e99
#   # if(!"agihi" %in% names(target)) target$agihi <- 9e99
#   as_tibble(target)
# }
