
# run from terminal (not console) with:
#   Rscript write_area_target_files.R phase6_states.json

# json files MUST be in the target_recipes folder
# Rscript test.r > output.log 2>&1

# load packages quietly -----------------------------------------------------------------

suppressPackageStartupMessages({
  library(rlang)
  library(tidyverse)
  library(here)
  library(fs)
  library(jsonlite)
})


# set folders -------------------------------------------------------------
# assume for NOW that this is called from the prepare/prepare_states folder
# later we will move it to the prepare folder
PREPDIR <- getwd() # folder in which the terminal is open; fs::path_abs("../")
# during development use the following:
#  PREPDIR <- "/home/donboyd5/Documents/python_projects/tax-microdata-benchmarking/tmd/areas/targets/prepare"

DRECIPES <- fs::path(PREPDIR, "target_recipes") 
DLIB <- fs::path(PREPDIR, "target_file_library") # output files go here

# input data
STATEINPUTS <- fs::path(PREPDIR, "prepare_states", "data", "intermediate", "enhanced_targets.csv")
CDINPUTS <- fs::path(PREPDIR, "prepare_cds", "cds", "intermediate", "cdbasefile_enhanced.csv")
                   
# output folders
STATEDIR <- fs::path(DLIB, "states")
CDDIR <- fs::path(DLIB, "cds")


# Check command-line arguments --------------------------------------------
print("checking arguments and getting data needed for target files...")
args <- commandArgs(trailingOnly = TRUE)

# Check if the correct number of arguments is provided
if (length(args) < 1) {
  stop("Error: No JSON file specified. Please provide the name of a JSON file in recipes folder as an argument.")
}

# Assign the first argument as the file path
fnrecipe <- args[1]

# ALTERNATIVE for testing: hardcode a file name -------------------------------------------
# uncomment a line below for interactive testing
# fnrecipe <- "phase6_states.json"
# fnrecipe <- "phase6_test.json"

# Check if the specified file exists in the target_recipes folder
fpath <- fs::path(DRECIPES, fnrecipe)
if (!file.exists(fpath)) {
  stop("The specified file does not exist: ", fpath)
}

# get target recipes and validate ------------------------------------------------------

recipe <- read_json(fpath) 
# print(recipe)
print(names(recipe))

#.. determine recipe type and set folders -------------------------------------

stopifnot(
  "areatype must be present and one of state or cd" = !is.null(recipe$areatype),
  "areatype must be one of state or cd" = recipe$areatype %in% c("state", "cd")
)

OUTDIR <- case_when(
  recipe$areatype == "state" ~ STATEDIR,
  recipe$areatype == "cd" ~ CDDIR,
  .default = "ERROR")

#.. check and set defaults for suffix ----
if (is.null(recipe$suffix)) {
  message("Note: Suffix value is missing. Defaulting to an empty string.")
  recipe$suffix <- ""
} else if (!recipe$suffix %in% c("", LETTERS)) {
  stop("Invalid suffix value: ", recipe$suffix, ". Valid values are an empty string or a single capital letter (A-Z).")
}

# If a CD list, check and set defaults for session variable
if (recipe$areatype == "cd") {
  if (is.null(recipe$session)) {
    message("Session value is missing for a Congressional District json file. Defaulting to 118.")
    recipe$session <- 118
  } else if (!(recipe$session %in% c(117, 118))) {
    stop("Invalid session value for Congressional District json file: ", recipe$session, ". Valid values are 117, 118.")
  }
}

# TODO: error checking on arealist

# Print updated recipe list
print(recipe)


# define variable mappings ------------------------------------------------
# allowable target variables are those mapped below
# MARS mappings let us get counts by filing status by agi range

vmap <- read_csv(fs::path(DRECIPES, "variable_mapping.csv"),
                 col_types = "ccci")

allcount_vars <- c("n1", "mars1", "mars2", "mars4")
vmap2 <- crossing(vmap, count=0:4) |> 
  # drop combinations we do not have in the SOI data
  filter(!(basesoivname == "XTOT" & (count != 0 | fstatus != 0))) |> # not allowed by definition
  filter(!(count == 1 & !basesoivname %in% allcount_vars)) |> # only allcount_vars allowed here
  filter(!(basesoivname %in% allcount_vars & count != 1))

# TODO: check whether target names are in vmap

# prepare target rules ----------------------------------------------------

# general rules, before exceptions
target_rules <- recipe$targets |> 
  purrr::map(as_tibble) |> 
  purrr::list_rbind()

# combine with agi ranges, before excluding any ranges
target_stubs <- target_rules |> 
  select(varname, scope, count, fstatus) |> 
  distinct() |> 
  cross_join(tibble(agistub=1:9)) |> # allow all agi ranges
  arrange(varname, scope, count, fstatus, agistub)

# update target_stubs to drop any agi ranges that are named for exclusion
if("agi_exclude" %in% names(target_rules)){
  target_drops <- target_rules |> 
    unnest(cols=agi_exclude)
  
  target_stubs <- target_stubs |> 
    anti_join(target_drops |> 
                rename(agistub=agi_exclude),
              join_by(varname, scope, count, fstatus, agistub))
  }

  
# create a dataframe to match against the stack data for targets
# vmap
# allcount_vars <- c("N1", "MARS1", "MARS2", "MARS4")
# allcount_vars <- c("n1", "mars1", "mars2", "mars4")
# vmap2 <- vmap |> 
#   select(varname, basesoivname, fstatus) |> 
#   mutate(basesoivname=ifelse(basesoivname %in% allcount_vars, "00100", basesoivname)) |> 
#   distinct()

# bring basesoivname in because we need it to match against targets file
targets_matchframe <- target_stubs |>
  mutate(sort=row_number() + 1) |> 
  rows_insert(tibble(varname="XTOT", scope=0, count=0, fstatus=0, agistub=0, sort=1),
              by="varname") |>
  arrange(sort) |> 
  left_join(vmap2, by = join_by(varname, fstatus, count)) |>
  relocate(sort)

# set up filters for areas, zero targets, negative targets, etc. --------------------

##.. areas filters ----
arealist <- unlist(recipe$arealist)
arealist
if(
  (length(arealist) > 1) ||
  ((length(arealist) ==1) && (arealist != "all"))
   ){
  area_filter <- expr(area %in% arealist)
} else if(length(arealist) == 1 & arealist == "all") {
  area_filter <- TRUE
} else stop('arealist must be "all" or a list of valid state or cd codes, as appropriate')

##.. zero-target filter --------
if(recipe$notzero) {
  zero_filter <- expr(target != 0)
} else zero_filter <- TRUE

#.. negative-target filter ----------
if(recipe$notnegative) {
  negative_filter <- expr(!(target < 0))
} else negative_filter <- TRUE

#.. if cd session filter ----
if(recipe$areatype == "cd") {
  session_filter <- expr(session %in% recipe$session)
} else session_filter <- TRUE


# TODO: make this flexible state or cd; load targets data -------------------------------------------------------
stack <- read_csv(STATEINPUTS, show_col_types = FALSE) |> 
  rename(area=stabbr)
# tmd18400_shared_by_soi18400" "tmd18400_shared_by_soi18400" "tmd18500_shared_by_soi18500" "tmd18500_shared_by_soi18500

# create mapped targets tibble --------------------------------------------

mapped <- targets_matchframe |>
  # inner_join -- must be in both the targets and the filtered stack
  inner_join(stack |>
              filter(!!area_filter,
                     !!zero_filter,
                     !!negative_filter,
                     session_filter) |> 
              rename(label=description) |> 
             select(-sort),
             # is sort correct?
            by = join_by(basesoivname, scope, count, fstatus, agistub),
            relationship = "many-to-many") |> 
  arrange(area, sort)

# write targets -----------------------------------------------------------

f <- function(data, group, suffix=""){
  area <- group$area |> 
    str_to_lower() |> 
    paste0(suffix)
  fname <- paste0(area, "_targets.csv")
  fpath <- fs::path(OUTDIR, fname)
  # print(fpath)
  write_csv(data, fpath)
}


print("writing targets files...")
mapped |> 
  select(area, varname, count, scope, agilo, agihi, fstatus, target) |> 
  group_by(area) |> 
  group_walk(~f(.x, .y, recipe$suffix))


ntargets <- count(mapped, area)
print("number of targets per area")
deframe(ntargets)

print("all done!")


