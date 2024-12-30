
# This program examines logfiles for two separate CD runs, which will be called
# main (the preferred run if its results are acceptable) and alt (a fallback
# run). It uses a set of decision rules defined below to decide, for each CD,
# which of the two runs to use in subsequent analysis. It then copies all 3
# files for that CD (the targets, log, and weights files) from its location to
# the dcdbest folder.

# If, after reviewing the files, the user wants to use the CD best files for
# examination, they could be copied to the cds folder.

# libraries and folders -------------------------------------------------------------------

source(here::here("R", "libraries.R"))
source(here::here("R", "functions.R"))

dafpi <- "/mnt/g/.shortcut-targets-by-id/1pEdofaxeQgEeDLM8NOpo0vOGL1jT8Qa1/AFPI_2024"

# dcdp5salt <- fs::path(dafpi, "Phase 5", "Phase 5_SALT")
dcd75 <- fs::path(dafpi, "Phase 6", "cds_75targets")
dcd128 <- fs::path(dafpi, "Phase 6", "cds_128targets")
dcdbest <- fs::path(dafpi, "Phase 6", "cds_best")

# set main and alt --------------------------------------------------------

dmain <- dcd128
dalt <- dcd75

# get file names -----------------------------------------------------------------

files_main <- fs::dir_ls(dmain)
files_alt <- fs::dir_ls(dalt)

logfiles_main <- files_main |> str_subset(coll(".log"))
logfiles_alt <- files_alt |> str_subset(coll(".log"))


# function to extract -----------------------------------------------------

extract_min_max_values <- function(logfile) {
  area <- fs::path_file(logfile) |> stringr::str_remove(coll(".log"))
  print(area)
  lines <- readLines(logfile)
  
  min_lines <- grep("MINIMUM VALUE OF TARGET ACT/EXP RATIO", lines, value = TRUE)
  max_lines <- grep("MAXIMUM VALUE OF TARGET ACT/EXP RATIO", lines, value = TRUE)
  
  min_vals <- str_extract(min_lines, "-?\\d+\\.\\d+") |> as.numeric()
  max_vals <- str_extract(max_lines, "-?\\d+\\.\\d+") |> as.numeric()
  
  result <- tibble(
    area = c(area, area),
    label = c("pre", "post"),
    minval = min_vals,
    maxval = max_vals
  )
  return(result)
}


# extract values and save -------------------------------------------------

values_main <- purrr::map(logfiles_main, extract_min_max_values, .progress = TRUE) |> 
  list_rbind()

values_alt <- purrr::map(logfiles_alt, extract_min_max_values, .progress = TRUE) |> 
  list_rbind()

stack <- bind_rows(
  values_main |> mutate(src="main"),
  values_alt |> mutate(src="alt")
)

write_csv(stack, here::here("data_cd", "logstack.csv"))

count(stack, src)

# get saved values and explore --------------------------------------------

stack <- read_csv(here::here("data_cd", "logstack.csv"))

long <- stack |> 
  mutate(range=maxval - minval) |> 
  pivot_longer(cols = c(minval, maxval, range)) |> 
  pivot_wider(names_from = src) |> 
  mutate(mma=main - alt)

long |> filter(area=="ut04") # ca18


# define best set of weights for each CD ----------------------------------

# low_threshold <- .98
# high_threshold <- 1.02

low_threshold <- .975
high_threshold <- 1.025

best_marked <- long |> 
  mutate(
    min_main = main[label=="post" & name=="minval"],
    max_main = main[label=="post" & name=="maxval"],
    range_main = main[label=="post" & name=="range"],
    
    min_alt = alt[label=="post" & name=="minval"],
    max_alt = alt[label=="post" & name=="maxval"],
    range_alt = alt[label=="post" & name=="range"],
    
    best = case_when(
      (min_main >= low_threshold) & 
        (max_main <= high_threshold) ~ "main",
      range_main <= range_alt ~ "main",
      range_main > range_alt ~ "alt",
      .default = "ERROR"),
    .by=area) |> 
  
  mutate(bestval=case_when(best == "main" ~ main,
                           best == "alt" ~ alt))
# count(best_marked, best)

best <- best_marked |> 
  filter(label=="post") |> 
  select(area, label, name, best, bestval) |> 
  pivot_wider(names_from = name, values_from = bestval)

count(best, best)

best_marked |> filter(area=="ca18")
best_marked |> filter(area=="ca36")


# copy best files ---------------------------------------------------------

copy_best_files <- function(best_tibble, dcdbest) {
  # For each row in the tibble
  for (i in seq_len(nrow(best_tibble))) {
    area <- best_tibble$area[i]
    bestname <- best_tibble$best[i]
    print(paste0("Row ", i, "Best source for area: ", area, " is: ", bestname))
    
    # Construct source paths
    if(bestname == "main"){
      source_dir <- dmain
    } else if(bestname=="alt"){
      source_dir <- dalt
    }
    log_path <- fs::path(source_dir, paste0(area, ".log"))
    targets_path <- fs::path(source_dir, paste0(area, "_targets.csv"))
    weights_path <- fs::path(source_dir, paste0(area, "_tmd_weights.csv.gz"))
    print(source_dir)
    
    # Copy files to best directory
    file.copy(
      from = c(log_path, targets_path, weights_path),
      to = dcdbest,
      overwrite = TRUE
    )
  }
}

# test run up to the point where we've seen main and alt
# temp <- best |> filter(row_number() <= 7)
# copy_best_files(temp, dcdbest)

copy_best_files(best, dcdbest)

