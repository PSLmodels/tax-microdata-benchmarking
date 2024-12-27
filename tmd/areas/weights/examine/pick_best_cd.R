

# libraries and folders -------------------------------------------------------------------

source(here::here("R", "libraries.R"))
source(here::here("R", "functions.R"))
# source(here::here("R", "functions_constants.R"))

dafpi <- "/mnt/g/.shortcut-targets-by-id/1pEdofaxeQgEeDLM8NOpo0vOGL1jT8Qa1/AFPI_2024"

dcd128 <- fs::path(dafpi, "Phase 6", "cds_128targets")
dcdp5salt <- fs::path(dafpi, "Phase 5", "Phase 5_SALT")
dcdbest <- fs::path(dafpi, "Phase 6", "cds_best")


# files -----------------------------------------------------------------

files128 <- fs::dir_ls(dcd128)
filesp5salt <- fs::dir_ls(dcdp5salt)

logfiles128 <- files128 |> str_subset(coll(".log"))
logfilesp5salt <- filesp5salt |> str_subset(coll(".log"))


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

logfiles128 <- files128 |> str_subset(coll(".log"))
logfilesp5salt <- filesp5salt |> str_subset(coll(".log"))

vals128 <- purrr::map(logfiles128, extract_min_max_values, .progress = TRUE) |> 
  list_rbind()

valsp5salt <- purrr::map(logfilesp5salt, extract_min_max_values, .progress = TRUE) |> 
  list_rbind()

stack <- bind_rows(
  vals128 |> mutate(src="cd128"),
  valsp5salt |> mutate(src="cdp5salt")
)

write_csv(stack, here::here("data_cd", "logstack.csv"))


# get saved values and explore --------------------------------------------

stack <- read_csv(here::here("data_cd", "logstack.csv"))

long <- stack |> 
  mutate(range=maxval - minval) |> 
  pivot_longer(cols = c(minval, maxval, range)) |> 
  pivot_wider(names_from = src) |> 
  relocate(cdp5salt, .before=cd128) |> 
  mutate(diff=cd128 - cdp5salt)

long |> filter(area=="ut04") # ca18

best_marked <- long |> 
  mutate(best = ifelse(cd128[label=="post" & name=="range"] < 
                         cdp5salt[label=="post" & name=="range"],
                       "cd128",
                       "cdp5salt"),
         .by=area) |> 
  mutate(bestval=case_when(best == "cd128" ~ cd128,
                           best == "cdp5salt" ~ cdp5salt))

low_threshold <- .975
high_threshold <- 1.025

# low_threshold <- .98
# high_threshold <- 1.02

best_marked <- long |> 
  mutate(
    min128 = cd128[label=="post" & name=="minval"],
    max128 = cd128[label=="post" & name=="maxval"],
    range128 = cd128[label=="post" & name=="range"],
    minp5 = cdp5salt[label=="post" & name=="minval"],
    maxp5 = cdp5salt[label=="post" & name=="maxval"],
    rangep5 = cdp5salt[label=="post" & name=="range"],
    
    best = case_when(
      min128 >= low_threshold & max128 <= high_threshold ~ "cd128",
      
      cd128[label=="post" & name=="range"] <= 
        cdp5salt[label=="post" & name=="range"] ~ "cd128",
      
      cd128[label=="post" & name=="range"] > 
        cdp5salt[label=="post" & name=="range"] ~ "cdp5salt",
      
    .default = "ERROR"),
    .by=area) |> 
  mutate(bestval=case_when(best == "cd128" ~ cd128,
                           best == "cdp5salt" ~ cdp5salt))
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
  # Create the 'best' directory if it doesn't exist
  # best_dir <- dcdbest  # file.path(base_path, "best")
  # dir.create(best_dir, showWarnings = FALSE)
  
  # For each row in the tibble
  for (i in seq_len(nrow(best_tibble))) {
    area <- best_tibble$area[i]
    bestname <- best_tibble$best[i]
    print(paste0("Row ", i, "Best source for area: ", area, " is: ", bestname))
    
    # Construct source paths
    if(bestname == "cd128"){
      source_dir <- dcd128
    } else if(bestname=="cdp5salt"){
      source_dir <- dcdp5salt
    }
    log_path <- fs::path(source_dir, paste0(area, ".log"))
    targets_path <- fs::path(source_dir, paste0(area, "_targets.csv"))
    weights_path <- fs::path(source_dir, paste0(area, "_tmd_weights.csv.gz"))
    
    # Copy files to best directory
    file.copy(
      from = c(log_path, targets_path, weights_path),
      to = dcdbest,
      overwrite = TRUE
    )
  }
}

# temp <- best |> filter(row_number() <= 8)
# copy_best_files(temp, dcdbest)
copy_best_files(best, dcdbest)


# logfile <- fs::path(dcd128, "ak00.log")
# logfile <- fs::path(dcd128, "ca18.log")
# extract_min_max_values(logfile)
# logfiles <- files |> str_subset(coll(".log"))
# df <- readLines(fs::path(dcd128, "ak00.log"))