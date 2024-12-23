

# utility functions -------------------------------------------------------

ht <- function(df, nrecs = 6) {
  print(utils::head(df, nrecs))
  print(utils::tail(df, nrecs))
}

ns <- function(obj){
  sort(names(obj))
}


# functions to prepare data -----------------------------------------------


save_weights <- function(){
  
  # Get and save a wide tibble of weights each time new weights are produced.
  # Once the file is created and saved, retrieval is much faster. It has one row
  # per tmd record and a column for the 2021 weight of each area (column names
  # are ak00, ..., wy00). Add a `row` (row number) column to make sorting and
  # merging safer when the file is used Save it in the `intermediate` folder of
  # this project as state_weights.rds or cd_weights.rds as appropriate. For
  # Congressional Districts the file is approximately 380 mb, for states it is
  # about 50mb. This program is time consuming for Congressional Districts (~2+
  # minutes) when weights for all 436 districts are read, so set `eval: false`
  # in the `get-save-weights` chunk after an initial run that gets the data.
  
  get_file <- function(fpath){
    fname <- fs::path_file(fpath)
    area <- stringr::word(fname, 1, sep = "_")
    print(area)
    vroom::vroom(fpath, col_select = "WT2021") |>
      rename(!!area := WT2021) # create a column named for the area, with its 2021 weight as its value
  }
  
  a <- proc.time()
  weightfiles <- fs::dir_ls(CONSTANTS$WEIGHTS_DIR) |> 
    str_subset("tmd_weights.csv.gz")
  weights <- purrr::map(weightfiles, get_file) |> 
    purrr::list_cbind() |> 
    dplyr::mutate(row=row_number()) |> 
    dplyr::relocate(row)
  b <- proc.time()
  print(b - a)
  print("saving weights file...")
  system.time(saveRDS(weights, fs::path(CONSTANTS$OUTPUT_DIR, "weights.rds")))
  print(paste0("Weights file includes: ", paste(names(weights)[-1], collapse=", ")))
}


combine_tmd_and_weights <- function(taxcalc_vars, tmd2021, weights){
  
  agilabels <- readr::read_csv(fs::path(CONSTANTS$RAW_DIR, "agilabels.csv")) # Martin Holmer uses 9e99 for top valueinstead of Inf
  agicuts <-  c(agilabels$agilo[-1], 9e99)
  
  tmdplusweights <- tmd2021 |> 
    select(RECID, data_source, MARS, us=s006, all_of(taxcalc_vars)) |> 
    mutate(row=row_number(), 
           agistub=cut(c00100, agicuts, right = FALSE, ordered_result = TRUE) |> 
             as.integer()) |>
    left_join(weights, by = join_by(row)) |> 
    relocate(row, agistub, .after = RECID) |> 
    relocate(us, .after = iitax)
  
  return(tmdplusweights)
}


get_tmdplusweights <- function(taxcalc_vars){
  # Get `cached_allvars.csv`, a saved version of data from an object constructed
  # during creation of area weights, in the file
  # `create_taxcalc_cached_files.py`. `cached_allvars.csv` is the then-current
  # tmd file with 2021 values, run through Tax-Calculator with 2021 law, written
  # as csv. It includes all Tax-Calculator input and output variables.
  fpath <-  fs::path(CONSTANTS$TMDDIR, "cached_allvars.csv")
  tmd2021 <- vroom(fpath) 
  weights <- readRDS(fs::path(CONSTANTS$OUTPUT_DIR, "weights.rds"))
  
  tmdplusweights <- combine_tmd_and_weights(taxcalc_vars, tmd2021, weights)
  return(tmdplusweights)
}

get_wtdsums <- function(tmdplusweights, taxcalc_vars){
  # Calculate and save sums by area, data_source, and AGI range. Making this
  # step efficient is crucial. For example, with 10 variables, up to 400+ areas,
  # 9 AGI categories, and 2 data_source categories, we'd gave over 70k potential
  # sums.
  
  # The approach taken here is to make a longer tmd file that has one row for
  # each tax-calculator variable of interest for each tax unit, while
  # maintaining the 400+ columns for areas, multiplying each variable's value by
  # all of the weights (400+ weighted values) and summing by groups of interest.
  # This is the second-fastest of the approaches investigated, and the easiest
  # and least-error-prone to maintain as we add variables of interest.
  
  # The resulting dataframe with sums and counts of interest is small, and easy
  # to manipulate.
  
  a <- proc.time()
  long1 <- tmdplusweights |> 
    pivot_longer(cols = all_of(taxcalc_vars),
                 names_to = "varname") |> # same name Martin Holmer uses
    relocate(varname, value, .before=us) # cds: ~ 3gb used in creating, ~ 7gb total size
  # pryr::object_size(long1)
  
  # ~ 8gb used in creating wtditems, but it is not very large
  wtdsums <- long1 |> 
    summarise(across(-c(RECID:value),
                     list(
                       sum = \(x) sum(x * value),
                       nzcount = \(x) sum(x * (value != 0)),
                       anycount = \(x) sum(x)
                     )
    ),
    .by=c(MARS, agistub, data_source, varname)) |> 
    pivot_longer(-c(MARS, agistub, data_source, varname),
                 names_to = "area_valuetype") |> 
    separate_wider_delim(area_valuetype, "_", names=c("area", "valuetype"))
  b <- proc.time()
  print(b - a)
  return(wtdsums)
}

enhance_wtdsums <- function(wtdsums){
  # sums across all marital status
  sums_plus_marstot <- wtdsums |> 
    summarise(value=sum(value), 
              .by=c(area, agistub, data_source, varname, valuetype)) |> 
    mutate(MARS=0) |> 
    bind_rows(wtdsums)
  
  # sum across all income ranges
  sums_plus_agistubtot <- sums_plus_marstot |> 
    summarise(value=sum(value), 
              .by=c(area, MARS, data_source, varname, valuetype)) |> 
    mutate(agistub=0) |> 
    bind_rows(sums_plus_marstot)
  
  # sum across all data_source values
  sums_plus_dstot <- sums_plus_agistubtot |> 
    summarise(value=sum(value),
              .by=c(area, MARS, agistub, varname, valuetype)) |> 
    mutate(data_source=9) |> 
    bind_rows(sums_plus_agistubtot) |> 
    rename(wtdsum = value)
  
  return(sums_plus_dstot)
}

save_enhanced_weighted_sums <- function(taxcalc_vars){
  
  tmdplusweights <- get_tmdplusweights(taxcalc_vars)
  wtdsums <- get_wtdsums(tmdplusweights, taxcalc_vars)
  wtdsums_enhanced <- enhance_wtdsums(wtdsums)
  
  write_csv(wtdsums_enhanced, fs::path(CONSTANTS$OUTPUT_DIR, "wtdsums_enhanced.csv"))
}



# create combined comparison file -----------------------------------------

get_combined_file <- function(){
  
  targets_available <- readr::read_csv(fs::path(CONSTANTS$TARGETS_DIR, "enhanced_targets.csv")) |> 
    dplyr::rename(area = 1) |> # fix this earlier in process
    dplyr::mutate(area = stringr::str_to_lower(area))
  
  target_files <- fs::dir_ls(CONSTANTS$WEIGHTS_DIR) |>
    stringr::str_subset("targets.csv") |> 
    stringr::str_subset("enhanced", negate = TRUE) # allows us to have enhanced_targets.csv in folder
  
  targets_used <- vroom::vroom(target_files, id="area") |> 
    dplyr::mutate(area = fs::path_file(area),
           area = stringr::word(area, sep = "_"),
           targeted = TRUE)
  
  wtdsums <- readr::read_csv(fs::path(CONSTANTS$OUTPUT_DIR, "wtdsums_enhanced.csv"))
  vmap <- readr::read_csv(fs::path(CONSTANTS$RECIPES_DIR, paste0(CONSTANTS$AREA_TYPE, "_variable_mapping.csv")))
  
  combined <- wtdsums |> 
    dplyr::rename(fstatus = MARS) |> 
    dplyr::mutate(scope = case_when(data_source == 0 ~ 2, # cps only
                             data_source == 1 ~ 1, # puf only
                             data_source == 9 ~ 0, # all records
                             .default = -9), # ERROR
           count = case_when(valuetype == "sum" ~ 0,
                             valuetype == "anycount" ~ 1,
                             valuetype == "nzcount" ~ 2,
                             .default = -9) # ERROR
    ) |> 
    left_join(vmap, relationship = "many-to-many",
              by = join_by(fstatus, varname)) |> 
    left_join(targets_available |> 
                select(-c(sort, description)),
              by = join_by(area, scope, count, fstatus, basesoivname, agistub)) |> 
    filter(!is.na(soivname)) |> 
    left_join(targets_used |> 
                select(-target),
              by = join_by(area, fstatus, varname, scope, count, agilo, agihi))|> 
    mutate(diff = wtdsum - target,
           pdiff = diff / target,
           targeted = ifelse(is.na(targeted), FALSE, TRUE)) |> 
    arrange(area, scope, count, fstatus, varname, agistub) |> 
    mutate(sort=row_number(), .by=area) |> 
    relocate(sort, .after = area) |> 
    select(area, sort, scope, count, fstatus, varname, basesoivname, agistub, agilabel, target, wtdsum, diff, pdiff, targeted, description)
  
  return(combined)
}

