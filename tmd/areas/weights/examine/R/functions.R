

# utility functions -------------------------------------------------------

ht <- function(df, nrecs = 6) {
  print(utils::head(df, nrecs))
  print(utils::tail(df, nrecs))
}

ns <- function(obj){
  sort(names(obj))
}


# functions to prepare data -----------------------------------------------

combine_tmd_and_weights <- function(areatype, taxcalc_vars, tmd2021, weights){
  
  if(areatype=="state") {
    agicuts <- state_cuts
  } else agicuts <- CDICUTS
  
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

save_weights <- function(areadir, areatype){
  
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
  weightfiles <- fs::dir_ls(areadir) |> 
    str_subset("tmd_weights.csv.gz")
  weights <- purrr::map(weightfiles, get_file) |> 
    purrr::list_cbind() |> 
    dplyr::mutate(row=row_number()) |> 
    dplyr::relocate(row)
  b <- proc.time()
  print(b - a)
  print("saving weights file...")
  system.time(saveRDS(weights, here::here("intermediate", paste0(areatype, "_weights.rds"))))
  print(paste0("Weights file includes: ", paste(names(weights)[-1], collapse=", ")))
}


save_weighted_tmd_sums <- function(areatype, taxcalc_vars, TMDDATA){
  # Get `cached_allvars.csv`, a saved version of data from an object constructed
  # during creation of area weights, in the file
  # `create_taxcalc_cached_files.py`. `cached_allvars.csv` is the then-current
  # tmd file with 2021 values, run through Tax-Calculator with 2021 law, written
  # as csv. It includes all Tax-Calculator input and output variables.
  fpath <-  fs::path(TMDDATA, "cached_allvars.csv")
  tmd2021 <- vroom(fpath) 
  weights <- readRDS(here::here("intermediate", paste0(areatype, "_weights.rds")))
  
  tmdplusweights <- combine_tmd_and_weights(areatype, taxcalc_vars, tmd2021, weights)
  save_wtditems(tmdplusweights, "state")
}



save_wtditems <- function(tmdplusweights, areatype){
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
                 names_to = "variable") |> 
    relocate(variable, value, .before=us) # ~ 3gb used in creating, ~ 7gb total size
  # pryr::object_size(long1)
  
  # ~ 8gb used in creating wtditems, but it is not very large
  wtditems <- long1 |> 
    summarise(across(-c(RECID:value),
                     list(
                       sum = \(x) sum(x * value),
                       nzcount = \(x) sum(x * (value != 0)),
                       anycount = \(x) sum(x)
                     )
    ),
    .by=c(MARS, agistub, data_source, variable)) |> 
    pivot_longer(-c(MARS, agistub, data_source, variable),
                 names_to = "area_valtype") |> 
    separate_wider_delim(area_valtype, "_", names=c("area", "valtype"))
  b <- proc.time()
  print(b - a)
  write_csv(wtditems, here::here("intermediate", paste0(areatype, "_wtditems.csv")))
}

save_enhanced_weighted_sums <- function(areatype, taxcalc_vars, TMDDATA){
  save_weighted_tmd_sums(areatype, taxcalc_vars, TMDDATA) # basic sums
  
  wtditems <- read_csv(here::here("intermediate", paste0(areatype, "_wtditems.csv"))) 
  
  # sums across all marital status
  sums_plus_marstot <- wtditems |> 
    summarise(value=sum(value), 
              .by=c(area, agistub, data_source, variable, valtype)) |> 
    mutate(MARS=0) |> 
    bind_rows(wtditems)
  
  # sum across all income ranges
  sums_plus_agistubtot <- sums_plus_marstot |> 
    summarise(value=sum(value), 
              .by=c(area, MARS, data_source, variable, valtype)) |> 
    mutate(agistub=0) |> 
    bind_rows(sums_plus_marstot)
  
  # sum across all data_source values
  sums_plus_dstot <- sums_plus_agistubtot |> 
    summarise(value=sum(value), 
              .by=c(area, MARS, agistub, variable, valtype)) |> 
    mutate(data_source=9) |> 
    bind_rows(sums_plus_agistubtot)
  
  write_csv(sums_plus_dstot, here::here("intermediate", paste0(area_type, "_wtditems_enhanced.csv")))
}

