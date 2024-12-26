# initial WEIGHTSDIR /home/donboyd5/Documents/python_projects/scratch/phase5_salt
# "\\wsl.localhost\Ubuntu-24.04\home\donboyd5\Documents\python_projects\backups\tmd_items_2024-12-20\states"
# pp <- "/home/donboyd5/Documents/python_projects"
# WEIGHTSDIR <- fs::path(pp, "backups/tmd_items_2024-12-20/states")

# \\wsl.localhost\Ubuntu-24.04\home\donboyd5\Documents\python_projects\tax-microdata-benchmarking\tmd\areas\targets\prepare\prepare_states\data\intermediate
# \\wsl.localhost\Ubuntu-24.04\home\donboyd5\Documents\python_projects\tax-microdata-benchmarking\tmd\areas\targets\prepare\prepare_cds\cds\intermediate

# functions_constants.R
get_constants <- function(area_type) {
  # Validate input
  valid_area_types <- c("state", "cd", "test")
  if (!area_type %in% valid_area_types) {
    stop("area_type must be one of: ", paste(valid_area_types, collapse = ", "))
  }
  
  # Common constants
  constants <- list(
    AREA_TYPE = area_type,
    PHASE6_STATES = c("AK", "MN", "NJ", "NM", "VA"),
    TMDHOME = fs::path(here::here(), "..", "..", "..", ".."),
    TMDDIR = NULL,  # Will be derived
    TMDAREAS = NULL, # Will be derived
    RECIPES_DIR = NULL # Will be derived
  )
  
  # Derive dependent common constants
  constants$TMDDIR <- fs::path(constants$TMDHOME, "tmd", "storage", "output")
  constants$TMDAREAS <- fs::path(constants$TMDHOME, "tmd", "areas")
  constants$RECIPES_DIR <- fs::path(constants$TMDAREAS, "targets", "prepare", "target_recipes")
  
  # area_type-specific constants
  area_constants <- switch(area_type,
                           "state" = list(
                             # local Google Drive folder
                             WEIGHTS_DIR = "/mnt/g/.shortcut-targets-by-id/1pEdofaxeQgEeDLM8NOpo0vOGL1jT8Qa1/AFPI_2024/Phase 6/states/",
                             RAW_DIR = fs::path(constants$TMDAREAS, "targets", "prepare", "prepare_states", "data", "data_raw"),
                             TARGETS_DIR = fs::path(constants$TMDAREAS, "targets", "prepare", "prepare_states", "data", "intermediate"),
                             OUTPUT_DIR = here::here("data_state")
                           ),
                           "cd" = list(
                             WEIGHTS_DIR = "/mnt/g/.shortcut-targets-by-id/1pEdofaxeQgEeDLM8NOpo0vOGL1jT8Qa1/AFPI_2024/Phase 6/cds/",
                             RAW_DIR = fs::path(constants$TMDAREAS, "targets", "prepare", "prepare_cds", "data", "data_raw"),
                             TARGETS_DIR = fs::path(constants$TMDAREAS, "targets", "prepare", "prepare_cds", "data", "intermediate"),
                             OUTPUT_DIR = here::here("data_cd")
                           ),"test" = dplyr::lst( # allows reference to earlier elements
                             WEIGHTS_DIR = "/mnt/e/test_states/",
                             RAW_DIR = WEIGHTS_DIR,
                             TARGETS_DIR = WEIGHTS_DIR,
                             OUTPUT_DIR = WEIGHTS_DIR
                           ),
  )
  
  # Combine common and area-specific constants
  c(constants, area_constants)
}

# normalizePath(TMDHOME)
# normalizePath(TMDDIR)
# normalizePath(TMDAREAS)
# normalizePath(STATEINTERMEDIATE)
# normalizePath(CDRAW)

