# functions_constants.R

# constants needed to switch between a run for Congressional Districts or for states

# For During development, 
# \\wsl.localhost\Ubuntu-24.04\home\donboyd5\Documents\python_projects\tax-microdata-benchmarking\tmd\areas\targets\prepare\prepare_states\data\intermediate
# \\wsl.localhost\Ubuntu-24.04\home\donboyd5\Documents\python_projects\tax-microdata-benchmarking\tmd\areas\targets\prepare\prepare_cds\cds\intermediate


get_constants <- function(area_type) {
  # Validate input
  valid_area_types <- c("state", "cd")
  if (!area_type %in% valid_area_types) {
    stop("area_type must be one of: ", paste(valid_area_types, collapse = ", "))
  }
  
  # Common constants
  constants <- list(
    AREA_TYPE = area_type,
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
                           # NOTE: WEIGHTS_DIR is system-specific. 
                           # During development it was a local Google Drive folder in the Windows file system.
                           
                           "state" = list(
                             WEIGHTS_DIR = "/mnt/g/.shortcut-targets-by-id/1pEdofaxeQgEeDLM8NOpo0vOGL1jT8Qa1/AFPI_2024/Phase 6/states/",
                             RAW_DIR = fs::path(constants$TMDAREAS, "targets", "prepare", "prepare_states", "data", "data_raw"),
                             TARGETS_DIR = fs::path(constants$TMDAREAS, "targets", "prepare", "prepare_states", "data", "intermediate"),
                             OUTPUT_DIR = here::here("data_state"),
                             LONG_NAME = "states",
                             AREAS = c("AK", "MN", "NJ", "NM", "VA", "SC") |> 
                               stringr::str_to_lower() # Phase 6 states plus SC
                           ),
                           
                           "cd" = list(
                             WEIGHTS_DIR = "/mnt/g/.shortcut-targets-by-id/1pEdofaxeQgEeDLM8NOpo0vOGL1jT8Qa1/AFPI_2024/Phase 6/cds/",
                             RAW_DIR = fs::path(constants$TMDAREAS, "targets", "prepare", "prepare_cds", "data", "data_raw"),
                             TARGETS_DIR = fs::path(constants$TMDAREAS, "targets", "prepare", "prepare_cds", "data", "intermediate"),
                             OUTPUT_DIR = here::here("data_cd"),
                             SESSION = 118,
                             LONG_NAME = "Congressional Districts",
                             AREAS = c("AK00", "DE00", "ID01", "ID02", "ME02", "MT00", "ND00", "PA08", "SD00", "WY00") |> 
                               stringr::str_to_lower()
                           )
  )
  
  # Combine common and area-specific constants
  c(constants, area_constants)
}

# normalizePath(TMDHOME)
# normalizePath(TMDDIR)
# normalizePath(TMDAREAS)
# normalizePath(STATEINTERMEDIATE)
# normalizePath(CDRAW)

