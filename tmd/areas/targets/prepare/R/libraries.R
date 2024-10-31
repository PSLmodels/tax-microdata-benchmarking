# libraries ---------------------------------------------------------------

library(renv)
library(here)

library(DT)
library(fs)
library(gt)
library(knitr)
library(readxl)
library(skimr)
library(stringr)
library(tidyverse)
# includes: dplyr, forcats, ggplot2, lubridate, purrr, stringr, tibble, tidyr

tprint <- 75  # default tibble print
options(tibble.print_max = tprint, tibble.print_min = tprint) # show up to tprint rows

# census_api_key("b27cb41e46ffe3488af186dd80c64dce66bd5e87", install = TRUE) # stored in .Renviron
# libraries needed for census population
library(sf)
library(tidycensus)
library(tigris)
options(tigris_use_cache = TRUE)


# possible libraries ------------------------------------------------------

# library(rlang)
# library(tidyverse)
# tprint <- 75  # default tibble print
# options(tibble.print_max = tprint, tibble.print_min = tprint) # show up to tprint rows
#  
# library(fs)
 
# tools
# library(vroom)
# library(readxl)
# library(openxlsx) # for writing xlsx files
# library(lubridate)
# library(RColorBrewer)
# library(RcppRoll)
# library(fredr)
# library(tidycensus)
# library(googledrive)
# library(arrow)
# 
# library(jsonlite)
# library(tidyjson)
# 
# 
# # boyd libraries
# # library(btools)
# # library(bdata)
# # library(bggtools)
# # library(bmaps)
# 
# # graphics
# library(scales)
# library(ggbeeswarm)
# library(patchwork)
# library(gridExtra)
# library(ggrepel)
# library(ggbreak)
# 
# # tables
# library(knitr)
# library(kableExtra)
# library(DT)
# library(gt)
# library(gtExtras)
# library(janitor)
# library(skimr)
# library(vtable)
# 
# # maps
# library(maps)
# # https://cran.r-project.org/web/packages/usmap/vignettes/mapping.html
# library(usmap)

