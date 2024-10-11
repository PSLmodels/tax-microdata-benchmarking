# libraries ---------------------------------------------------------------

library(rlang)
library(tidyverse)
tprint <- 75  # default tibble print
options(tibble.print_max = tprint, tibble.print_min = tprint) # show up to tprint rows

library(fs)

# tools
library(vroom)
library(readxl)
library(openxlsx) # for writing xlsx files
library(lubridate)
library(RColorBrewer)
library(RcppRoll)
library(fredr)
library(tidycensus)
library(googledrive)
library(arrow)

library(jsonlite)
library(tidyjson)


# boyd libraries
# library(btools)
# library(bdata)
# library(bggtools)
# library(bmaps)

# graphics
library(scales)
library(ggbeeswarm)
library(patchwork)
library(gridExtra)
library(ggrepel)
library(ggbreak)

# tables
library(knitr)
library(kableExtra)
library(DT)
library(gt)
library(gtExtras)
library(janitor)
library(skimr)
library(vtable)

# maps
library(maps)
# https://cran.r-project.org/web/packages/usmap/vignettes/mapping.html
library(usmap)

