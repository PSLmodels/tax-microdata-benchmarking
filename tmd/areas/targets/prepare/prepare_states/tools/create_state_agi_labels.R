
suppressPackageStartupMessages(source(here::here("R", "libraries.R")))
source(here::here("R", "constants.R"))
source(here::here("R", "functions.R"))

agilabels <- read_delim(
  "agistub; agilo; agihi; agilabel
0; -9E+99; 9e99; Total
1; -9E+99; 1; Under $1
2; 1; 10000; $1 under $10,000
3; 10000; 25000; $10,000 under $25,000
4; 25000; 50000; $25,000 under $50,000
5; 50000; 75000; $50,000 under $75,000
6; 75000; 100000; $75,000 under $100,000
7; 100000; 200000; $100,000 under $200,000
8; 200000; 500000; $200,000 under $500,000
9; 500000; 1000000; $500,000 under $1,000,000
10; 1000000; 9E+99; $1,000,000 or more
", delim=";", trim_ws=TRUE)
agilabels

write_csv(agilabels, fs::path(DRAW, "agilabels.csv"))
