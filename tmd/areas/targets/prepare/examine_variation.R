
source(here::here("R", "libraries.R"))
source(here::here("R", "constants.R"))

phase4cds <- c("AK00", "DE00", "ID01", "ID02", "ME02", "MT00", "ND00", "PA08", "SD00", "WY00")


agibins <- read_csv(fs::path(CDINTERMEDIATE, "cd_agi_bins.csv"))
cddata <- read_csv(fs::path(CDINTERMEDIATE, "cddata_wide_clean.csv"))
base <- read_csv(fs::path(CDINTERMEDIATE, "cdbasefile_sessions.csv"))

glimpse(cddata)
count(cddata, rectype)

glimpse(base)
count(base, agistub, agilo, agihi, agirange)
names(base)

# wage share
wshare <- base |> 
  filter(statecd %in% phase4cds, session==117) |> 
  filter(count==0, scope==1, fstatus==0, vname %in% c("A00100", "A00200")) |> 
  select(rectype, stabbr, statecd, agistub, agirange, vname, target) |> 
  pivot_wider(names_from = vname, values_from = target) |> 
  mutate(wageshare=A00200 / A00100)

wshare |> filter(agistub==0) |> arrange(desc(wageshare)) # all
wshare |> filter(agistub==1) |> arrange(desc(wageshare)) # < $1; -08 -20
wshare |> filter(agistub==2) |> arrange(desc(wageshare)) # 1-10k; 66-83  AK, ND
wshare |> filter(agistub==3) |> arrange(desc(wageshare)) # 10-25k; 56-77, 21, PA08, ID02
wshare |> filter(agistub==4) |> arrange(desc(wageshare)) # 25-50k; 75-82, 7
wshare |> filter(agistub==5) |> arrange(desc(wageshare)) # 50-75k 70-77, 
wshare |> filter(agistub==6) |> arrange(desc(wageshare)) # 75-100k 67-74
wshare |> filter(agistub==7) |> arrange(desc(wageshare)) # 100-200k 64-70
wshare |> filter(agistub==8) |> arrange(desc(wageshare)) # 200-500k  45-61
wshare |> filter(agistub==9) |> arrange(desc(wageshare)) # 500+k 12-30


# test PA08 and ID02
wshare |>
  filter(statecd %in% c("PA08", "ID02")) |> 
  select(statecd, agistub, agirange, wageshare) |> 
  pivot_wider(names_from = statecd, values_from = wageshare) |> 
  mutate(pamid=PA08 - ID02)
# agistub agirange                   ID02   PA08   pamid
# <dbl> <chr>                     <dbl>  <dbl>   <dbl>
#   1       0 Total                    0.564   0.645  0.0806
# 2       1 Under $1                -0.0809 -0.174 -0.0932
# 3       2 $1 under $10,000         0.821   0.675 -0.146 
# 4       3 $10,000 under $25,000    0.771   0.562 -0.209 
# 5       4 $25,000 under $50,000    0.817   0.750 -0.0672
# 6       5 $50,000 under $75,000    0.768   0.730 -0.0373
# 7       6 $75,000 under $100,000   0.737   0.697 -0.0400
# 8       7 $100,000 under $200,000  0.680   0.698  0.0179
# 9       8 $200,000 under $500,000  0.517   0.597  0.0795
# 10       9 $500,000 or more         0.185   0.297  0.113 






cddata |>
  filter(rectype %in% c("cdstate", "cd", "DC")) |> 
  mutate(wageshare=A00200 / A00100) |> 
  select(STATE, CONG_DISTRICT, AGI_STUB, agirange, wageshare)

cddata |>
  filter(rectype %in% c("cdstate", "cd", "DC")) |> 
  filter(statecd %in% phase4cds)
  mutate(wageshare=A00200 / A00100) |> 
  select(STATE, CONG_DISTRICT, AGI_STUB, agirange, wageshare)

cddata |>
  filter(rectype %in% c("cdstate", "cd", "DC")) |> 
  mutate(wageshare=A00200 / A00100) |> 
  select(STATE, CONG_DISTRICT, agirange, wageshare) |> 
  pivot_wider(names_from = agirange, values_from = wageshare) |> 
  summary()

cddata |>
  filter(rectype %in% c("cdstate", "cd", "DC")) |> 
  mutate(wageshare=A00200 / A00100) |> 
  select(STATE, CONG_DISTRICT, AGI_STUB, agirange, wageshare) |> 
  filter(AGI_STUB==4) |> 
  filter(wageshare < .62 | wageshare >.89)

cddata |>
  filter(rectype %in% c("cdstate", "cd", "DC")) |> 
  mutate(wageshare=A00200 / A00100) |> 
  select(STATE, CONG_DISTRICT, AGI_STUB, agirange, wageshare) |> 
  filter(AGI_STUB==7) |> 
  filter(wageshare < .44 | wageshare >.86)


