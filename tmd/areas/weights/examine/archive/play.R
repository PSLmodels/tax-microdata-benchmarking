

# libraries ---------------------------------------------------------------

source(here::here("R", "libraries.R"))
source(here::here("R", "constants.R"))

phase4_statecds <- c("AK00", "DE00", "ID01", "ID02", "ME02", "MT00", "ND00", "PA08", "SD00", "WY00")


# get data ----------------------------------------------------------------

# OLD: djbout <- read_csv(here::here("temp_data", "djbout.csv")) # this is tax calc output vdf from create_area_weights.py
tmd2021 <- readRDS(here::here("temp_data", "tmd2021.rds")) # now based on djbout.csv
sum(tmd2021$s006) # is what we want to see
ns(tmd2021)
tmdbase <- readRDS(here::here("temp_data", "tmd_base.rds"))
usweights <- readRDS(here::here("temp_data", "us_weights.rds"))
ns(usweights)
area_weights <- readRDS(here::here("temp_data", "area_weights.rds"))


# debug ak00 agistub 9 c00100 amount (target 10, row 11) --------------------------------------------------------

#.. get data ----
# djbout 225,256 data rows
# same for weights
# same for masks

# ...national_population 334283385.27000004
# ...scale 2.0948260728756473e-10

# target 10
national_population <- 334283385.27000004
cdpopulation <- 732673 # row 2 xtot
unscaled_target <- 4773666000 # unscaled_target 4773666000 good
(initial_weights_scale = cdpopulation / national_population)
# 1 / cdpopulation
1 / unscaled_target
scale <- 2.0948260728756473e-10 # good
(scaled_target <- unscaled_target * scale)

iweights <- read_csv(here::here("temp_data", "unmasked_varray.csv"),
                col_names = "umv") # this should be c00100

xvalues <- read_csv(here::here("temp_data", "xvalues.csv"),
                     col_names = "x") # this should be c00100

umv <- read_csv(here::here("temp_data", "unmasked_varray.csv"),
                col_names = "umv") # this should be c00100

mask <- read_csv(here::here("temp_data", "mask.csv"),
                 col_names = "mask")

smv <- read_csv(here::here("temp_data", "scaled_masked_varray.csv"),
                col_names = "smv")


# ..check some of the data ------------------------------------------------
class(umv); class(mask); class(smv)

akweights <- area_weights |> filter(src=="ak00")

combo <- cbind(tmd2021 |> 
                 select(data_source, s006, XTOT, c00100), 
               xvalues, umv, mask, smv,
               akweights |> select(fweight=WT2021)
               ) |> 
  mutate(row=row_number(),
         iweight = s006 * initial_weights_scale) |> 
  relocate(row)

# population check - good
combo |> 
  summarise(pop=sum(s006 * XTOT)) # 334,283,385 vs expected 334,283,385.27000004

# c00100 sums - good
combo |> 
  mutate(diff = umv - c00100,
         pdiff = diff / c00100) |> 
  filter(pdiff != 0) |> 
  arrange(desc(abs(pdiff))) # super minor diffs

sum(umv) # 638131482635
sum(tmd2021$c00100) # 638131482635

# mask check
combo |> 
  mutate(maskcheck = c00100 )

# final weights check
check <- combo |> 
  mutate(fweight_check = iweight * x,
         diff=fweight_check - fweight,
         pdiff=diff / fweight)
check |> arrange(desc(abs(pdiff)))
check |> 
  summarise(fweight=sum(fweight),
            fweight_check=sum(fweight), 
            .by=data_source)


# smv looks good
smvcheck <- combo |> 
  mutate(smvcheck = c00100 * mask * scale,
         diff = smv - smvcheck,
         pdiff = diff / smvcheck) |> 
  filter(pdiff != 0)
sum(abs(smvcheck$pdiff)) # 1.118748e-12
  
smvcheck <- tmd2021$c00100 * mask * scale
class(smvcheck)
sum(smv) # 130.6638
sum(smvcheck) # 130.6638

tibble(smv=as.numeric(smv), check=as.numeric(smvcheck)) |> 
  mutate(diff=smv - check) |> 
  filter(diff != 0)



sum(mask$mask)
smv |> 
  mutate(nz=(smv != 0) * 1) |> 
  summarise(n=n(), smv=sum(smv), .by=nz)

nrow(mask)
nrow(smv)

combo |> 
  filter(mask == 1) |> 
  mutate(fweight_round=round(iweight * x, 2)) |> 
  summarise(totmod=sum(umv * fweight),
            totcalc=sum(c00100 * iweight * x),
            totcalc_round=sum(c00100 * fweight_round))

# 4,773,666,000




# weights analysis --------------------------------------------------------

check <- combo |> 
  mutate(fweight_calc = iweight * x,
         fweight_calcround = round(fweight, 2),
         rnddiff=fweight - fweight_calcround,
         rndpdiff=rnddiff / fweight_calcround,
         calcdiff=fweight - fweight_calc,
         calcpdiff=calcdiff / fweight_calc)

check |> 
  summarise(mdns006=median(s006),
            mdnfwcalc=median(fweight_calc),
            mdnfweight=median(fweight),
            .by=mask)


# analysis ----------------------------------------------------------------



glimpse(tmd3)

tmp <- tmd3 |> 
  mutate(agistub=as.integer(irange) - 1) |> 
  filter(scope==1, agistub==1)

tmp |> 
  summarise(c00100a = sum(de00 * c00100),
            c00100n = sum(de00 * (c00100 != 0)),
            nrets = sum(de00),
            e00200a = sum(de00 * e00200))

# c00100a agistub 1 amount
# -248171000 target
# -248057066. result

# c00100n agistub 1 count
# 8760 target
# 6420 result
# 8757 if we just summarize returns

# e00200a agistub 1 amount
# 40998000 target
# 40805367 result

tmp |>
  filter(fstatus==1) |> 
  summarise(mars1 = sum(de00))

# mars1 agistub 1 count
# 6040 target
# 6038  result


tmp |>
  filter(fstatus==1, c00100 != 0) |> 
  summarise(mars1 = sum(de00))

# mars1 agistub 1 count
# 6040 target
# 4300 result


# compare tmd2021 to djbout -----------------------------------------------

setdiff(names(tmd2021), names(djbout))

# simple checks on tmd2021 vs. tmdbase
sum(tmd2021$s006) # 184,024,650 why not identical to other files?
sum(tmdbase$s006) # 184,024,657 same as in US weights
sum(usweights$WT2021) / 100. # 184,024,656.95 
sum(round(usweights$WT2021 / 100)) # 184,023,729
sum(djbout$s006) # 184,024,657

sum(tmd2021$e00200) # 126,004,562,344
sum(tmdbase$e00200) # 126,004,434,333
sum(djbout$e00200) # 126,004,434,333

checkstub9 <- bind_rows(tmd2021 |> filter(c00100 >= 500e3, c00100 < 9e99) |> mutate(src="tmd"),
                        djbout |> filter(c00100 >= 500e3, c00100 < 9e99) |> mutate(src="djb")) |> 
  summarise(n=n(), agisum=sum(c00100), wtsum=sum(s006), wtdagisum=sum(s006 * c00100), .by=src)

checkstub9 |> gt()
checkstub9 |> 
  pivot_longer(-src) |> 
  pivot_wider(names_from = src) |> 
  mutate(diff = djb - tmd,
         pdiff = diff / tmd)

baddjb <- djbout |> 
  mutate(n=n(), .by=RECID) |> 
  filter(n!=1) |> 
  arrange(RECID) |> 
  relocate(RECID, n, data_source, c00100)
count(baddjb, data_source)

baddjbds0 <- djbout |> 
  filter(data_source==0) |>  # CPS records
  mutate(n=n(), .by=RECID) |> 
  filter(n!=1) |> 
  arrange(RECID) |> 
  relocate(RECID, n, data_source, c00100)

baddjbds1 <- djbout |> 
  filter(data_source==1) |> # PUF records
  mutate(n=n(), .by=RECID) |> 
  filter(n!=1) |> 
  arrange(RECID) |> 
  relocate(RECID, n, data_source, c00100)

comp <- bind_rows(
  tmd2021 |> select(RECID, data_source, c00100, s006) |> mutate(src="tmd"),
  djbout |> select(RECID, data_source, c00100, s006) |> mutate(src="djb")) |> 
  arrange(RECID, desc(src))

count(comp, src)

bad <- comp |> 
  mutate(n=n(), .by=RECID) |> 
  filter(n != 2)


comp |> 
  pivot_longer(-c(RECID, src)) |> 
  pivot_wider(names_from = src, values_from = value)

