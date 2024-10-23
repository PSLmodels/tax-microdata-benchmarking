

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

tmd2021 <- readRDS(here::here("temp_data", "tmd2021.rds"))
tmdbase <- readRDS(here::here("temp_data", "tmd_base.rds"))
usweights <- readRDS(here::here("temp_data", "us_weights.rds"))
ns(usweights)

djbout <- read_csv(here::here("temp_data", "djbout.csv")) # this is tax calc output vdf from create_area_weights.py
glimpse(djbout)
ns(djbout)

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

