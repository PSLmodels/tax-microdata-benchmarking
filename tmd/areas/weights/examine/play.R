

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

djbout <- read_csv(here::here("temp_data", "djbout.csv"))
glimpse(djbout)

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

