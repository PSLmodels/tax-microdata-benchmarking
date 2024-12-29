
# look at Social Security relationships

glimpse(tmd2021)
ns(tmd2021) # e01500 e01700  e02400 c02500

tmp <- tmd2021 |> 
  filter(data_source == 1) |> 
  select(RECID, s006, e02400, c02500, agistub, agistublab)
glimpse(tmp)
skim(tmp)

tmp |> 
  summarise(e02400 = sum(e02400 * s006),
            c02500 = sum(c02500 * s006))

tmp2 <- tmp |> 
  mutate(nz2400 = e02400 != 0,
         nz2500 = c02500 != 0,
         nzboth = e02400 * c02500 != 0)
skim(tmp2)

tmp2 |> 
  filter(nzboth) |> # 37,267 obs
  select(e02400, c02500) |> 
  cor()

# e02400    c02500
# e02400 1.0000000 0.9008058
# c02500 0.9008058 1.0000000

tmp2 |> 
  filter(nz2400 | nz2500) |> # 42,664 obs
  select(e02400, c02500) |> 
  cor()

# e02400    c02500
# e02400 1.0000000 0.8520736
# c02500 0.8520736 1.0000000
  
# is there a pattern of zero c02500 by agi range?
tmp2 |> 
  summarise(
    across(c(nz2400, nz2500, nzboth, e02400, c02500), 
           \(x) sum(x * s006)),
    .by=c(agistub, agistublab)) |> 
  arrange(agistub) |> 
  mutate(ntxblpct = nz2500 / nz2400,
         txblpct = c02500 / e02400)

