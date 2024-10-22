

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


