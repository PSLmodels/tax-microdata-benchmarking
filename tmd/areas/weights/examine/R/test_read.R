

source(here::here("R", "libraries.R"))
source(here::here("R", "constants.R"))

ns <- function(obj){
  sort(names(obj))
}

mhhi <- get_acs(
  geography = "congressional district",
  # variables = "S1903_C03",  # Median household income variable
  table = "S1903",
  year = 2022,
  survey = "acs1"  # Consider using 5-year estimates for better coverage, especially in smaller areas
)

mhhi2022 <- get_acs(
  geography = "congressional district",
  variables = "S1903_C03_001",  # Median household income variable S1903_C03_001E
  year = 2022,
  survey = "acs1"  # Consider using 5-year estimates for better coverage, especially in smaller areas
)
# write_csv(cdpop1year, fs::path(CDRAW, "cdpop1year_acs.csv"))
# "S1903_C03_001E"

# tmd2021_cache.csv -------------------------------------------------------
TMDDIR <- here::here("..", "..", "..", "storage", "output")
fpath <-  fs::path(TMDDIR, "tmd2021_cache.csv")
# tmd2021 <- read_csv(fpath)
tmd2021 <- vroom(fpath)
ns(tmd2021)
tmd2021 |> filter(row_number() < 20) |> select(RECID, s006, c00100)


# weights -----------------------------------------------------------------


weightfiles <- dir_ls(here::here("temp_data")) |> str_subset("tmd_weights.csv.gz")
# weights_long <- vroom(weightfiles, col_select = "WT2021")
# all_columns <- purrr::map(weightfiles, \(x) vroom(x, col_select = "WT2021")) |> 
#   list_cbind()
f <- function(fpath){
  cd <- str_sub(fs::path_file(fpath), 1, 4)
  print(cd)
  vroom(fpath, col_select = "WT2021") |> 
    rename(!!cd := WT2021)
}

a <- proc.time()
weights <- test <- purrr::map(weightfiles, f) |> 
  list_cbind() |> 
  mutate(row=row_number()) |> 
  relocate(row)
b <- proc.time()
b - a # 79 seconds

system.time(saveRDS(weights, here::here("intermediate", "weights.rds"))) # 23 seconds


# summaries ---------------------------------------------------------------
system.time(cdweights <- readRDS(here::here("intermediate", "weights.rds"))) # 4.5 secs, 390 mb
# names(cdweights)

ns(tmd2021)
df <- tmd2021 |> 
  select(RECID, data_source, us=s006, c00100, e00200, e00300, e26270, iitax) |> 
  mutate(row=row_number(), wtdn=1) |> 
  left_join(cdweights, by = join_by(row))
  
glimpse(df)
df[1:5, 1:5]

a <- proc.time()
dfxsums <- df |> 
  summarise(across(c(us, ak00:wy00),
                   list(wtdn = \(x) sum(x * wtdn),
                        c00100 = \(x) sum(x * c00100),
                        e00200 = \(x) sum(x * e00200),
                        e00300 = \(x) sum(x * e00300),
                        e26270 = \(x) sum(x * e26270),
                        iitax = \(x) sum(x * iitax))),
            .by=data_source) |>  
  pivot_longer(-data_source, values_to = "sum") |> 
  separate(name, into=c("statecd", "variable"))
b <- proc.time()
b - a # 2.2

dfxsums

avgs <- dfxsums |> 
  mutate(avg=sum / sum[variable=="wtdn"],
         .by=c(data_source, statecd))


awide <- avgs |> 
  select(-sum) |> 
  pivot_wider(names_from = variable, values_from = avg)

awide |> 
  arrange(desc(iitax)) |> 
  slice_head(n=5)

awide |> 
  arrange(desc(iitax)) |> 
  slice_tail(n=5)


variable_names <- c("c00100", "e00200", "e00300", "iitax")

# Generic approach using a function to pass the list of variable names
dfxsums <- df |> 
  summarise(across(c(s006, ak00:wy00), 
                   list = setNames(
                     lapply(variable_names, function(var) {
                       function(x) sum(x * .data[[var]])
                     }), 
                     variable_names
                   )),
            .by = data_source)


variable_names <- c("c00100", "e00200", "e00300", "iitax")

# Generic approach using purrr::map to iterate over variables
dfxsums <- df |> 
  summarise(
    # Summarise over s006, ak00 to wy00
    across(c(s006, ak00:wy00), sum),
    # Add summary for each variable dynamically
    !!!map(variable_names, ~summarise(
      df, !!sym(.x) := sum(.data[[.x]] * df[[.x]])
    )),
    .by = data_source
  )

dfxsums[, 1:5]

variable_names <- c("c00100", "e00200", "e00300", "iitax")

# Create a list of functions to apply for each variable
variable_functions <- setNames(
  map(variable_names, ~function(x) sum(x * .data[[.x]])),
  variable_names
)

# Use summarise with across to generate suffixes dynamically
dfxsums <- df |> 
  summarise(
    across(
      c(s006, ak00:wy00), 
      .fns = variable_functions,
      .names = "{.fn}_{.col}"
    ),
    .by = data_source
  )



sum_weighted <- function(df, weight_vars, value_vars) {
  df |> 
    summarise(across(all_of(weight_vars),
                     \(x) sapply(value_vars, \(y) sum(x * .data[[y]])),
                     .names = "{.col}_{.fn}"),
              .by = data_source)
}

# Usage
weight_vars <- c("s006", paste0(letters[1:2], "k00"))
value_vars <- c("c00100", "e00200", "e00300", "iitax")

dfxsums2 <- sum_weighted(df, weight_vars, value_vars)



df2 <- df |> 
  summarise(across(c(s006, ak00:wy00),
                   \(x) sum(x * c00100)),
            .by=data_source)
  
df3 <- df2 |> 
  pivot_longer(cols = -data_source) |> 
  pivot_wider(names_from = data_source, names_prefix = "ds") |> 
  mutate(dssum=ds0 + ds1)

df3 |> 
  filter(name=="ny11") |> 
  kable()

glimpse(df)
df2 <- df |> 
  summarise(
    # nested across
    across(c(ak00:wy00), 
           \(weight_col) across(c(wtdn, c00100, iitax),
                                \(value_col) sum(weight_col * value_col, na.rm = TRUE)), 
                                .names = "{.col}_{.fn}"), 
    .by = data_source)


df2 <- df |> 
  summarise(
    # nested across
    across(c(ak00:wy00), 
           \(weight_col) across(c(wtdn, c00100, iitax),
                                \(value_col) sum(weight_col * value_col, na.rm = TRUE)), 
           .names = "{.col}_{.fn}"), 
    .by = data_source)


df2 <- df |> 
  summarise(across(c(s006, ak00:wy00),
                   \(x) sum(x * c00100)),
            .by=data_source)

a <- proc.time()
df3 <- df |> 
  rename(us=s006) |> 
  pivot_longer(cols=c(us, ak00:wy00), names_to = "statecd", values_to = "weight")
glimpse(df3)

sums <- df3 |> 
  summarise(across(c(wtdn, c00100, e00200, e00300, iitax), 
                   \(x) sum(x * weight)),
            .by=c(data_source, statecd)) # a little slow
b <- proc.time()
b - a # 17.5 secs

sums |> filter(statecd=="ny11")
sums |> filter(statecd=="ca16")

avgs <- sums |> 
  mutate(across(c(c00100:iitax), \(x) x / wtdn)) |> 
  filter(data_source==1) |> 
  arrange(desc(c00100))

summary(avgs)  
head(avgs)
tail(avgs)

mhhi <- read_csv(here::here("raw_data", "mhhi2022.csv"))

comp <- avgs |> 
  left_join(mhhi, by = join_by(statecd))

comp |> 
  ggplot(aes(mhhi, c00100)) +
  geom_point(colour="blue") +
  geom_smooth()
  geom_abline(slope=1) +
  theme_bw()


comp |> 
  mutate(rmhhi=rank(mhhi), rc00100=rank(c00100)) |> 
  ggplot(aes(rmhhi, rc00100)) +
  geom_point(colour="blue") +
  geom_abline(slope=1) +
  theme_bw()


comp |> 
  mutate(rmhhi=rank(mhhi), rc00100=rank(c00100), riitax=rank(iitax)) |> 
  ggplot(aes(rmhhi, riitax)) +
  geom_point(colour="blue") +
  geom_abline(slope=1) +
  theme_bw()

comp |> 
  mutate(rmhhi=rank(mhhi), rc00100=rank(c00100), riitax=rank(iitax),
         rdiff=rc00100 - rmhhi,
         wagepct=e00200 / c00100) |> 
  filter(abs(rdiff) > 200)

comp |> 
  mutate(rmhhi=rank(mhhi), rc00100=rank(c00100), riitax=rank(iitax),
         rdiff=rc00100 - rmhhi,
         wagepct=e00200 / c00100) |> 
  ggplot(aes(wagepct, rdiff)) +
  geom_point()

comp |> 
  mutate(rmhhi=rank(mhhi), rc00100=rank(c00100), riitax=rank(iitax),
         rdiff=rc00100 - rmhhi,
         wagepct=e00200 / c00100) |> 
  ggplot(aes(wagepct, iitax)) +
  geom_point()



df3 |>
  arrange(desc(value))

ns(df)
dfxsums <- df |> 
  summarise(across(c(c00100, e00200, e00300, iitax),
                   \(y) across(c(ak00:wy00),
                               \(x) sum(x * y),
                               .names = "{.col}_{.fn}"),
                   .names = "{.col}"),
            .by = data_source)


dfxsums <- df |> 
  summarise(across(c(c00100, e00200, e00300, iitax),
                   \(y) across(c(ak00:wy00),
                               \(x) sum(x * .data[[cur_column()]]),
                               .names = "{.col}_{.fn}"),
                   .names = "{.col}"),
            .by = data_source)

dfxsums <- df |> 
  summarise(across(c(c00100, e00200, e00300, iitax),
                   \(y) across(c(ak00:wy00),
                               \(x) sum(x * y),
                               .names = "{.col}_{.fn}"),
                   .names = "{.col}"),
            .by = data_source)


library(dplyr)

dfxsums <- df %>% 
  group_by(data_source) %>%
  summarise(across(c(c00100, e00200, e00300, iitax), 
                   list(ak00 = ~sum(.x * ak00),
                        al00 = ~sum(.x * al00),
                        wy00 = ~sum(.x * wy00)),
                   .names = "{.col}_{.fn}"))



targetfiles <- dir_ls(here::here("temp_data")) |> str_subset("targets.csv")


# Assuming the column you need is called "needed_column"
# Assuming your base file is read as follows
base_file <- read_csv("path/to/base_file.csv")

# Read each CSV file and select only the required column
all_columns <- map(csv_files, ~ read_csv(.x, col_select = c("needed_column")))

# Combine all the columns
# This assumes that each column has the same order as the base file
combined_data <- bind_cols(base_file, all_columns)

# View the combined data
print(combined_data)

# c00100, e00200, e00300, iitax
dfxsums <- df |> 
  summarise(across(c(s006, ak00:wy00),
                   \(x) sum(x * c00100)),
            .by=data_source)


dfxsums[, 1:5]

# .names = "{.col}_{.fn}")

variable_names <- c("c00100", "e00200", "e00300", "iitax")

# Summarise across the columns of interest, applying each calculation
dfxsums <- df %>%
  summarise(
    across(
      c(s006, ak00:wy00),
      list = setNames(
        lapply(variable_names, function(var) {
          function(x) sum(x * df[[var]])
        }),
        variable_names
      ),
      .names = "{.fn}_{.col}"
    ),
    .by = data_source
  )

dfxsums

