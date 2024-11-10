
source(here::here("R", "libraries.R"))
source(here::here("R", "constants.R"))
source(here::here("R", "functions.R"))

phase4cds <- c("AK00", "DE00", "ID01", "ID02", "ME02", "MT00", "ND00", "PA08", "SD00", "WY00")


# constants in prepare ----------------------------------------------------

CDZIPURL <- "https://www.irs.gov/pub/irs-soi/congressional2021.zip"
CDDOCURL <- "https://www.irs.gov/pub/irs-soi/21incddocguide.docx"


# dir_ls(here::here("..", "..", "targets", "prepare"))
CDDIR <- here::here("..", "..", "targets", "prepare", "cds")
CDRAW <- fs::path(CDDIR, "raw_data")
CDINTERMEDIATE <- fs::path(CDDIR, "intermediate")
CDFINAL <- fs::path(CDDIR, "final")

CDDOCEXTRACT <- "cd_documentation_extracted_from_21incddocguide.docx.xlsx"

HERE <- here::here()
CDTARGETSDIR <- fs::path(HERE, "..", "..", "targets", "prepare", "cds", "intermediate")

# dir_ls(here::here("..", "..", "weights"))
WTSDIR <- here::here("..", "..", "weights")
files <- dir_ls(WTSDIR)

CHISQ <- fs::path(path_home(), "Documents/python_projects", "scratch", "chisquare_test")
TARGDIR <- here::here("..", "..", "targets")
dir_ls(CHISQ)
# \\wsl.localhost\Ubuntu\home\donboyd5\Documents\python_projects\scratch\chisquare_test

# Martin's approach to weights ----
# get data ----------------------------------------------------------------
# targets ----
targets <- read_csv(fs::path(CDTARGETSDIR, "cdbasefile_sessions.csv"))
glimpse(targets)

targ2 <- targets |> 
  filter(statecd %in% c("ID02", "PA08")) |> 
  filter(session==117, basevname %in% c("v00100", "v00200", "v26270")) |> 
  select(statecd, basevname, vname, agistub, agilo, agihi, scope, fstatus, count, target)

count(targ2, basevname, vname)

targ64 <- bind_rows(
  read_csv(fs::path(CHISQ, "id02A_targets.csv")) |> mutate(cd="ID02"),
  read_csv(fs::path(CHISQ, "pa08A_targets.csv")) |> mutate(cd="PA08"))


# tmd2021 ----
TMDDIR <- here::here("..", "..", "..", "storage", "output")
fpath <-  fs::path(TMDDIR, "tmd2021_cache.csv")
tmd2021 <- vroom(fpath)
ns(tmd2021)

# weights ----

weightfiles <- c(str_subset(files, "id02"), str_subset(files, "pa08"))
f <- function(fpath){
  cd <- str_sub(fs::path_file(fpath), 1, 5)
  print(cd)
  vroom(fpath, col_select = "WT2021") |> 
    rename(!!cd := WT2021) # create a column named for the cd, with its 2021 as its value
}

weights <- test <- purrr::map(weightfiles, f) |> 
  list_cbind() |> 
  mutate(row=row_number()) |> 
  relocate(row)
glimpse(weights)


# combine weights and data ------------------------------------------------

icuts <- c(-Inf, 1, 10e3, 25e3, 50e3, 75e3, 100e3, 200e3, 500e3, Inf)
combo <- tmd2021 |> 
  select(RECID, data_source, c00100, e00200, e26270, iitax, s006) |> 
  mutate(row=row_number(),
         irange=cut(c00100, icuts, right = FALSE, ordered_result = TRUE),
         irange = factor(irange, 
                         levels = levels(irange), # Ensure ordering is maintained
                         labels = str_replace(levels(irange), ",", ", ")), # make more-readable labels
                # add a total to the factor because down the road we will have totals ??
                irange = fct_expand(irange, "total"),
                irange = fct_relevel(irange, "total"),
         agistub=as.integer(irange) - 1) |> 
  left_join(weights, by = join_by(row))
count(combo, agistub, irange)

groupsums <- combo |> 
  filter(data_source==1) |> 
  summarise(across(id02A:pa08B,
                   list(c00100 = \(x) sum(x * c00100),
                        e00200 = \(x) sum(x * e00200),
                        e26270 = \(x) sum(x * e26270),
                        iitax = \(x) sum(x * iitax))),
            .by=c(agistub, irange)) |> 
  pivot_longer(-c(agistub, irange)) |> 
  separate(name, into=c("cd", "variable")) |> 
  pivot_wider(names_from = variable) |> 
  mutate(otherinc=c00100 - e00200 - e26270)

groupsums
cdsums <- groupsums |> 
  summarise(across(-c(agistub, irange), sum),
            .by=cd) |> 
  mutate(agistub=0, irange="total")

allsums <- bind_rows(groupsums, cdsums) |> 
  arrange(cd, agistub) |> 
  relocate(otherinc, .before=iitax)

allsums |> 
  filter(str_starts(cd, "id")) |> 
  select(agistub, irange, cd, value=c00100) |> 
  pivot_wider(names_from = cd) |> 
  mutate(diff=pick(4))


ftab <- function(ststart, var){
  tabdata <- allsums |> 
    filter(str_starts(cd, ststart)) |> 
    select(agistub, irange, cd, value=all_of(var)) |> 
    pivot_wider(names_from = cd) |> 
    arrange(agistub) |> 
    select(-agistub) |> 
    mutate(diff = unlist(pick(3) - pick(2)),
           pdiff= diff / unlist(pick(2)))
  
  tabdata |> 
    gt() |> 
    tab_header(title=paste0(var, ": totals and differences under two sets of targets"),
               subtitle="A=64 targets, B=37 targets. Amounts are in $ millions.") |> 
    cols_label(irange="AGI range") |> 
    fmt_number(columns=2:4,
               scale=1e-6,
               decimals=1) |> 
    fmt_percent(columns=pdiff,
                decimals = 1) |> 
    fmt_currency(columns=2:4,
                 rows = 1,
                 scale=1e-6,
                 decimals=1) |> 
    tab_footnote(
      footnote = "B targets exclude 9 each of: returns count for all-marital-statuses, wage amount, and partnership and S corporation income amount"
    )
  }
ftab("id", "c00100")
ftab("id", "iitax")
ftab("id", "e00200")
ftab("id", "e26270")

ftab("pa", "c00100")
ftab("pa", "iitax")
ftab("pa", "e00200")
ftab("pa", "e26270")


allsums |> 
  mutate(wageshare=e00200 / c00100) |> 
  select(agistub, irange, cd, wageshare) |> 
  pivot_wider(names_from = cd, values_from = wageshare) |> 
  select(irange, id02A, pa08A, id02B, pa08B) |> 
  mutate(pa_idA=pa08A - id02A, pa_idB =pa08B - id02B) |> 
  gt() |> 
  tab_header(title=paste0("Wage share of AGI under two sets of targets"),
             subtitle="A=64 targets, B=37 targets. Amounts are in $ millions.") |> 
  cols_label(irange="AGI range") |> 
  fmt_percent(columns=-c(irange),
              decimals = 0) |> 
  tab_footnote(
    footnote = "B targets exclude 9 each of: returns count for all-marital-statuses, wage amount, and partnership and S corporation income amount"
  ) 



# vars <- c("c00100", "e00200", "e26270", "iitax")


# wage analysis -----------------------------------------------------------
wageshare <- combo |> 
  summarise(across(id02A:pa08B,
                   list(c00100 = \(x) sum(x * c00100),
                        e00200 = \(x) sum(x * e00200))),
            .by=c(irange, data_source)) |> 
  arrange(data_source, irange) |> 
  pivot_longer(-c(irange, data_source)) |> 
  separate(name, into=c("cd", "variable")) |> 
  pivot_wider(names_from = variable) |> 
  mutate(wshare=e00200 / c00100)

wageshare

wageshare |> 
  select(data_source, irange, cd, wshare) |> 
  pivot_wider(names_from = cd, values_from = wshare) |> 
  filter(data_source==1) |> 
  mutate(iddiff=id02B - id02A,
         padiff=pa08B - pa08A,
         pamidA=pa08A - id02A,
         pamidB=pa08B - id02B)
  

# explore targets

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


