
fname <- "soi_states_variable_documentation.xlsx"
fpath <- fs::path(DRAW, fname)

# df1 <- readxl::read_xlsx(fpath, sheet = "2021", col_types = "list")
df1 <- readxl::read_xlsx(fpath, sheet = "2021", col_types = "text")
df2 <- df1 |> 
  select(vname=1, description=2, reference=3, type=4) |> 
  filter(if_any(everything(), ~!is.na(.))) |> 
  # after verifying that AGI_STUB is the only variable with NA in vname
  # fill down and then concatenate the reference column
  fill(vname, description, type, .direction="down") |> 
  mutate(reference = paste(reference, collapse = "\n"), .by=vname) |> 
  distinct() |> 
  # for now, make mistaken references NA
  mutate(reference=ifelse(!is.na(as.numeric(reference)), NA_character_, reference),
         reference=ifelse(reference=="NA", NA_character_, reference))
    
df2
