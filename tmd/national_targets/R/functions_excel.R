# excel helpers for reading the target recipes xlsx file and individual xls* files ----
xlcol_to_num <- function(s) {
  # convert vector of Excel column labels s such as c("BF", "AG") to column numbers
  sapply(strsplit(toupper(s), "", fixed = TRUE), \(chars) {
    Reduce(\(a, b) a * 26 + b, match(chars, LETTERS))
  })
}

xlnum_to_col <- function(n) {
  # convert vector of Excel column numbers s such as c(58, 33) to column labels
  unname(sapply(n, \(x) {
    out <- character()
    while (x > 0) {
      x <- x - 1L
      out <- c(LETTERS[x %% 26 + 1L], out)
      x <- x %/% 26
    }
    paste0(out, collapse = "")
  }))
}

xlcols <- function(n) {
  # create a vector of letters in the order that Excel uses

  # a helper function that allows us to put letter column names on a dataframe
  #   that was read from an Excel file

  # usage:
  #   xlcols(53)
  #   gets the letters for the first 53 columns in a spreadsheet
  # only good for 1- and 2-letter columns, or 26 + 26 x 26 = 702 columns
  xl_letters <- c(
    LETTERS,
    sapply(LETTERS, function(x) paste0(x, LETTERS, sep = ""))
  )
  return(xl_letters[1:n])
}


get_rowmap <- function(tab, DATADIR, targfn) {
  # reads the target recipes xlsx file to
  # get start and end row for key data for each year of a particular IRS spreadsheet
  # from its associated mapping tab in the recipes file
  # targfn is the targets filename
  sheet <- paste0(tab, "_map")
  readxl::read_excel(
    fs::path(DATADIR, targfn),
    sheet = sheet,
    range = cellranger::cell_rows(1:3)
  ) |>
    tidyr::pivot_longer(-rowtype, values_to = "xlrownum") |>
    tidyr::separate_wider_delim(
      name,
      delim = "_",
      names = c("datatype", "year")
    ) |>
    dplyr::mutate(
      table = tab,
      year = as.integer(year),
      xlrownum = as.integer(xlrownum)
    ) |>
    dplyr::select(table, datatype, year, rowtype, xlrownum) |>
    dplyr::arrange(table, year, datatype, desc(rowtype))
}


get_colmap <- function(tab, DATADIR, targfn) {
  # reads the target_recipes.xlsx file to
  # get columns of interest for each year of a particular IRS spreadsheet,
  # from its associated mapping tab in the recipes file

  # assumes DATADIR, targfn (targets filename), and allcols are in the environment
  sheet <- paste0(tab, "_map")
  col_map <- readxl::read_excel(
    path(DATADIR, targfn),
    sheet = sheet,
    skip = 3
  ) |>
    pivot_longer(
      -c(vname, description, units, notes),
      values_to = "xlcolumn"
    ) |>
    separate_wider_delim(name, delim = "_", names = c("datatype", "year")) |>
    mutate(
      table = tab,
      year = as.integer(year),
      # xl_colnumber = match(xlcolumn, allcols)
      xl_colnumber = xlcol_to_num(xlcolumn)
    ) |>
    select(
      table,
      datatype,
      year,
      xl_colnumber,
      xlcolumn,
      vname,
      description,
      units,
      notes
    ) |>
    filter(!is.na(xlcolumn), !is.na(vname)) |>
    arrange(table, datatype, year, xl_colnumber)
  col_map
}

# allcols <- xlcols(400); get_colmap("tab11")
