

ht <- function(df, nrecs = 6) {
  print(utils::head(df, nrecs))
  print(utils::tail(df, nrecs))
}

ns <- function(obj){
  sort(names(obj))
}

