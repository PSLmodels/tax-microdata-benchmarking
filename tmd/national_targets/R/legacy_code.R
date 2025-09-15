# Legacy code to download IRS files

f <- function(upath, destfolder) {
  print(upath)
  download.file(
    url = upath,
    destfile = fs::path(destfolder, fs::path_file(upath)),
    mode = "wb"
  )
}

# uncomment the next 3 lines to download files
# urls <- xxxx  # for example, tabmeta$upath
# destfolder <- zzzz # the destination folder
# walk(urls, \(upath) f(upath, destfolder)) # walk through the list of paths, downloading and saving each file

# legacy code to define locations of files, superseded by project-dir option in _quarto.yml

# if (Sys.getenv("QUARTO_PROJECT_DIR") != "") {
#   # When rendering via quarto render
#   QDIR <- Sys.getenv("QUARTO_PROJECT_DIR")
# } else {
#   # When running interactively
#   QDIR <- here::here("tmd", "national_targets")
# }
