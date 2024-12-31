# Create one local web site with examination results for each area type (state,
# cd, or both) passed to the function render_area_reports(). The most efficient
# and effective way to do this appears to be to create a single site (in the
# _site folder) and then copy it to an appropriately named folder.

# Optionally deploy the web site to netlify.

# To run, source this file and then run render_area_reports() from the console.

library(quarto)
library(fs)
library(stringr)
library(tidyverse)

generate_quarto_yaml <- function(book_title) {
  template <- readLines("_quarto_template.yml")
  rendered <- stringr::str_replace(template, coll("{{book_title}}"), book_title)
  writeLines(rendered, "_quarto.yml")
}

render_area_reports <- function(area_types = c("cd", "state"), eval_data = TRUE, deploy = FALSE) {
  
  for(area_type in area_types) {
    print(paste0("Area type: ", area_type))
    
    output_dir <- paste0("_", area_type)
    fs::dir_create(output_dir, recurse = TRUE)
    
    if(area_type == "state") {suffix <- "states"} else
      if(area_type == "cd") {suffix <- "Congressional Districts"}
    
    book_title <- paste0("Examination report for ", suffix)
    generate_quarto_yaml(book_title)
    
    # Render entire project with parameters
    quarto::quarto_render(
      as_job = FALSE,  # Ensures synchronous execution
      execute_params = list(
        area_type = area_type,
        eval_data = eval_data
      )
    )
    
    # Move all generated files
    fs::dir_copy("_site", output_dir, overwrite = TRUE)
    fs::dir_delete("_site")
    
    # Conditionally deploy to Netlify
    if(deploy){
      siteid <- case_when(area_type=="state" ~ "4842eca7-3a3b-4183-8b73-5635ad95101d",
                          area_type == "cd" ~ "573ad544-144b-4535-88cb-f2c41792fe84",
                          .default = "ERROR")
      system2("netlify",
              args = c("deploy",
                       "--prod",
                       paste0("--dir=", output_dir),
                       paste0(" --site=", siteid)))
    }
  }
}

