# National targets — reference documentation

This folder holds documentation for the national targets pipeline.

## Files tracked in this folder

- `adding_a_new_year.md` — how to extend the pipeline when IRS SOI data for a
  new tax year becomes available.
- `README.md` — this file.

## SOI reference documents (not committed; download as needed)

The two IRS publications below are useful when verifying what SOI variables
mean — in particular, when aligning a SOI aggregate with a TaxCalc variable
(e.g., confirming that SOI `tottax` = income tax after credits + NIIT, or
checking which 1040 line each SOI variable aggregates).

They are not tracked in git because each is large (6–15 MB) and always
available from the IRS. Drop the PDFs into this folder (filenames below are
already in `.gitignore`) if you want local copies.

### Pub 1304 — Individual Income Tax Returns, Complete Report

Published annually by the IRS Statistics of Income Division. The tax-year
2022 edition (released October 2024) is the current reference used when
building and validating TMD's 2022 targets.

- **Hub page (all years):**
  <https://www.irs.gov/statistics/soi-tax-stats-individual-income-tax-returns-complete-report-publication-1304>
- **Latest release (always current):** <https://www.irs.gov/pub/irs-pdf/p1304.pdf>
- **Tax-year 2022 edition (direct link, as of 2026-04-24):**
  <https://www.irs.gov/pub/irs-prior/p1304--2024.pdf>
  (IRS `irs-prior` URLs sometimes shift as new editions release — start from
  the hub page if a direct link 404s.)
- Local filename used by convention in this folder: `p1304_2022.pdf` (~15 MB)

### Pub 1304 — Tax Year 2015 edition

Tax year 2015 is the vintage of the PUF that TMD started from, before
uprating and imputation. The TY2015 SOI report is the authoritative
documentation for the underlying data concepts at the starting point.

- **Direct link:** <https://www.irs.gov/pub/irs-soi/15inalcr.pdf>
- Local filename used by convention in this folder: `15inalcr_2021.pdf`
  (~6 MB; the `_2021` in the filename reflects the filename someone saved
  it under, not an IRS publication year — the PDF is the TY2015 report.)
