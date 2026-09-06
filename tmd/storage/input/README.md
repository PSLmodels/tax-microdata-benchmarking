# Input files

Files read by the `tmd/create_*.py` scripts during `make data`.

Two of these are themselves produced by pre-processing pipelines that
run *before* `make data`, and are committed here as vetted artifacts:

- `soi.csv` — see [tmd/national_targets](../../national_targets/README.md)
- `taxdata<VINTAGE>_growfactors.csv` — see
  [tmd/growfactors](../../growfactors/README.md)

The growth factors files are frozen once a `TAXYEAR` has been built with
them, so that past builds stay reproducible; a refresh of the growth
factors source data produces a new `VINTAGE` rather than rewriting an
existing file.
