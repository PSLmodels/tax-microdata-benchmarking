# Area preparation — raw input data

This directory contains raw input files used by `tmd.areas.prepare_shares`
and `tmd.areas.prepare_targets` to build per-state and per-CD targets.

## Congressional District crosswalks

The SOI Congressional District (CD) micro-data are published on
**117th Congress** boundaries for both tax years 2021 and 2022.  To
produce targets on later Congressional session boundaries we apply a
Geocorr 2022 population-weighted crosswalk.

| File | Direction | Used by |
| --- | --- | --- |
| `geocorr2022_cd117_to_cd118.csv` | 117th → 118th | `--congress 118` |
| `geocorr2022_cd117_to_cd119.csv` | 117th → 119th | `--congress 119` |

Both files share the same column schema (with `cd118` / `cd119`
swapped) and both include a descriptive label row immediately after
the header.  The loader in
[tmd/areas/prepare/soi_cd_data.py](../soi_cd_data.py) detects the
target column automatically and renames it to a neutral `cd_target`
column so the rest of the pipeline is congress-agnostic.

### Geocorr settings

Both files were generated at [mcdc.missouri.edu/geocorr2022.html](https://mcdc.missouri.edu/applications/geocorr2022.html)
with the following settings:

- **Source geography:** State × Congressional District {118, 119}
- **Target geography:** State × Congressional District 117
- **Weighting:** 2020 Decennial population (`pop20`)
- **Block-groups with zero population:** excluded
- **Generate second allocation factor:** checked — produces `afact2`,
  the inverse (`cd117 → cd{118,119}`) factor that the pipeline uses

Geocorr's primary `afact` column is therefore
`cd{118,119}-to-cd117`, while `afact2` is `cd117-to-cd{118,119}`.
Because the SOI CD data are published on 117th-Congress boundaries,
the pipeline multiplies SOI values by `afact2` to allocate 117th-
Congress CD data onto 118th- or 119th-Congress CDs.

### Validation

Run

```
python -m tmd.areas.prepare.validate_crosswalk --congress 118
python -m tmd.areas.prepare.validate_crosswalk --congress 119
```

or the equivalent pytest suite `tests/test_cd_crosswalk.py`, to check
that:

1. `afact2` per `(stabbr, cd117)` sums to 1.0 within CSV rounding
2. At-large states (MT, DE, WY, SD, ND, VT, AK) are recoded from
   `00` / `98` to `01`
3. Population is conserved between source and target
4. 436 distinct target CDs (excluding PR)
5. For 119 only: AL, GA, LA, NY, and NC differ from their 118
   factors; the other 47 state/jurisdictions are identical

## Other files in this directory

- `soi_cds/` — SOI Congressional District micro-data by tax year
- `soi_states/` — SOI state-level micro-data by tax year
- `state_populations.json` — Census 2022 state populations
- `census_2022_state_local_finance.xlsx` — Census SALT proxy data
