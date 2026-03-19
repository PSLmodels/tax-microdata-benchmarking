# Target Recipes

Target recipes define which constraints the area weight optimizer must
satisfy. Each recipe is a JSON file paired with a variable mapping CSV.

## Files

- `states.json` — State recipe (2022 SOI data). See inline comments
  for keyword documentation.
- `state_variable_mapping.csv` — Maps recipe variable names to
  internal column names in the enhanced targets DataFrame.

## State Target Summary (~178 per state)

### Base targets (from recipe, all 10 AGI bins unless noted)

| Variable | Measure | Filing Status | AGI Bins | Targets | Description |
|----------|---------|---------------|----------|---------|-------------|
| XTOT | population | all | all | 1 | Census state population |
| c00100 | amount | all | 1–10 | 10 | Adjusted gross income |
| c00100 | return count | all | 1–10 | 10 | Total returns |
| c00100 | return count | single | 1–9 | 9 | Single filer returns |
| c00100 | return count | MFJ | 1–9 | 9 | Married filing jointly returns |
| c00100 | return count | HoH | 1–9 | 9 | Head of household returns |
| e00200 | amount | all | 1–10 | 10 | Wages and salaries |
| e00200 | nz-count | all | 1–9 | 9 | Wage earner count |
| e00300 | amount | all | 1–10 | 10 | Taxable interest income |
| e01500 | amount | all | 1–10 | 10 | Pensions (shared by taxable) |
| e02400 | amount | all | 1–10 | 10 | Social Security (shared by taxable) |
| c18300 | amount | all | 3–10 | 8 | SALT deduction (after $10K cap) |
| e26270 | amount | all | 1–10 | 10 | Partnership/S-corp income |

Filing-status counts and wage nz-counts are excluded from the $1M+
bin (stub 10) because dual variable analysis showed these small-cell
count targets dominate weight distortion. SALT skips the two lowest
bins where itemization is rare.

### Extended targets (stubs 5–10 only, $50K+)

| Variable | Source | Targets | Description |
|----------|--------|---------|-------------|
| e01700 | SOI a01700 | 6 | Taxable pensions |
| c02500 | SOI a02500 | 6 | Taxable Social Security |
| e01400 | SOI a01400 | 6 | Taxable IRA distributions |
| capgains_net | SOI a01000 | 6 | Net capital gains |
| e00600 | SOI a00600 | 6 | Ordinary dividends |
| e00900 | SOI a00900 | 6 | Business/professional income |
| c19200 | SOI a19300 | 6 | Mortgage interest deduction |
| c19700 | SOI a19700 | 6 | Charitable contributions |
| e18400 | Census S&L taxes | 6 | SALT income/sales (available) |
| e18500 | Census property tax | 6 | SALT real estate (available) |
| eitc | SOI a59660 | 2 | EITC (amount + count, aggregate) |
| ctc_total | SOI a07225+a11070 | 2 | Child Tax Credit (amount + count, aggregate) |

Extended targets use SOI or Census geographic shares applied to TMD
national totals. They are restricted to high-income bins ($50K+)
to avoid noisy low-income data. Credit targets are aggregate
(one per state, no AGI breakdown).

### AGI bins (states)

| Stub | Range |
|------|-------|
| 1 | Under $1,000 |
| 2 | $1,000 – $10,000 |
| 3 | $10,000 – $25,000 |
| 4 | $25,000 – $50,000 |
| 5 | $50,000 – $75,000 |
| 6 | $75,000 – $100,000 |
| 7 | $100,000 – $200,000 |
| 8 | $200,000 – $500,000 |
| 9 | $500,000 – $1,000,000 |
| 10 | $1,000,000 or more |

## Variable Mapping CSV

The mapping CSV connects the `varname` in the recipe to the
`basesoivname` column in the enhanced targets DataFrame produced
by `target_sharing.py`.

Columns:

| Column | Description |
|--------|-------------|
| varname | TMD variable name used in the recipe JSON |
| basesoivname | Internal name in the enhanced targets DataFrame |
| description | Human-readable description |
| fstatus | Filing status (0=all, 1=single, 2=MFJ, 4=HoH) |

### Naming conventions for basesoivname

- **Direct SOI names**: `00100`, `00200`, `n1`, `mars1`, etc.
  Used when the SOI variable directly matches the TMD variable.
- **Shared names**: `tmd01500_shared_by_soi01700` means TMD variable
  e01500 (total pensions) uses SOI variable 01700 (taxable pensions)
  for geographic distribution. The TMD national total is shared
  across states using the SOI geographic proportions.
- **XTOT**: Population (Census data, not SOI).

### Adding a new variable

1. Add the variable to the recipe JSON with appropriate scope, count,
   and fstatus values.
2. Add a row to the mapping CSV connecting the varname to the correct
   basesoivname.
3. If the variable uses SOI sharing, ensure the SOI base variable is
   in `ALL_SHARING_MAPPINGS` in `constants.py`.
