# Area Weighting: Lessons Learned

Practical guidance from developing the state weight optimization
pipeline. Intended for future maintainers and anyone extending
this to new geographies (e.g., Congressional districts).

## The optimization problem

For each sub-national area, we find weight multipliers `x_i` for
every record such that weighted sums match area targets within a
tolerance band, while keeping multipliers close to 1.0
(population-proportional).

```
minimize    sum((x_i - 1)^2)          [stay close to proportional]
subject to  target*(1-tol) <= Bx <= target*(1+tol)  [hit targets]
            0 <= x_i <= multiplier_max               [bounds]
```

Solved independently per area using Clarabel (constrained QP with
elastic slack for infeasibility).

## Key parameters and what they do

| Parameter | Default | Effect |
|-----------|---------|--------|
| `AREA_CONSTRAINT_TOL` | 0.005 (0.5%) | Target tolerance band. Matches national reweighting. |
| `AREA_MULTIPLIER_MAX` | 25.0 | Per-record upper bound on weight multiplier. Most important lever for controlling exhaustion. |
| `AREA_SLACK_PENALTY` | 1e6 | Penalty on constraint slack. Very high = hard constraints. |
| `weight_penalty` | 1.0 | Penalty on `(x-1)^2`. Higher values keep multipliers closer to 1.0. |

## Weight exhaustion

**Definition**: For each record, exhaustion = (sum of all area
weights) / (national weight). A value of 1.0 means the record's
national weight is fully allocated across areas. Values above 1.0
mean the record is "oversubscribed" — used more in total than its
national weight warrants.

**Why it matters**: High exhaustion means a small number of records
are doing heavy lifting across many states, creating fragile
solutions where one record's characteristics drive multiple states'
estimates.

**What drives it**: Rare high-income PUF records with small national
weights (s006 ~ 10-50) get pulled by many states to hit their
high-AGI-bin targets. These are typically wealthy MFJ households
with large investment income (interest, dividends, capital gains,
partnership income).

### Exhaustion statistics (2022 SOI, 178 targets/state)

| Percentile | Exhaustion |
|------------|-----------|
| Median | 1.007 |
| p99 | 2.0 |
| p99.9 | 4.4 |
| Max | 25.2 (at mult_max=100) |
| Max | 16.6 (at mult_max=25) |

Only ~150 records exceed 5x. The problem is concentrated in
the extreme tail.

## Parameter sweep results (2023 tax year, 51 states)

Tested `multiplier_max` x `weight_penalty` grid. Key finding:
**`weight_penalty` has no effect** on exhaustion, wRMSE, or %zero
— it only increases target violations. The solver reaches the
same weight structure regardless; higher penalty just makes it
fail to meet more targets.

**`multiplier_max` is the only effective lever:**

| mult_max | Violations | MaxViol% | wRMSE | %zero | MaxExh | >10x |
|----------|-----------|---------|-------|-------|--------|------|
| 100 | 33 | 0.50% | 0.594 | 7.3% | 25.2 | 16 |
| **25** | **35** | **0.50%** | **0.609** | **7.8%** | **16.6** | **15** |
| 15 | 210 | 100% | 0.601 | 9.9% | 11.9 | 3 |
| 10 | 402 | 100% | 0.625 | 12.2% | 8.8 | 0 |

**`mult_max=25` is the sweet spot**: only 2 extra violations
vs baseline, max violation still at the 0.50% tolerance boundary,
but max exhaustion drops 34% (25.2 to 16.6). Below 25, targets
become infeasible (100% violations appear).

## Two-pass exhaustion limiting (not recommended for production)

We tested an iterative approach: solve unconstrained, compute
exhaustion, set per-record caps, re-solve. With `max_exhaustion=5`:

- Pass 1: 33 violations (normal)
- Pass 2: **8,979 violations**, max **100%** — catastrophic

The proportional cap scaling was too aggressive. Records at 25x
got scaled to 20% of their multiplier, making targets infeasible.
The approach is fragile and hard to tune.

**Recommendation**: Use `multiplier_max` (single pass) rather
than iterative exhaustion capping. Simpler, more robust, and
no slower.

## Dual variable analysis (constraint cost)

Clarabel's dual variables reveal which constraints are expensive
to satisfy — a target can be perfectly hit but still cause massive
weight distortion. This is invisible from violations alone.

**Finding**: The $1M+ AGI bin filing-status count targets (single,
MFJ, HoH) are essentially the **only expensive constraints**. Their
dual costs are 6-8 orders of magnitude larger than all other targets.
Extended targets (capital gains, pensions, charitable, etc.) have
near-zero dual cost — they're "free" to add.

**Action taken**: Excluded filing-status counts from the $1M+ bin.
Kept total return count as an anchor. This eliminated virtually all
constraint cost while maintaining full target coverage.

## SALT targeting

Use **actual SALT** (`c18300`, the post-$10K-cap amount) rather than
uncapped potential SALT (`e18400`/`e18500`). This is apples-to-apples:
Tax-Calculator's actual SALT under current law, shared using observed
SOI actual SALT by state.

| Metric | e18400/e18500 | c18300 |
|--------|--------------|--------|
| Violations | 85 | 75 |
| wRMSE avg | 0.479 | 0.289 |
| SALT aggregation error | -6.23% | -0.18% |

## OA (Other Areas) share rescaling

SOI's "Other Areas" category (~0.5% of returns, covers territories
and overseas filers) is excluded from state targeting. Raw SOI
shares (state/US) are rescaled so that the 51 state shares sum
to 1.0 for each variable-AGI-bin combination.

## Guidance for Congressional Districts

CDs have not been implemented yet. Expected differences from states:

- **435 areas** (vs 51) — grid search infeasible; use state-derived
  parameter settings
- **9 AGI bins** (vs 10) — no $1M+ separate bin in SOI CD data
- **Smaller populations** — more CDs will hit multiplier ceilings;
  may need different `multiplier_max`
- **Crosswalk complexity** — SOI uses 117th Congress boundaries for
  both 2021 and 2022 data; need geocorr crosswalk to map to 118th
  Congress boundaries
- **Exhaustion will be worse** — 435 areas competing for the same
  records means much higher potential exhaustion; `multiplier_max`
  may need to be lower

## Running the parameter sweep

The sweep utility tests combinations of solver parameters on all
states and reports target accuracy, weight distortion, and exhaustion:

```bash
python -m tmd.areas.sweep_params
```

Edit `GRID_MULTIPLIER_MAX` and `GRID_WEIGHT_PENALTY` in
`tmd/areas/sweep_params.py` to change the search grid. Each
combination solves all 51 states (~3-4 minutes with 8 workers).

**Should be rerun when**:
- The tax year changes (different TMD data, different SOI shares)
- The target recipe changes materially
- Extending to a new area type (CDs)
