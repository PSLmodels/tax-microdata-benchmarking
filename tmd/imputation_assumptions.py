"""
Central location for assumptions about data generation and data imputation.
"""

TAXYEAR = 2022  # single source of truth for the target tax year

IMPUTATION_RF_RNG_SEED = 1928374  # random number seed used by RandomForest
IMPUTATION_BETA_RNG_SEED = 37465  # random number seed used for Beta variates

W2_WAGES_RATIO = {  # parameter used to impute pass-through W-2 wages
    2021: 0.150,
    2022: 0.155,
}

# RNG seeds used for demographic decoding and earnings splits in puf.py:
FILER_AGE_HEAD_RNG_SEED = 64963751
FILER_AGE_SPOUSE_RNG_SEED = 64963753
DEP_AGE_RNG_SEED = 24354657
EARN_SPLIT_RNG_SEED = 18374659

# annual growth rates for SALT and all other itemized deductions in PUF dataset
SALT_GROW_RATE = 0.047  # for SALT (E18400, E18500) itemized deduction amounts
ITMDED_GROW_RATE = 0.02  # for non-SALT itemized deduction amounts
# The SALT_GROW_RATE is based on growth in state and local property,
# general sales, and individual income taxes, from 2015 to 2022, per the
# Census Bureau Annual Survey of State and Local Government Finances.

# parameters used to impute CPS variables:
CPS_TAXABLE_INTEREST_FRACTION = 0.680  # from SOI 2020 data
CPS_QUALIFIED_DIVIDEND_FRACTION = 0.448  # from SOI 2018 data
CPS_TAXABLE_PENSION_FRACTION = 1.0  # no source, so arbitrary assumption
CPS_LONG_TERM_CAPGAIN_FRACTION = 0.880  # from SOI 2012 data

# parameters used to identify CPS nonfilers:
FILER_MIN_INCOME = {
    2021: 8600,
    2022: 9300,
}
EITC_CLAIM_THD = {
    2021: 1800,  # reduces 2023 EITC from $82.3b to $71.6b, a claim rate of 87%
    2022: 1600,  # reduces 2023 EITC from $80.5b to $72.0b, a claim rate of 89%
}
ACTC_CLAIM_THD = {
    2021: 0,  # always leave 2021 value at zero
    2022: 1500,
}
CPS_FILER_MIN_INCOME = FILER_MIN_INCOME[TAXYEAR]
CREDIT_CLAIMING = {
    "eitc_claim_thd": {f"{TAXYEAR}": EITC_CLAIM_THD[TAXYEAR]},
    "actc_claim_thd": {f"{TAXYEAR}": ACTC_CLAIM_THD[TAXYEAR]},
}
CPS_WEIGHTS_SCALE = {2021: 1.0, 2022: 1.0}  # for scaling CPS nonfiler weights

# parameters used in creation of national sampling weights:
REWEIGHT_MULTIPLIER_MIN = 0.1
REWEIGHT_MULTIPLIER_MAX = 10.0
# parameters for constrained-quadratic-programming reweighting:
CLARABEL_CONSTRAINT_TOL = 0.005  # relative tolerance on constraints (+-0.5%)
CLARABEL_SLACK_PENALTY = 1e6  # elastic penalty for constraint violations
CLARABEL_MAX_ITER = 1000  # maximum solver iterations

# parameters for MICE imputation of missing OBBBA deduction variables:
# ... overtime_income:
OTM_convert_zero_prob = {2021: 0.077, 2022: 0.078}
OTM_scale = {2021: 2.4, 2022: 2.4}
# ... tip_income:
TIP_convert_zero_prob = {2021: 0.014, 2022: 0.014}
TIP_scale = {2021: 1.0, 2022: 0.96}
# ... auto_loan_interest:
ALI_convert_zero_prob = {2021: 0.060, 2022: 0.060}
ALI_scale = {2021: 4.0, 2022: 4.0}

# population projection file used to extrapolate TAXYEAR sampling weights
POP_FILE = {
    2021: "cbo25_population.yaml",
    2022: "cbo26_population.yaml",
}
POPULATION_FILE = POP_FILE[TAXYEAR]
