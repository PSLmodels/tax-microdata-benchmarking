"""
Central location for data imputation assumptions.
"""

TAXYEAR = 2021  # single source of truth for the target tax year

IMPUTATION_RF_RNG_SEED = 1928374  # random number seed used by RandomForest

IMPUTATION_BETA_RNG_SEED = 37465  # random number seed used for Beta variates

W2_WAGES_SCALE = 0.15  # parameter used to impute pass-through W-2 wages

# RNG seeds used for demographic decoding and earnings splits in puf.py:
FILER_AGE_HEAD_RNG_SEED = 64963751
FILER_AGE_SPOUSE_RNG_SEED = 64963753
DEP_AGE_RNG_SEED = 24354657
EARN_SPLIT_RNG_SEED = 18374659

ITMDED_GROW_RATE = 0.02  # annual growth rate in itemized deduction amounts
# grow rate applied to inflate 2015 amounts to 2021 amounts in uprate_puf.py

CPS_FILER_MIN_INCOME = 8600
EITC_CLAIM_THD = {2021: 1800, 2022: 0}
ACTC_CLAIM_THD = {2021: 0, 2022: 0}
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

# parameters for MICE imputation of missing OBBBA deduction variables
# ... overtime_income:
OTM_convert_zero_prob = {2021: 0.077, 2022: 0.0}
OTM_scale = {2021: 2.4, 2022: 1.0}
# ... tip_income:
TIP_convert_zero_prob = {2021: 0.014, 2022: 0.0}
TIP_scale = {2021: 1.0, 2022: 1.0}
# ... auto_loan_interest:
ALI_convert_zero_prob = {2021: 0.060, 2022: 0.0}
ALI_scale = {2021: 4.0, 2022: 1.0}
