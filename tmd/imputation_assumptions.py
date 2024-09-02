"""
Central location for data imputation assumptions.
"""

IMPUTATION_RF_RNG_SEED = 1928374  # random number seed used by RandomForest

IMPUTATION_BETA_RNG_SEED = 37465  # random number seed used for Beta variates

W2_WAGES_SCALE = 0.15  # parameter used to impute pass-through W-2 wages

ITMDED_GROW_RATE = 0.02  # annual growth rate in itemized deduction amounts
# grow rate applied to inflate 2015 amounts to 2021 amounts in uprate_puf.py

CPS_WEIGHTS_SCALE = 0.5806  # used to scale CPS-subsample population

# parameters used in creation of national sampling weights:
REWEIGHT_MULTIPLIER_MIN = 0.1
REWEIGHT_MULTIPLIER_MAX = 10.0
REWEIGHT_DEVIATION_PENALTY = 0.0
# penalty value of 1.0 says "this is as important as everything else"
# penalty value of 0.0 imposes no penalty

# AGI bins used in weight creation for subnational areas:
SOI_ZIP_AGI_BINS = [
    -9e99,
    1.0,
    25e3,
    50e3,
    75e3,
    100e3,
    200e3,
    9e99,
]
