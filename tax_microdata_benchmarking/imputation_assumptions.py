"""
Central location for data imputation assumptions.
"""

IMPUTATION_RF_RNG_SEED = 1928374  # random number seed used by RandomForest

IMPUTATION_BETA_RNG_SEED = 37465  # random number seed used for Beta variates

W2_WAGES_SCALE = 0.16  # parameter used to impute pass-through W-2 wages

REWEIGHT_MULTIPLIER_MIN = 0.1
REWEIGHT_MULTIPLIER_MAX = 10.0
REWEIGHT_DEVIATION_PENALTY = 0.2
# penalty value of 1.0 says "this is as important as everything else"
# penalty value of 0.0 imposes no penalty
