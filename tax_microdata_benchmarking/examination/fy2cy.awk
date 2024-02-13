# Convert US federal fiscal year (FY) amount to calendar year (CY) amount
# using linear interpolation and linear extrapolation to estimate the
# adjustments for the one quarter difference between the CY and the FY.
#
# USAGE:  awk -f fy2cy.awk INFILE
#
#   where each line of the space-delimited INFILE contains two consecutive
#   FY amounts followed by at least one field containing metadata.
#   The returned amount is an estimate of the CY amount corresponding to
#   the first FY amount plus the third field containing metadata.
#   Note that all lines without three space-delimited fields are skipped
#   and can, therefore, be used for documentation.

BEGIN {
    # middle of subsequent FY Q1 is 0.625 years after the middle of prior FY
    frac = 0.625
}

NF==3 {
    fy1 = $1
    growth = $2 - $1
    fy2q1 = fy1 + frac*growth      # annual amount in first quarter of FY2
    fy1q1 = fy1 - (1-frac)*growth  # annual amount in first quarter of FY1
    cy1 = fy1 + 0.25*(fy2q1-fy1q1) # adjust FY1 using first quarter amounts
    printf "%6.1f   %s\n", cy1, $3
}
