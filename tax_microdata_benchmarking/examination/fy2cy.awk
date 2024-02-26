# Convert US federal fiscal year (FY) amount to calendar year (CY) amount
# using linear interpolation and linear extrapolation to estimate the
# adjustments for the one quarter difference between the CY and the FY.
#
# USAGE:  awk -f fy2cy.awk INFILE
#
#   where each non-blank line of the CSV-formatted INFILE contains four fields:
#   (1) FY1 amount (MUST be empty on a metadata line)
#   (2) FY2 amount (MUST be empty on a metadata line)
#   (3) CY1 label (MUST be empty on a metadata line)
#   (4) metadata (MUST be empty on lines that have non-empty fields 1-3)
#   The returned amount is an estimate of the CY1 amount corresponding
#   to the FY1 and FY2 amounts plus the third field containing a label.

BEGIN {
    # expects CSV-formatted INFILE
    FS = ","
    # middle of FY2 Q1 is 0.625 years after the middle of FY1
    frac = 0.625
}

NR>1 && NF==4 && $1!=0 && $2!=0 && $4=="" {
    fy1 = $1
    growth = $2 - $1
    fy2q1 = fy1 + frac*growth      # annual amount in first quarter of FY2
    fy1q1 = fy1 - (1-frac)*growth  # annual amount in first quarter of FY1
    cy1 = fy1 + 0.25*(fy2q1-fy1q1) # adjust FY1 using first quarter amounts
    printf "%6.1f   %s\n", cy1, $3
}
