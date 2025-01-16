# The examination/2022/generate_tmd_results.sh script.
# PREREQUISITE:
#   (taxcalc-dev) tax-microdata-benchmarking% make test
# USAGE:
#   (taxcalc-dev) 2022% ./generate_tmd_results.sh

# === SETUP ===
TMD=../..
cp $TMD/storage/output/tmd*csv* .
gunzip -f tmd.csv.gz
STATES="ak mn nj nm sc va"

# === WEIGHTS ===
for S in $STATES; do
    echo Generating weights for $S ...
    unzip -oq phase6-state-targets.zip ${S}_targets.csv
    mv -f ${S}_targets.csv $TMD/areas/targets
    pushd $TMD/areas > /dev/null
        rm -f weights/${S}_tmd_weights.csv.gz
        python create_area_weights.py $S > ${S}_local.log
        mv -f ${S}_local.log ../examination/2022
        mv -f weights/${S}_tmd_weights.csv.gz ../examination/2022
    popd > /dev/null
    awk -f log_extract.awk ${S}_local.log
done

# === RESULTS ===
cd $TMD/examination/2022
TC_OPTIONS="--exact --dump --dvars outvars --sqldb"
echo Generating results for US ...
tc tmd.csv 2022 $TC_OPTIONS | grep -v data
sqlite3 tmd-22-#-#-#.db < dbtab.sql
for S in $STATES; do
    echo Generating results for $S ...
    TMD_AREA=$S tc tmd.csv 2022 $TC_OPTIONS | grep -v data
    sqlite3 tmd_${S}-22-#-#-#.db < dbtab.sql
done

# === CLEANUP ===
rm -f ./*tmd*csv*
rm -f ./tmd*-22-*
rm -f ./*.log
exit 0


# WE HAVE THESE RESULTS ON 2025-01-16:
#
# (taxcalc-dev) 2022% time ./generate_tmd_results.sh
# Generating weights for ak ...
#   ::loop,delta,misses,exectime(secs):   1   1.000000e-09   0   401.1
# DISTRIBUTION OF TARGET ACT/EXP RATIOS (n=146):
#   with REGULARIZATION_DELTA= 1.000000e-09
# low bin ratio    high bin ratio    bin #    cum #     bin %     cum %
# >=     0.996000, <     1.004000:     146      146   100.00%   100.00%
# MINIMUM VALUE OF TARGET ACT/EXP RATIO = 0.998
# MAXIMUM VALUE OF TARGET ACT/EXP RATIO = 1.001
# Generating weights for mn ...
#   ::loop,delta,misses,exectime(secs):   1   1.000000e-09   0   240.2
# DISTRIBUTION OF TARGET ACT/EXP RATIOS (n=147):
#   with REGULARIZATION_DELTA= 1.000000e-09
# low bin ratio    high bin ratio    bin #    cum #     bin %     cum %
# >=     0.996000, <     1.004000:     147      147   100.00%   100.00%
# MINIMUM VALUE OF TARGET ACT/EXP RATIO = 0.999
# MAXIMUM VALUE OF TARGET ACT/EXP RATIO = 1.001
# Generating weights for nj ...
#   ::loop,delta,misses,exectime(secs):   1   1.000000e-09   0   180.3
# DISTRIBUTION OF TARGET ACT/EXP RATIOS (n=147):
#   with REGULARIZATION_DELTA= 1.000000e-09
# low bin ratio    high bin ratio    bin #    cum #     bin %     cum %
# >=     0.996000, <     1.004000:     147      147   100.00%   100.00%
# MINIMUM VALUE OF TARGET ACT/EXP RATIO = 0.999
# MAXIMUM VALUE OF TARGET ACT/EXP RATIO = 1.000
# Generating weights for nm ...
#   ::loop,delta,misses,exectime(secs):   1   1.000000e-09   0   168.4
# DISTRIBUTION OF TARGET ACT/EXP RATIOS (n=147):
#   with REGULARIZATION_DELTA= 1.000000e-09
# low bin ratio    high bin ratio    bin #    cum #     bin %     cum %
# >=     0.996000, <     1.004000:     147      147   100.00%   100.00%
# MINIMUM VALUE OF TARGET ACT/EXP RATIO = 1.000
# MAXIMUM VALUE OF TARGET ACT/EXP RATIO = 1.001
# Generating weights for sc ...
#   ::loop,delta,misses,exectime(secs):   1   1.000000e-09   0   231.2
# DISTRIBUTION OF TARGET ACT/EXP RATIOS (n=147):
#   with REGULARIZATION_DELTA= 1.000000e-09
# low bin ratio    high bin ratio    bin #    cum #     bin %     cum %
# >=     0.996000, <     1.004000:     147      147   100.00%   100.00%
# MINIMUM VALUE OF TARGET ACT/EXP RATIO = 0.999
# MAXIMUM VALUE OF TARGET ACT/EXP RATIO = 1.000
# Generating weights for va ...
#   ::loop,delta,misses,exectime(secs):   1   1.000000e-09   2   456.7
# DISTRIBUTION OF TARGET ACT/EXP RATIOS (n=147):
#   with REGULARIZATION_DELTA= 1.000000e-09
# low bin ratio    high bin ratio    bin #    cum #     bin %     cum %
# >=     0.900000, <     0.990000:       1        1     0.68%     0.68%
# >=     0.990000, <     0.996000:       0        1     0.00%     0.68%
# >=     0.996000, <     1.004000:     145      146    98.64%    99.32%
# >=     1.004000, <     1.010000:       0      146     0.00%    99.32%
# >=     1.010000, <     1.100000:       1      147     0.68%   100.00%
# MINIMUM VALUE OF TARGET ACT/EXP RATIO = 0.948
# MAXIMUM VALUE OF TARGET ACT/EXP RATIO = 1.050
# Generating results for US ...
# 9654.475|420.299|396.303|1114.474|29.909|473.755|14851.081|11842.505|116.717
# 2289.792
# Generating results for ak ...
# 19.843|0.554|0.607|1.911|0.109|0.906|29.261|22.975|0.217
# 3.796
# Generating results for mn ...
# 179.963|6.321|6.362|20.908|0.71|9.72|267.728|212.373|1.392
# 38.722
# Generating results for nj ...
# 346.77|16.023|10.934|32.693|1.049|16.227|506.593|415.249|2.642
# 82.935
# Generating results for nm ...
# 41.203|1.354|2.04|3.184|0.199|2.932|63.316|46.637|0.947
# 7.416
# Generating results for sc ...
# 119.387|3.966|4.309|14.67|0.429|8.407|185.522|141.508|2.133
# 24.706
# Generating results for va ...
# 276.779|10.304|8.814|28.721|0.767|13.378|408.918|327.695|2.597
# 61.405
# ./generate_tmd_results.sh  6396.15s user 93.15s system 310% cpu 34:52.81 total
