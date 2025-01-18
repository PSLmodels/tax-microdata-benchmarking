# The examination/2022/precision.sh script using bootstapping methods to
# estimate the sampling precision of the weighted positive iitax estimate.
# PREREQUISITE:
#   (taxcalc-dev) tax-microdata-benchmarking% make test
# USAGE:
#   (taxcalc-dev) 2022% ./precision.sh

# === SETUP ===
TMD=../..
cp $TMD/storage/output/tmd*csv* .
gunzip -f tmd.csv.gz
STATES="ak mn nj nm sc va"
STATES=""

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
done

# === RESULTS ===
cd $TMD/examination/2022
TC_OPTIONS="--exact --dump --dvars precisionvars"
echo Generating results for US ...
tc tmd.csv 2022 $TC_OPTIONS | grep -v data
python bootstrap_sampling.py tmd-22-#-#-#.csv
for S in $STATES; do
    echo Generating results for $S ...
    TMD_AREA=$S tc tmd.csv 2022 $TC_OPTIONS | grep -v data
    python bootstrap_sampling.py tmd_${S}-22-#-#-#.csv
done

# === CLEANUP ===
rm -f ./*tmd*csv*
rm -f ./tmd*-22-*
rm -f ./*.log
exit 0
