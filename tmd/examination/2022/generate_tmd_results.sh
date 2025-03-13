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
    zip state-weights.zip ${S}_tmd_weights.csv.gz
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
