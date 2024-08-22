#!/bin/zsh
# Execute several Tax-Calculator runs using the execute.sh script.
# USAGE:
#   ./runs.sh  DATAFILE(without_trailing_.csv) YEAR(tailing_two_digits)
#   DATAFILES: puf|pe23|xb23|xb26|tmd
#              (where puf is from the taxdata repository)
#              (where pe23 is the Phase 1 work product)
#              (where xb23 and xb26 are Phase 2 work products)
#              (where tmd is from a Phase 3 work product)

D="DATAFILE(without_trailing_.csv)"
Y="YEAR(tailing_two_digits)"
if [[ "$#" -ne 2 ]]; then
    echo "USAGE: ./runs.sh $D $Y" >&2
    exit 1
fi
OK=0
if [[ "$1" == "puf" ]]; then
    unzip -oq puf.csv.zip
    OK=1
fi
if [[ "$1" == "pe23" ]]; then
    unzip -oq pe23.csv.zip
    if [[ "$2" != "23" ]]; then
        echo "ERROR: YEAR not equal to 23" >&2
        echo "USAGE: ./runs.sh pe23 23" >&2
        exit 1
    fi
    OK=1
fi
if [[ "$1" == "xb23" ]]; then
    unzip -oq xb23.csv.zip
    if [[ "$2" != "23" ]]; then
        echo "ERROR: YEAR not equal to 23" >&2
        echo "USAGE: ./runs.sh xb23 23" >&2
        exit 1
    fi
    OK=1
fi
if [[ "$1" == "xb26" ]]; then
    unzip -oq xb26.csv.zip
    if [[ "$2" != "26" ]]; then
        echo "ERROR: YEAR not equal to 26" >&2
        echo "USAGE: ./runs.sh xb26 26" >&2
        exit 1
    fi
    OK=1
fi
if [[ "$1" == "tmd" ]]; then
    unzip -oq tmd.csv.zip
    OK=1
fi
if [[ "$OK" -ne 1 ]]; then
    echo "ERROR: illegal DATAFILE" >&2
    echo "USAGE: ./runs.sh $D $Y" >&2
    exit 1
fi

date

echo CLP
./execute.sh $1.csv 20$2 clp

echo CTC
./execute.sh $1.csv 20$2 ctc
if [[ -v QUIT ]]; then
    exit 1
fi

echo EITC
./execute.sh $1.csv 20$2 eitc

echo SSBEN
./execute.sh $1.csv 20$2 ssben

echo NIIT
./execute.sh $1.csv 20$2 niit

echo CGQD
./execute.sh $1.csv 20$2 cgqd

echo QBID
./execute.sh $1.csv 20$2 qbid

echo SALT
./execute.sh $1.csv 20$2 salt

date

head -15 $1-$2-*clp*tab.text | \
    awk '$1~/Wei/||$1~/Ret/||$1~/A/' > $1-$2.res-actual
echo >> $1-$2.res-actual
tail -1 $1-$2-*tab.text >> $1-$2.res-actual
diff -q $1-$2.res-actual $1-$2.res-expect
RC=$?
if [[ $RC -eq 0 ]]; then
    echo "NO ACTUAL-vs-EXPECT $1-$2 differences"
    rm -f $1-$2.res-actual
    rm -f *doc.text *tab.text *.csv
else
    diff $1-$2.res-actual $1-$2.res-expect
fi

exit 0
