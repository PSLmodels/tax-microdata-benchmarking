#!/bin/zsh
# Execute several Tax-Calculator runs using the execute.sh script.
# USAGE:
#   ./runs.sh  DATAFILE(without_trailing_.csv) YEAR(tailing_two_digits)
#   DATAFILES: puf|pe23  (where puf is from the taxdata repository)

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
if [[ "$OK" -ne 1 ]]; then
    echo "ERROR: DATAFILE is neither 'puf' nor 'pe23'" >&2
    echo "USAGE: ./runs.sh $D $Y" >&2
    exit 1
fi

date

echo CLP
./execute.sh $1.csv 20$2 clp

echo CTC
./execute.sh $1.csv 20$2 ctc

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
