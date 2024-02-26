#!/bin/zsh
# Execute several Tax-Calculator runs using the execute.sh script.
# USAGE:
#   ./runs.sh  DATAFILE(without_trailing_.csv)

if [[ "$#" -ne 1 ]]; then
  echo "USAGE: ./runs.sh DATAFILE(without_trailing_.csv)" >&2
  exit 1
fi
if [[ "$1" == "puf" ]]; then
    unzip -oq puf.csv.zip
fi
if [[ "$1" == "td23" ]]; then
    unzip -oq td23.csv.zip
fi
if [[ "$1" == "pe23" ]]; then
    unzip -oq pe23.csv.zip
fi

date

echo CLP
./execute.sh $1.csv 2023 clp

echo CTC
./execute.sh $1.csv 2023 ctc

echo EITC
./execute.sh $1.csv 2023 eitc

echo SSBEN
./execute.sh $1.csv 2023 ssben

echo NIIT
./execute.sh $1.csv 2023 niit

echo CGQD
./execute.sh $1.csv 2023 cgqd

echo QBID
./execute.sh $1.csv 2023 qbid

date

head -15 $1-*clp*tab.text | awk '$1~/Wei/||$1~/Ret/||$1~/A/' > $1.res-actual
echo >> $1.res-actual
tail -1 $1-*tab.text >> $1.res-actual
diff -q $1.res-actual $1.res-expect
RC=$?
if [[ $RC -eq 0 ]]; then
    echo "NO ACTUAL-vs-EXPECT $1 differences"
    rm -f $1.res-actual
    rm -f *doc.text *tab.text *.csv
else
    diff $1.res-actual $1.res-expect
fi

exit 0
