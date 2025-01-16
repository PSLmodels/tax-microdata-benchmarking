$1~/::loop/{scan=1; print $0; next}
$0~/final delta loop iterations/{print $0; next}
$1~/message:/{print $0; next}
scan==1 && $0~/DISTRIBUTION OF TARGET/{prnt=1}
scan==1 && $0~/AREA-OPTIMIZED/{prnt=0}
prnt==1{print $0}
