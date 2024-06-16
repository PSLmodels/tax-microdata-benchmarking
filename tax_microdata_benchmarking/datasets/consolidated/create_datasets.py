from create_pe_puf import *
from create_tc_dataset import *

create_pe_puf_2015()
create_tc_dataset(PUF_2015).to_csv("tc_puf_2015.csv", index=False)
