import streamlit as st

st.set_page_config(layout="wide")

from utils import puf_pe_21, pe_21, td_23
from tax_microdata_benchmarking.utils.taxcalc import (
    get_tc_variable_description,
    get_tc_is_input,
)
import pandas as pd
import numpy as np

st.title("AGI breakdowns")
