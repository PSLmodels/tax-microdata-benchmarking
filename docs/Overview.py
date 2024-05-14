import streamlit as st
from tax_microdata_benchmarking.storage import STORAGE_FOLDER
import yaml
import plotly.express as px

st.title("Microdata comparison dashboard")

st.markdown(
    f"This app compares multiple microsimulation model input datasets (after running through Tax-Calculator)."
)
