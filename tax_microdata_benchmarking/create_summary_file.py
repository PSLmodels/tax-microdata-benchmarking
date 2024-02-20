from tax_microdata_benchmarking.create_flat_file import taxcalc_extension
from policyengine_us import Simulation
import pandas as pd
from pathlib import Path
import yaml

FOLDER = Path(__file__).parent
sim = Simulation(reform=taxcalc_extension, situation={"person_id": 1})

with open(FOLDER / "taxcalc_variable_metadata.yaml") as file:
    taxcalc_variable_metadata = yaml.safe_load(file)

tc_variables = taxcalc_variable_metadata["read"]
tc_puf_variables = [
    key
    for key, data in tc_variables.items()
    if "taxdata_puf" in data["availability"]
]

summary_file = """# PolicyEngine US Tax-Calculator flat file

This file contains a summary of the Tax-Calculator microdata file. It is intended to be used as a reference for the Tax-Calculator microdata file.
"""

added_columns = []
added_unnecessary_columns = []

variables = sim.tax_benefit_system.variables
for variable in variables.values():
    if variable.name.startswith("tc_"):
        tc_name = variable.name[3:]
        if tc_name in tc_puf_variables:
            added_columns.append(tc_name)
        else:
            added_unnecessary_columns.append(tc_name)


summary_file += f"\nThe flat file currently has {len(added_columns)} out of 68 ({len(added_columns) / len(tc_puf_variables):.0%}) columns in the Tax-Calculator PUF microdata file.\n\n"
missing_columns = [
    column for column in tc_puf_variables if column not in added_columns
]
summary_file += f"Missing columns: \n- " + "\n- ".join(missing_columns) + "\n"

summary_file += (
    f"\nExtra, non-taxdata-PUF columns: \n- "
    + "\n- ".join(added_unnecessary_columns)
    + "\n"
)

for variable in variables.values():
    if variable.name.startswith("tc_"):
        summary_file += f"\n## {variable.name[3:]}\n\n{variable.label}\n\n"

FOLDER = Path(__file__).parent

with open(FOLDER / "summary.md", "w") as file:
    file.write(summary_file)
