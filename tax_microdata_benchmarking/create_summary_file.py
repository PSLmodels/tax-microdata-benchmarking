from tax_microdata_benchmarking.create_flat_file import taxcalc_extension
from policyengine_us import Simulation
import pandas as pd
from pathlib import Path

sim = Simulation(reform=taxcalc_extension, situation={"person_id": 1})
taxcalc_cps = pd.read_csv("cps.csv.gz")

summary_file = """# PolicyEngine US Tax-Calculator flat file

This file contains a summary of the Tax-Calculator microdata file. It is intended to be used as a reference for the Tax-Calculator microdata file.
"""

added_columns = []

variables = sim.tax_benefit_system.variables
for variable in variables.values():
    if variable.name.startswith("tc_"):
        added_columns.append(variable.name[3:])

# Add 'The flat file currently has X out of Y (Z%) columns in the Tax-Calculator CPS microdata file'.

summary_file += f"\nThe flat file currently has {len(added_columns)} out of 68 ({len(added_columns) / len(taxcalc_cps.columns):.0%}) columns in the Tax-Calculator CPS microdata file.\n\n"
missing_columns = [
    column for column in taxcalc_cps.columns if column not in added_columns
]
summary_file += f"Missing columns: \n- " + "\n- ".join(missing_columns) + "\n"

for variable in variables.values():
    if variable.name.startswith("tc_"):
        summary_file += f"\n## {variable.name[3:]}\n\n{variable.label}\n\n"

FOLDER = Path(__file__).parent

with open(FOLDER / "summary.md", "w") as file:
    file.write(summary_file)
