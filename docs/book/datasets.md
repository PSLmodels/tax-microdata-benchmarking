# Datasets

This repo creates a number of datasets. The full process (including redundant 2015 datasets for validation) is as follows (datasets highlights in bold):

1. Take the **2015 PUF**.
2. Impute missing demographics, and create the **PolicyEngine hierarchical 2015 PUF**.
3. Convert the **PolicyEngine hierarchical 2015 PUF** to the **Tax-Calculator 2015 PUF**.
4. Uprate the **2015 PUF** to the **2021 PUF** (using SOI statistics), impute missing demographics and create the **PolicyEngine hierarchical 2021 PUF**.
5. Convert the **PolicyEngine hierarchical 2021 PUF** to the **Tax-Calculator 2021 PUF**.
6. Take the **2021 CPS**.
7. Create the **PolicyEngine hierarchical 2021 CPS**.
8. Stack the non-filer (PolicyEngine US-determined) part of the **PolicyEngine hierarchical 2021 CPS** onto the **PolicyEngine hierarchical 2021 PUF** to create the **TMD file**.
