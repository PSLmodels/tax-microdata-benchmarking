# areas / targets

Contains sub-national area targets files.


Targets File Content
====================

An areas targets file is a CSV-formatted file with its first row
containing column names and its second row containing the area
population target.  Each subsequent row contains another target.
Rows after the first two that start with a `#` character are
considered comments and are skipped.

Here are the column names and their valid values:

- **`varname`**: any Tax-Calculator variable name.

- **`count`**: integer in [0,2] range: count==0 implies dollar total
               of varname is tabulated, count==1 implies number of
               tax units with any value of varname is tabulated,
               count==2 implies number of tax units with non-zero
               values of varname is tabulated.

- **`scope`**: integer in [0,2] range: scope==0 implies all tax units,
               scope==1 implies PUF-derived filing units, and
               scope==2 implies CPS-derived filing units.

- **`agilo`**: float representing lower bound of an AGI range,
               which is included in the range.

- **`agihi`**: float representing upper bound of an AGI range,
               which is excluded from the range.

- **`fstatus`**: integer in [0,5] range: fstatus=0 implies all
                 tax units, other fstatus values imply just the
                 tax units with the Tax-Calculator `MARS` variable
                 equal to fstatus.

- **`target`**: target amount (dollars if count==0 or number of
                tax units if count==1 or count==2).
