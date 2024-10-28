# areas / targets

Contains sub-national area targets files. word


## Targets File Content

An areas targets file is a CSV-formatted file with its first row
containing column names and its second row containing the area
population target.  Each subsequent row contains another target.
Rows after the first two that start with a `#` character are
considered comments and are skipped.

Here are the column names and their valid values:

1. __`varname`__: any Tax-Calculator input variable name plus any
                  Tax-Calculator calculated variable in the list of
                  cached variables in the `tmd/storage/__init__.py` file
2. __`count`__: integer in [0,4] range:
      * count==0 implies dollar total of varname is tabulated
      * count==1 implies number of tax units
                 with __any__ value of varname is tabulated
      * count==2 implies number of tax units
                 with a __nonzero__ value of varname is tabulated
      * count==3 implies number of tax units
                 with a __positive__ value of varname is tabulated
      * count==4 implies number of tax units
                 with a __negative__ value of varname is tabulated
3. __`scope`__: integer in [0,2] range:
      * scope==0 implies all tax units are tabulated
      * scope==1 implies only PUF-derived filing units are tabulated
      * scope==2 implies only CPS-derived filing units are tabulated
4. __`agilo`__: float representing lower bound of the AGI range (which
                is included in the range) that is tabulated.
5. __`agihi`__: float representing upper bound of the AGI range (which
                is excluded from the range) that is tabulated.
6. __`fstatus`__: integer in [0,5] range:
      * fstatus=0 implies all filing statuses are tabulated
      * other fstatus values imply just the tax units with
              the Tax-Calculator `MARS` variable equal to fstatus
              are included in the tabulation
7. __`target`__: target amount:
      * dollars if count==0
      * number of tax units if count>0
