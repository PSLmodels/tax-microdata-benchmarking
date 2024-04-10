def test_flat_file_runs():
    import taxcalc as tc
    from tax_microdata_benchmarking.create_flat_file import (
        create_stacked_flat_file,
    )
    import pandas as pd

    stacked_file = create_stacked_flat_file(target_year=2024, use_puf=False)
    stacked_file.to_csv(
        "tax_microdata.csv.gz", index=False, compression="gzip"
    )

    input_data = tc.Records("tax_microdata.csv.gz")
    policy = tc.Policy()
    simulation = tc.Calculator(records=input_data, policy=policy)

    simulation.calc_all()
