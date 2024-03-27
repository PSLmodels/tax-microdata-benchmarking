def test_flat_file_runs():
    import taxcalc as tc
    from tax_microdata_benchmarking.create_flat_file import create_flat_file
    import pandas as pd

    cps_based_flat_file = create_flat_file(source_dataset="enhanced_cps_2022")

    try:
        puf_based_flat_file = create_flat_file(source_dataset="puf_2022")
        nonfilers_file = cps_based_flat_file[
            cps_based_flat_file.is_tax_filer == 0
        ]
        stacked_file = pd.concat([puf_based_flat_file, nonfilers_file])
        cps_based_flat_file.to_csv(
            "tax_microdata_cps_based.csv.gz", index=False
        )
        puf_based_flat_file.to_csv(
            "tax_microdata_puf_based.csv.gz", index=False
        )
        nonfilers_file.to_csv("tax_microdata_nonfilers.csv.gz", index=False)
        stacked_file.to_csv("tax_microdata.csv.gz", index=False)
    except:
        print("PUF-based data not available.")
        cps_based_flat_file.to_csv("tax_microdata.csv.gz", index=False)

    input_data = tc.Records("tax_microdata.csv.gz")
    policy = tc.Policy()
    simulation = tc.Calculator(records=input_data, policy=policy)

    simulation.calc_all()
