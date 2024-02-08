def test_flat_file_runs():
    import taxcalc as tc
    from tax_microdata_benchmarking.create_flat_file import create_flat_file

    create_flat_file()

    input_data = tc.Records("tax_microdata.csv.gz")
    policy = tc.Policy()
    simulation = tc.Calculator(records=input_data, policy=policy)

    simulation.calc_all()
