def test_flat_file_runs():
    import taxcalc as tc

    input_data = tc.Records("tax_microdata.csv.gz")
    policy = tc.Policy()
    simulation = tc.Calculator(records=input_data, policy=policy)

    simulation.calc_all()
