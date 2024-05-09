import taxcalc as tc
import pandas as pd


def add_taxcalc_outputs(
    flat_file: pd.DataFrame,
    time_period: int,
) -> pd.DataFrame:
    """
    Run a flat file through Tax-Calculator.

    Args:
        flat_file (pd.DataFrame): The flat file to run through Tax-Calculator.
        time_period (int): The year to run the simulation for.

    Returns:
        pd.DataFrame: The Tax-Calculator output.
    """
    input_data = tc.Records(data=flat_file, start_year=time_period)
    policy = tc.Policy()
    simulation = tc.Calculator(records=input_data, policy=policy)
    simulation.calc_all()
    output = simulation.dataframe(None, all_vars=True)
    output.s006 = flat_file.s006  # Tax-Calculator seems to change the weights.
    return output
