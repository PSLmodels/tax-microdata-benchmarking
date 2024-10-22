from pathlib import Path

STORAGE_FOLDER = Path(__file__).parent

CACHED_TAXCALC_VARIABLES = [
    "c00100",  # AGI
    "iitax",  # individual income tax liability (including refundable credits)
]
