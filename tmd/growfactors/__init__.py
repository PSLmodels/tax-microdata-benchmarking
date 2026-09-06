"""
Growth factors pipeline: source data --> Tax-Calculator-style growfactors.

See tmd/growfactors/README.md for an overview and for how to run the
pipeline, and docs/annual_update.md for the annual source-data refresh
procedure.
"""

from pathlib import Path

GROWFACTORS_FOLDER = Path(__file__).resolve().parent
DATA_FOLDER = GROWFACTORS_FOLDER / "data"
