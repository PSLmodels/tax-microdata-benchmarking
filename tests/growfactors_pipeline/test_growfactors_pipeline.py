"""
Reproducibility tests for the growth factors pipeline.

Each test re-runs a pipeline stage into a temporary directory and
compares the result with the committed artifact, which catches source
data in tmd/growfactors/data being edited without the downstream
artifacts being regenerated.

These tests are excluded from the default `make test` run (they rebuild
both pipeline stages); run them with:
    make -C tmd/growfactors test
"""

import pandas as pd

from tmd.growfactors import DATA_FOLDER
from tmd.growfactors import build_growfactors, build_stage1_factors

STAGE1_FILE = DATA_FOLDER / "Stage_I_factors.csv"


def read_text(path):
    """
    Return contents of path with any CRLF line endings normalized, so
    that the historical line endings of a committed file do not cause a
    spurious mismatch.
    """
    return path.read_text().replace("\r\n", "\n")


def test_stage1_factors_reproducible(tmp_path, monkeypatch):
    """
    Check that build_stage1_factors reproduces the committed
    data/Stage_I_factors.csv from the committed source data.
    """
    rebuilt = tmp_path / "Stage_I_factors.csv"
    monkeypatch.setattr(build_stage1_factors, "OUTFILE", rebuilt)
    build_stage1_factors.create_stage1_factors()
    assert read_text(rebuilt) == read_text(STAGE1_FILE)


def test_growfactors_reproducible(tmp_path, monkeypatch):
    """
    Check that build_growfactors reproduces the committed growfactors
    file for the current data vintage.
    """
    vintage = build_growfactors.DEFAULT_VINTAGE
    committed = build_growfactors.growfactors_path(vintage)
    rebuilt = tmp_path / committed.name
    monkeypatch.setattr(
        build_growfactors, "growfactors_path", lambda _vintage: rebuilt
    )
    build_growfactors.create_growfactors_file(vintage)
    assert read_text(rebuilt) == read_text(committed)


def test_growfactors_have_no_missing_values():
    """
    Check that no growth factor is missing in any year covered by the
    Stage_I_factors.csv file.  Benefit growth rates stop several years
    before the last stage 1 year, so the benefit factors must be padded
    rather than left undefined.
    """
    vintage = build_growfactors.DEFAULT_VINTAGE
    committed = build_growfactors.growfactors_path(vintage)
    gfdf = pd.read_csv(committed, index_col="YEAR")
    stage1 = pd.read_csv(STAGE1_FILE, index_col="YEAR")
    assert gfdf.index.max() == stage1.index.max()
    assert gfdf.isnull().sum().sum() == 0
