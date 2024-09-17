"""
Call create_area_weights.py for each out-of-date or non-existent
weights file for which there is a targets file, and remove each
weights file for which there is no corresponding targets file.
"""

import sys
import time
import subprocess
from tmd.areas import AREAS_FOLDER
from tmd.storage import STORAGE_FOLDER

OTHER_DEPENDENCIES = [
    AREAS_FOLDER / "create_area_weights.py",
    STORAGE_FOLDER / "output" / "tmd.csv.gz",
    STORAGE_FOLDER / "output" / "tmd_weights.csv.gz",
    STORAGE_FOLDER / "output" / "tmd_growfactors.csv",
    STORAGE_FOLDER / "input" / "cbo_population_forecast.yaml",
    # Tax-Calculator is a dependency, so do "make tmd_files" when upgrading T-C
]


def time_of_newest_other_dependency():
    """
    Return time of newest file in the OTHER_DEPENDENCIES list.
    """
    max_dep_time = 0.0
    for dpath in OTHER_DEPENDENCIES:
        max_dep_time = max(dpath.stat().st_mtime, max_dep_time)
    return max_dep_time


# --- High-level logic of the script


def make_all_areas(make_only_list=None):
    """
    Call create_area_weights.py for each out-of-date or non-existent
    weights file for which there is a targets file.
    """
    # remove each weights file for which there is no correponding targets file
    wfolder = AREAS_FOLDER / "weights"
    wpaths = sorted(list(wfolder.glob("*_tmd_weights.csv.gz")))
    for wpath in wpaths:
        area = wpath.name.split("_")[0]
        tpath = AREAS_FOLDER / "targets" / f"{area}_targets.csv"
        lpath = AREAS_FOLDER / "targets" / f"{area}.log"
        if not tpath.exists():
            print(f"removing orphan weights file {wpath.name}")
            wpath.unlink()
            lpath.unlink(missing_ok=True)

    # prepare area list of not up-to-date weights files
    todo_areas = []
    newest_dtime = time_of_newest_other_dependency()
    tfolder = AREAS_FOLDER / "targets"
    tpaths = sorted(list(tfolder.glob("*_targets.csv")))
    for tpath in tpaths:
        area = tpath.name.split("_")[0]
        if make_only_list and area not in make_only_list:
            continue  # skip this area
        wpath = AREAS_FOLDER / "weights" / f"{area}_tmd_weights.csv.gz"
        if wpath.exists():
            wtime = wpath.stat().st_mtime
        else:
            wtime = 0.0
        up_to_date = wtime > max(newest_dtime, tpath.stat().st_mtime)
        if not up_to_date:
            todo_areas.append(area)

    # show processing plan
    if todo_areas:
        msg = "(press Ctrl-C to stop)"
        print(f"Plan to create_area_weights for the following areas {msg}:")
        area_num = 0
        for area in todo_areas:
            area_num += 1
            sys.stdout.write(f"{area:>5s}")
            if area_num % 15 == 0:
                sys.stdout.write("\n")
        if area_num % 15 != 0:
            sys.stdout.write("\n")

    # process each target file for which the weights file is not up-to-date
    for area in todo_areas:
        time0 = time.time()
        print(f"creating weights file and log file for {area} ...")
        cmd = [
            "python",
            str(OTHER_DEPENDENCIES[0]),
            area,
        ]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )
        logpath = AREAS_FOLDER / "weights" / f"{area}.log"
        with open(logpath, "w", encoding="utf-8") as logfile:
            logfile.write(result.stdout)
        exectime = time.time() - time0
        if result.returncode != 0:
            print(f"  ... failed after {exectime:.1f} secs")
        else:
            print(f"  ... finished after {exectime:.1f} secs")

    return 0


if __name__ == "__main__":
    sys.exit(make_all_areas())
