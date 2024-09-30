"""
Call create_area_weights_file() in the create_area_weights.py module
for each out-of-date or non-existent weights file for which there is
a targets file, and remove each weights and log file for which there
is no corresponding targets file.
"""

import sys
import time
from multiprocessing import Pool
from tmd.areas.create_area_weights import (
    create_area_weights_file,
    TAXCALC_AGI_CACHE,
)
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


def to_do_areas(make_only_list=None):
    """
    Return list of areas that need to have a weights file created.
    """
    # remove each weights file for which there is no correponding targets file
    wfolder = AREAS_FOLDER / "weights"
    wpaths = sorted(list(wfolder.glob("*_tmd_weights.csv.gz")))
    for wpath in wpaths:
        area = wpath.name.split("_")[0]
        tpath = AREAS_FOLDER / "targets" / f"{area}_targets.csv"
        lpath = AREAS_FOLDER / "weights" / f"{area}.log"
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
        tpath_time = tpath.stat().st_mtime
        ppath = AREAS_FOLDER / "targets" / f"{area}_params.yaml"
        if ppath.exists():
            ppath_time = ppath.stat().st_mtime
        else:
            ppath_time = 0.0
        up_to_date = wtime > max(newest_dtime, tpath_time, ppath_time)
        if not up_to_date:
            todo_areas.append(area)
    return todo_areas


def create_area_weights(area: str):
    """
    Call create_area_weights_file for specified area.
    """
    time0 = time.time()
    create_area_weights_file(
        area,
        write_log=True,
        write_file=True,
        write_cache=True,
    )
    time1 = time.time()
    print(f"... {area} exectime(secs)= {(time1 - time0):.1f}")


# --- High-level logic of the script


def make_all_areas(num_workers, make_only_list=None):
    """
    Call create_area_weights(area) for each out-of-date or non-existent
    area weights file for which there is an area targets file.
    """
    TAXCALC_AGI_CACHE.unlink(missing_ok=True)
    todo_areas = to_do_areas(make_only_list=make_only_list)
    # show processing plan
    if todo_areas:
        print("Create area weights for the following areas:")
        area_num = 0
        for area in todo_areas:
            area_num += 1
            sys.stdout.write(f"{area:>7s}")
            if area_num % 10 == 0:
                sys.stdout.write("\n")
        if area_num % 10 != 0:
            sys.stdout.write("\n")
    else:
        sys.stdout.write("Nothing to do\n")
    # process each target file for which the weights file is not up-to-date
    with Pool(num_workers) as pool:
        pool.map(create_area_weights, todo_areas)
    TAXCALC_AGI_CACHE.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    WORKERS = 1
    if len(sys.argv) == 2:
        WORKERS = int(sys.argv[1])
        if WORKERS < 1:
            sys.stderr.write(f"ERROR: {WORKERS} is not a positive integer\n")
            sys.exit(1)
    sys.exit(make_all_areas(WORKERS))
