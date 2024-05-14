import pandas as pd
from tax_microdata_benchmarking.storage import STORAGE_FOLDER
from tax_microdata_benchmarking.utils.cloud import download_gh_release_asset
from tqdm import tqdm
import streamlit as st

output = STORAGE_FOLDER / "output"

file_paths = [
    output / "puf_ecps_2021.csv.gz",
    output / "ecps_2021.csv.gz",
    output / "taxdata_puf_2023.csv.gz",
]

for file_path in file_paths:
    if not file_path.exists():
        df = download_gh_release_asset(
            release_name="latest",
            repo="nikhilwoodruff/tax-microdata-benchmarking-releases",
            asset_name=file_path.name,
        )
        df.to_csv(file_path, index=False)


@st.cache_data
def read_csv_cached(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path)


puf_pe_21 = read_csv_cached(output / "puf_ecps_2021.csv.gz")
pe_21 = read_csv_cached(output / "ecps_2021.csv.gz")
td_23 = read_csv_cached(output / "taxdata_puf_2023.csv.gz")


IRS_TOTALS = {
    "Salaries and wages": {
        "All returns": {
            "Number of returns": 126_082_290,
            "Amount": 9_022_352_941,
        },
        "Taxable returns": {
            "Number of returns": 87_103_951,
            "Amount": 8_193_035_658,
        },
    },
    "Taxable interest": {
        "All returns": {
            "Number of returns": 48_990_485,
            "Amount": 103_535_203,
        },
        "Taxable returns": {
            "Number of returns": 39_236_213,
            "Amount": 95_196_481,
        },
    },
    "Tax-exempt interest": {
        "All returns": {
            "Number of returns": 6_569_327,
            "Amount": 55_518_422,
        },
        "Taxable returns": {
            "Number of returns": 5_942_441,
            "Amount": 52_319_278,
        },
    },
    "Ordinary dividends": {
        "All returns": {
            "Number of returns": 32_247_057,
            "Amount": 386_961_461,
        },
        "Taxable returns": {
            "Number of returns": 27_486_117,
            "Amount": 369_999_552,
        },
    },
    "Qualified dividends": {
        "All returns": {
            "Number of returns": 30_524_800,
            "Amount": 295_906_194,
        },
        "Taxable returns": {
            "Number of returns": 26_162_099,
            "Amount": 283_901_845,
        },
    },
    "State income tax refunds": {
        "All returns": {
            "Number of returns": 3_150_440,
            "Amount": 3_567_122,
        },
        "Taxable returns": {
            "Number of returns": 2_852_012,
            "Amount": 3_269_848,
        },
    },
    "Alimony received": {
        "All returns": {
            "Number of returns": 258_837,
            "Amount": 8_507_104,
        },
        "Taxable returns": {
            "Number of returns": 196_851,
            "Amount": 7_794_550,
        },
    },
    "Business or profession net income": {
        "All returns": {
            "Number of returns": 21_105_685,
            "Amount": 517_081_772,
        },
        "Taxable returns": {
            "Number of returns": 11_131_241,
            "Amount": 391_845_112,
        },
    },
    "Business or profession net loss": {
        "All returns": {
            "Number of returns": 7_546_660,
            "Amount": 105_580_403,
        },
        "Taxable returns": {
            "Number of returns": 4_864_730,
            "Amount": 58_987_558,
        },
    },
    "Capital gain distributions reported on Form 1040": {
        "All returns": {
            "Number of returns": 4_505_544,
            "Amount": 23_889_533,
        },
        "Taxable returns": {
            "Number of returns": 3_796_666,
            "Amount": 22_214_764,
        },
    },
    "Sales of capital assets reported on Form 1040, Schedule D, taxable net gain": {
        "All returns": {
            "Number of returns": 20_497_375,
            "Amount": 2_048_795_356,
        },
        "Taxable returns": {
            "Number of returns": 17_770_358,
            "Amount": 2_003_617_745,
        },
    },
    "Sales of capital assets reported on Form 1040, Schedule D, taxable net loss": {
        "All returns": {
            "Number of returns": 8_074_079,
            "Amount": 16_241_889,
        },
        "Taxable returns": {
            "Number of returns": 6_214_471,
            "Amount": 12_449_661,
        },
    },
    "Sales of property other than capital assets, net gain": {
        "All returns": {
            "Number of returns": 1_106_072,
            "Amount": 71_724_946,
        },
        "Taxable returns": {
            "Number of returns": 902_287,
            "Amount": 67_224_851,
        },
    },
    "Sales of property other than capital assets, net loss": {
        "All returns": {
            "Number of returns": 889_755,
            "Amount": 21_038_506,
        },
        "Taxable returns": {
            "Number of returns": 683_720,
            "Amount": 11_110_541,
        },
    },
    "Taxable Individual Retirement Arrangement (IRA) distributions": {
        "All returns": {
            "Number of returns": 15_584_165,
            "Amount": 408_382_461,
        },
        "Taxable returns": {
            "Number of returns": 13_040_403,
            "Amount": 386_984_025,
        },
    },
    "Pensions and annuities total": {
        "All returns": {
            "Number of returns": 32_171_355,
            "Amount": 1_506_948_061,
        },
        "Taxable returns": {
            "Number of returns": 26_146_876,
            "Amount": 1_419_681_261,
        },
    },
    "Pensions and annuities taxable": {
        "All returns": {
            "Number of returns": 29_357_159,
            "Amount": 858_038_339,
        },
        "Taxable returns": {
            "Number of returns": 23_800_727,
            "Amount": 804_861_939,
        },
    },
    "Rent net income": {
        "All returns": {
            "Number of returns": 4_928_465,
            "Amount": 91_678_200,
        },
        "Taxable returns": {
            "Number of returns": 3_970_996,
            "Amount": 82_443_516,
        },
    },
    "Rent net loss": {
        "All returns": {
            "Number of returns": 4_490_482,
            "Amount": 75_817_327,
        },
        "Taxable returns": {
            "Number of returns": 3_695_715,
            "Amount": 61_684_018,
        },
    },
    "Royalty net income": {
        "All returns": {
            "Number of returns": 1_680_569,
            "Amount": 31_380_913,
        },
        "Taxable returns": {
            "Number of returns": 1_407_405,
            "Amount": 30_147_950,
        },
    },
    "Royalty net loss": {
        "All returns": {
            "Number of returns": 76_543,
            "Amount": 660_474,
        },
        "Taxable returns": {
            "Number of returns": 67_335,
            "Amount": 516_041,
        },
    },
    "Farm rental net income": {
        "All returns": {
            "Number of returns": 350_326,
            "Amount": 6_186_997,
        },
        "Taxable returns": {
            "Number of returns": 295_334,
            "Amount": 5_636_012,
        },
    },
    "Farm rental net loss": {
        "All returns": {
            "Number of returns": 72_972,
            "Amount": 631_042,
        },
        "Taxable returns": {
            "Number of returns": 53_533,
            "Amount": 451_383,
        },
    },
    "Total rental and royalty net income": {
        "All returns": {
            "Number of returns": 6_305_037,
            "Amount": 125_168_233,
        },
        "Taxable returns": {
            "Number of returns": 5_098_234,
            "Amount": 114_802_378,
        },
    },
    "Total rental and royalty net loss": {
        "All returns": {
            "Number of returns": 3_496_912,
            "Amount": 56_765_983,
        },
        "Taxable returns": {
            "Number of returns": 2_710_213,
            "Amount": 42_378_399,
        },
    },
    "Partnership net income": {
        "All returns": {
            "Number of returns": 3_201_572,
            "Amount": 469_816_308,
        },
        "Taxable returns": {
            "Number of returns": 2_798_300,
            "Amount": 460_036_489,
        },
    },
    "Partnership net loss": {
        "All returns": {
            "Number of returns": 1_990_357,
            "Amount": 168_152_043,
        },
        "Taxable returns": {
            "Number of returns": 1_587_897,
            "Amount": 120_188_214,
        },
    },
    "S corporation net income": {
        "All returns": {
            "Number of returns": 3_878_815,
            "Amount": 766_681_240,
        },
        "Taxable returns": {
            "Number of returns": 3_426_302,
            "Amount": 752_703_546,
        },
    },
    "S corporation net loss": {
        "All returns": {
            "Number of returns": 1_453_974,
            "Amount": 92_689_104,
        },
        "Taxable returns": {
            "Number of returns": 1_020_176,
            "Amount": 53_357_685,
        },
    },
    "Estate and trust net income": {
        "All returns": {
            "Number of returns": 624_529,
            "Amount": 49_387_898,
        },
        "Taxable returns": {
            "Number of returns": 568_700,
            "Amount": 48_546_916,
        },
    },
    "Estate and trust net loss": {
        "All returns": {
            "Number of returns": 49_450,
            "Amount": 5_899_376,
        },
        "Taxable returns": {
            "Number of returns": 37_341,
            "Amount": 4_560_718,
        },
    },
    "Farm net income": {
        "All returns": {
            "Number of returns": 449_238,
            "Amount": 13_337_377,
        },
        "Taxable returns": {
            "Number of returns": 310_547,
            "Amount": 10_933_739,
        },
    },
    "Farm net loss": {
        "All returns": {
            "Number of returns": 1_274_905,
            "Amount": 39_479_321,
        },
        "Taxable returns": {
            "Number of returns": 864_243,
            "Amount": 25_537_551,
        },
    },
    "Unemployment compensation": {
        "All returns": {
            "Number of returns": 15_809_172,
            "Amount": 208_872_354,
        },
        "Taxable returns": {
            "Number of returns": 9_637_973,
            "Amount": 129_988_826,
        },
    },
    "Social security benefits total": {
        "All returns": {
            "Number of returns": 31_293_066,
            "Amount": 791_161_174,
        },
        "Taxable returns": {
            "Number of returns": 21_585_543,
            "Amount": 581_999_590,
        },
    },
    "Social security benefits taxable": {
        "All returns": {
            "Number of returns": 23_798_351,
            "Amount": 412_830_233,
        },
        "Taxable returns": {
            "Number of returns": 21_193_613,
            "Amount": 400_328_413,
        },
    },
    "Foreign-earned income exclusion": {
        "All returns": {
            "Number of returns": 430_205,
            "Amount": 28_104_316,
        },
        "Taxable returns": {
            "Number of returns": 135_423,
            "Amount": 11_028_185,
        },
    },
    "Other income net income": {
        "All returns": {
            "Number of returns": 5_930_776,
            "Amount": 62_702_551,
        },
        "Taxable returns": {
            "Number of returns": 4_578_314,
            "Amount": 56_862_540,
        },
    },
    "Other income net loss": {
        "All returns": {
            "Number of returns": 453_932,
            "Amount": 12_163_207,
        },
        "Taxable returns": {
            "Number of returns": 276_074,
            "Amount": 7_591_814,
        },
    },
    "Net operating loss": {
        "All returns": {
            "Number of returns": 1_155_701,
            "Amount": 185_261_326,
        },
        "Taxable returns": {
            "Number of returns": 346_922,
            "Amount": 49_476_662,
        },
    },
    "Gambling earnings": {
        "All returns": {
            "Number of returns": 2_249_499,
            "Amount": 46_630_967,
        },
        "Taxable returns": {
            "Number of returns": 1_778_979,
            "Amount": 42_432_731,
        },
    },
    "Cancellation of debt": {
        "All returns": {
            "Number of returns": 798_188,
            "Amount": 6_955_173,
        },
        "Taxable returns": {
            "Number of returns": 590_607,
            "Amount": 5_392_515,
        },
    },
    "Taxable health savings account distributions": {
        "All returns": {
            "Number of returns": 533_041,
            "Amount": 677_875,
        },
        "Taxable returns": {
            "Number of returns": 463_498,
            "Amount": 606_561,
        },
    },
    "Statutory adjustments total": {
        "All returns": {
            "Number of returns": 32_835_517,
            "Amount": 141_160_696,
        },
        "Taxable returns": {
            "Number of returns": 21_398_920,
            "Amount": 121_717_396,
        },
    },
    "Educator expenses deduction": {
        "All returns": {
            "Number of returns": 3_115_144,
            "Amount": 826_402,
        },
        "Taxable returns": {
            "Number of returns": 2_723_849,
            "Amount": 726_411,
        },
    },
    "Certain business expenses of reservists, performing artists, etc.": {
        "All returns": {
            "Number of returns": 273_260,
            "Amount": 2_082_951,
        },
        "Taxable returns": {
            "Number of returns": 214_886,
            "Amount": 1_499_156,
        },
    },
    "Health savings account deduction": {
        "All returns": {
            "Number of returns": 1_933_557,
            "Amount": 5_888_886,
        },
        "Taxable returns": {
            "Number of returns": 1_731_722,
            "Amount": 5_317_987,
        },
    },
    "Moving expenses adjustment": {
        "All returns": {
            "Number of returns": 94_125,
            "Amount": 268_955,
        },
        "Taxable returns": {
            "Number of returns": 62_677,
            "Amount": 151_252,
        },
    },
    "Deductible part of self-employment tax": {
        "All returns": {
            "Number of returns": 21_622_374,
            "Amount": 38_595_947,
        },
        "Taxable returns": {
            "Number of returns": 11_871_962,
            "Amount": 29_295_216,
        },
    },
    "Payments to a Keogh plan": {
        "All returns": {
            "Number of returns": 998_658,
            "Amount": 28_919_016,
        },
        "Taxable returns": {
            "Number of returns": 944_525,
            "Amount": 28_132_815,
        },
    },
    "Self-employed health insurance deduction": {
        "All returns": {
            "Number of returns": 3_667_399,
            "Amount": 30_805_238,
        },
        "Taxable returns": {
            "Number of returns": 2_940_416,
            "Amount": 27_379_284,
        },
    },
    "Penalty on early withdrawal of savings": {
        "All returns": {
            "Number of returns": 306_653,
            "Amount": 123_875,
        },
        "Taxable returns": {
            "Number of returns": 235_819,
            "Amount": 110_513,
        },
    },
    "Alimony paid": {
        "All returns": {
            "Number of returns": 377_781,
            "Amount": 9_743_923,
        },
        "Taxable returns": {
            "Number of returns": 323_334,
            "Amount": 8_913_794,
        },
    },
    "IRA payments": {
        "All returns": {
            "Number of returns": 2_415_869,
            "Amount": 13_682_667,
        },
        "Taxable returns": {
            "Number of returns": 2_031_157,
            "Amount": 11_850_372,
        },
    },
    "Student loan interest deduction": {
        "All returns": {
            "Number of returns": 4_941_992,
            "Amount": 4_289_185,
        },
        "Taxable returns": {
            "Number of returns": 4_008_749,
            "Amount": 3_467_135,
        },
    },
    "Other adjustments": {
        "All returns": {
            "Number of returns": 154_225,
            "Amount": 4_502_757,
        },
        "Taxable returns": {
            "Number of returns": 115_079,
            "Amount": 3_867_208,
        },
    },
    "Charitable contributions if took standard deduction": {
        "All returns": {
            "Number of returns": 47_979_584,
            "Amount": 17_928_056,
        },
        "Taxable returns": {
            "Number of returns": 37_822_164,
            "Amount": 14_645_410,
        },
    },
    "Basic standard deduction": {
        "All returns": {
            "Number of returns": 141_872_935,
            "Amount": 2_452_790_173,
        },
        "Taxable returns": {
            "Number of returns": 91_128_892,
            "Amount": 1_631_707_478,
        },
    },
    "Additional standard deduction": {
        "All returns": {
            "Number of returns": 26_009_049,
            "Amount": 52_824_316,
        },
        "Taxable returns": {
            "Number of returns": 19_008_069,
            "Amount": 38_818_137,
        },
    },
    "Disaster loss deduction": {
        "All returns": {
            "Number of returns": 52_172,
            "Amount": 924_126,
        },
        "Taxable returns": {
            "Number of returns": 38_001,
            "Amount": 602_968,
        },
    },
    "Total itemized deductions": {
        "All returns": {
            "Number of returns": 14_842_685,
            "Amount": 659_680_547,
        },
        "Taxable returns": {
            "Number of returns": 13_435_335,
            "Amount": 598_354_572,
        },
    },
    "Qualified business income deduction": {
        "All returns": {
            "Number of returns": 25_924_668,
            "Amount": 205_779_729,
        },
        "Taxable returns": {
            "Number of returns": 21_720_910,
            "Amount": 197_308_692,
        },
    },
    "Total standard or itemized deduction plus qualified business income deduction": {
        "All returns": {
            "Number of returns": 156_248_101,
            "Amount": 3_381_030_109,
        },
        "Taxable returns": {
            "Number of returns": 104_160_759,
            "Amount": 2_475_773_258,
        },
    },
    "Taxable income": {
        "All returns": {
            "Number of returns": 128_519_569,
            "Amount": 11_767_185_281,
        },
        "Taxable returns": {
            "Number of returns": 104_558_480,
            "Amount": 11_410_488_827,
        },
    },
    "Alternative minimum tax": {
        "All returns": {
            "Number of returns": 243_550,
            "Amount": 5_598_598,
        },
        "Taxable returns": {
            "Number of returns": 240_182,
            "Amount": 5_570_698,
        },
    },
    "Excess advance premium tax credit repayment": {
        "All returns": {
            "Number of returns": 2_632_104,
            "Amount": 3_862_542,
        },
        "Taxable returns": {
            "Number of returns": 1_986_729,
            "Amount": 3_380_414,
        },
    },
    "Income tax before credits": {
        "All returns": {
            "Number of returns": 127_874_599,
            "Amount": 2_290_478_645,
        },
        "Taxable returns": {
            "Number of returns": 104_566_159,
            "Amount": 2_252_025_728,
        },
    },
}

IRS_NAME_TO_TC_NAME = {
    "Salaries and wages": "e00200",
    "Taxable interest": "e00300",
    "Tax-exempt interest": "e00400",
    "Ordinary dividends": "e00600",
    "Qualified dividends": "e00650",
    "State income tax refunds": "e00700",
    "Alimony received": "e00800",
    "Business or profession net income": "e00900",
    "Business or profession net loss": "e00900",
    "Capital gain distributions reported on Form 1040": "e01100",
    "Sales of capital assets reported on Form 1040, Schedule D, taxable net gain": "p23250",
    "Sales of capital assets reported on Form 1040, Schedule D, taxable net loss": "p23250",
    "Sales of property other than capital assets, net gain": "p23250",
    "Sales of property other than capital assets, net loss": "p23250",
    "Taxable Individual Retirement Arrangement (IRA) distributions": "e01400",
    "Pensions and annuities total": "e01500",
    "Pensions and annuities taxable": "e01700",
    "Farm rental net income": "e27200",
    "Farm rental net loss": "e27200",
    "Partnership net income": "e26270",
    "Partnership net loss": "e26270",
    "S corporation net income": "e26270",
    "S corporation net loss": "e26270",
    "Farm net income": "e02100",
    "Farm net loss": "e02100",
    "Unemployment compensation": "e02300",
    "Social security benefits total": "e02400",
    "Social security benefits taxable": "e02500",
    "Statutory adjustments total": "c02900",
    "Educator expenses deduction": "e03220",
    "Health savings account deduction": "e03290",
    "Self-employed health insurance deduction": "e03270",
    "Penalty on early withdrawal of savings": "e03400",
    "Alimony paid": "e03500",
    "IRA payments": "e03150",
    "Student loan interest deduction": "e03210",
    "Basic standard deduction": "standard",
    "Additional standard deduction": "standard",
    "Total itemized deductions": "c21060",
    "Qualified business income deduction": "qbided",
    "Taxable income": "c04800",
    "Alternative minimum tax": "c09600",
    "Income tax before credits": "taxbc",
}

TC_NAME_TO_IRS_NAMES = {}

for variable in IRS_NAME_TO_TC_NAME:
    if IRS_NAME_TO_TC_NAME[variable] not in TC_NAME_TO_IRS_NAMES:
        TC_NAME_TO_IRS_NAMES[IRS_NAME_TO_TC_NAME[variable]] = [variable]
    else:
        TC_NAME_TO_IRS_NAMES[IRS_NAME_TO_TC_NAME[variable]].append(variable)
