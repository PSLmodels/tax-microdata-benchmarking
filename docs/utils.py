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

"""
					All returns, total	Taxable returns, total
Salaries and wages			"Number of
returns"		126,082,290	87,103,951
			Amount		9,022,352,941	8,193,035,658
Taxable interest			"Number of
returns"		48,990,485	39,236,213
			Amount		103,535,203	95,196,481
Tax-exempt interest [1]			"Number of
returns"		6,569,327	5,942,441
			Amount		55,518,422	52,319,278
Ordinary dividends			"Number of
returns"		32,247,057	27,486,117
			Amount		386,961,461	369,999,552
Qualified dividends [1]			"Number of
returns"		30,524,800	26,162,099
			Amount		295,906,194	283,901,845
State income tax refunds			"Number of
returns"		3,150,440	2,852,012
			Amount		3,567,122	3,269,848
Alimony received			"Number of
returns"		258,837	196,851
			Amount		8,507,104	7,794,550
Business or profession	"Net
income"		"Number of
returns"		21,105,685	11,131,241
			Amount		517,081,772	391,845,112
	"Net
loss"		"Number of
returns"		7,546,660	4,864,730
			Amount		105,580,403	58,987,558
" Capital gain distributions
reported on Form 1040"			"Number of
returns"		4,505,544	3,796,666
			Amount		23,889,533	22,214,764
Sales of capital assets reported on Form 1040, Schedule D [2]	"Taxable
net gain"		"Number of
returns"		20,497,375	17,770,358
			Amount		2,048,795,356	2,003,617,745
	"Taxable
net loss"		"Number of
returns"		8,074,079	6,214,471
			Amount		16,241,889	12,449,661
"Sales of property
other than capital assets"		Net gain	"Number of
returns"		1,106,072	902,287
			Amount		71,724,946	67,224,851
		Net loss	"Number of
returns"		889,755	683,720
			Amount		21,038,506	11,110,541
"Taxable Individual Retirement
Arrangement (IRA) distributions"			"Number of
returns"		15,584,165	13,040,403
			Amount		408,382,461	386,984,025
Pensions and annuities		Total [1]	"Number of
returns"		32,171,355	26,146,876
			Amount		1,506,948,061	1,419,681,261
		Taxable [3]	"Number of
returns"		29,357,159	23,800,727
			Amount		858,038,339	804,861,939
Rent	"Net
income"		"Number of
returns"		4,928,465	3,970,996
			Amount		91,678,200	82,443,516
	"Net loss (includes
nondeductible loss)"		"Number of
returns"		4,490,482	3,695,715
			Amount		75,817,327	61,684,018
Royalty	"Net
income"		"Number of
returns"		1,680,569	1,407,405
			Amount		31,380,913	30,147,950
	"Net
loss"		"Number of
returns"		76,543	67,335
			Amount		660,474	516,041
Farm rental	"Net
income"		"Number of
returns"		350,326	295,334
			Amount		6,186,997	5,636,012
	"Net
loss"		"Number of
returns"		72,972	53,533
			Amount		631,042	451,383
Total rental and royalty	"Net
income"		"Number of
returns"		6,305,037	5,098,234
			Amount		125,168,233	114,802,378
	"Net
loss"		"Number of
returns"		3,496,912	2,710,213
			Amount		56,765,983	42,378,399
Partnership 	"Net
income"		"Number of
returns"		3,201,572	2,798,300
			Amount		469,816,308	460,036,489
	"Net
loss"		"Number of
returns"		1,990,357	1,587,897
			Amount		168,152,043	120,188,214
S corporation	"Net
income"		"Number of
returns"		3,878,815	3,426,302
			Amount		766,681,240	752,703,546
	"Net
loss"		"Number of
returns"		1,453,974	1,020,176
			Amount		92,689,104	53,357,685
Estate and trust	"Net
income"		"Number of
returns"		624,529	568,700
			Amount		49,387,898	48,546,916
	"Net
loss"		"Number of
returns"		49,450	37,341
			Amount		5,899,376	4,560,718
Farm	"Net
income"		"Number of
returns"		449,238	310,547
			Amount		13,337,377	10,933,739
	"Net
loss"		"Number of
returns"		1,274,905	864,243
			Amount		39,479,321	25,537,551
Unemployment compensation			"Number of
returns"		15,809,172	9,637,973
			Amount		208,872,354	129,988,826
Social security benefits	Total [1]		"Number of
returns"		31,293,066	21,585,543
			Amount		791,161,174	581,999,590
	Taxable		"Number of
returns"		23,798,351	21,193,613
			Amount		412,830,233	400,328,413
Foreign-earned income exclusion			"Number of
returns"		430,205	135,423
			Amount		28,104,316	11,028,185
Other income	"Net
income"		"Number of
returns"		5,930,776	4,578,314
			Amount		62,702,551	56,862,540
	"Net
loss"		"Number of
returns"		453,932	276,074
			Amount		12,163,207	7,591,814
Net operating loss			"Number of
returns"		1,155,701	346,922
			Amount		185,261,326	49,476,662
Gambling earnings			"Number of
returns"		2,249,499	1,778,979
			Amount		46,630,967	42,432,731
Cancellation of debt			"Number of
returns"		798,188	590,607
			Amount		6,955,173	5,392,515
"Taxable health savings
account distributions"			"Number of
returns"		533,041	463,498
			Amount		677,875	606,561
Statutory adjustments	Total [3]		"Number of
returns"		32,835,517	21,398,920
			Amount		141,160,696	121,717,396
	"Educator expenses
deduction"		"Number of
returns"		3,115,144	2,723,849
			Amount		826,402	726,411
	"Certain business expenses of
reservists, performing artists, etc."		"Number of
returns"		273,260	214,886
			Amount		2,082,951	1,499,156
	"Health savings
account deduction"		"Number of
returns"		1,933,557	1,731,722
			Amount		5,888,886	5,317,987
	"Moving expenses
adjustment"		"Number of
returns"		94,125	62,677
			Amount		268,955	151,252
	"Deductible part of
self-employment tax"		"Number of
returns"		21,622,374	11,871,962
			Amount		38,595,947	29,295,216
	"Payments to a
Keogh plan"		"Number of
returns"		998,658	944,525
			Amount		28,919,016	28,132,815
	"Self-employed health
insurance deduction"		"Number of
returns"		3,667,399	2,940,416
			Amount		30,805,238	27,379,284
	"Penalty on early
withdrawal of savings"		"Number of
returns"		306,653	235,819
			Amount		123,875	110,513
	"Alimony
paid"		"Number of
returns"		377,781	323,334
			Amount		9,743,923	8,913,794
	IRA payments		"Number of
returns"		2,415,869	2,031,157
			Amount		13,682,667	11,850,372
	"Student loan
interest deduction"		"Number of
returns"		4,941,992	4,008,749
			Amount		4,289,185	3,467,135
	"Other
adjustments"		"Number of
returns"		154,225	115,079
			Amount		4,502,757	3,867,208
"Charitable contributions if
took standard deduction"			"Number of
returns"		47,979,584	37,822,164
			Amount		17,928,056	14,645,410
Basic standard deduction			"Number of
returns"		141,872,935	91,128,892
			Amount		2,452,790,173	1,631,707,478
Additional standard deduction			"Number of
returns"		26,009,049	19,008,069
			Amount		52,824,316	38,818,137
Disaster loss deduction			"Number of
returns"		52,172	38,001
			Amount		924,126	602,968
Total itemized deductions			"Number of
returns"		14,842,685	13,435,335
			Amount		659,680,547	598,354,572
"Qualified business
income deduction"			"Number of
returns"		25,924,668	21,720,910
			Amount		205,779,729	197,308,692
"Total standard or itemized deduction
plus qualified business
income deduction"			"Number of
returns"		156,248,101	104,160,759
			Amount		3,381,030,109	2,475,773,258
Taxable income			"Number of
returns"		128,519,569	104,558,480
			Amount		11,767,185,281	11,410,488,827
Alternative minimum tax			"Number of
returns"		243,550	240,182
			Amount		5,598,598	5,570,698
"Excess advance premium
tax credit repayment"			"Number of
returns"		2,632,104	1,986,729
			Amount		3,862,542	3,380,414
Income tax before credits			"Number of
returns"		127,874,599	104,566,159
			Amount		2,290,478,645	2,252,025,728
"""


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
    "Disaster loss deduction": "g20500",
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
