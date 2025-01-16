-- tabulate iitax component results using only PUF records:
SELECT
    round(sum(s006*e00200)*1e-9, 3),  -- wage and salary (ALL)
    round(sum(s006*e00600)*1e-9, 3),  -- ordinary dividends
    round(sum(s006*e00900)*1e-9, 3),  -- Sch C net business income
    round(sum(s006*e02000)*1e-9, 3),  -- rent,partnership,S-Corp income (Sch E)
    round(sum(s006*e02300)*1e-9, 3),  -- unemployment compensation
    round(sum(s006*c02500)*1e-9, 3),  -- taxable social security benefits
    round(sum(s006*c00100)*1e-9, 3),  -- adjusted gross income (AGI)
    round(sum(s006*c04800)*1e-9, 3),  -- taxable income
    round(sum(s006*refund)*1e-9, 3)   -- refundable credits
    --round(sum(s006*iitax)*1e-9, 3)  -- total iitax liability (after credits)
  FROM baseline
  WHERE data_source = 1;
SELECT
    round(sum(s006*iitax)*1e-9, 3)    -- total POSITIVE iitax liability
FROM baseline
WHERE data_source = 1
  AND iitax >= 0;


-- USING TC_OPTIONS="--exact --dump --dvars outvars --sqldb"
-- 2022 US targets from SOI Bulletin (Spring 2024) in $B:
-- < https://www.irs.gov/pub/irs-soi/soi-a-inpre-id2401.pdf >
--                             SOI        TMD   (TMD/SOI-1)*100(%)
--: wage_and_salary       9648.553   9654.475   +0.1
--: ordin_dividends        420.403    420.299   -0.0
--: SchC_net_income        395.136    396.303   +0.3
--: SchE_part_Scorp       1108.445   1114.474   +0.5
--: unemploy_compen         29.554     29.909   +1.2
--: taxable_soc_sec        471.017    473.755   +0.6
--: adj_gross_income     15142.763  14851.081   -1.9
--: taxable_income       11954.522  11842.505   -0.9
--: refundable_credits     106.380    116.717   +9.7
--: itax_after_credits    2285.496   2289.792   +0.2
--
-- % awk '$1~/--:/{print 100*($4/$3-1)}' dbtab-by-component.sql
-- 0.0613771
-- -0.0247382
-- 0.295341
-- 0.543915
-- 1.20119
-- 0.581295
-- -1.92621
-- -0.937026
-- 9.71705
-- 0.187968
