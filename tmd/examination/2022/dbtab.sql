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
