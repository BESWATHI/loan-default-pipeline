-- Modeling table: finished loans only (fully paid or defaulted), with the target label.
-- Loans that are still 'Current' or 'Late' have no final outcome yet, so they are excluded.
with loans as (
    select * from {{ ref('stg_loans') }}
),

finished as (
    select *
    from loans
    where loan_status in (
        'Fully Paid',
        'Charged Off',
        'Default',
        'Does not meet the credit policy. Status:Fully Paid',
        'Does not meet the credit policy. Status:Charged Off'
    )
)

select
    loan_id,
    issue_date,
    extract(year from issue_date)::int as issue_year,
    loan_status,
    case
        when loan_status in (
            'Charged Off',
            'Default',
            'Does not meet the credit policy. Status:Charged Off'
        ) then 1
        else 0
    end as is_default,

    loan_amnt,
    term_months,
    int_rate,
    installment,
    grade,
    emp_length_years,
    home_ownership,
    annual_inc,
    verification_status,
    purpose,
    addr_state,
    application_type,
    dti,
    delinq_2yrs,
    fico_range_low,
    fico_range_high,
    inq_last_6mths,
    open_acc,
    pub_rec,
    revol_bal,
    revol_util,
    total_acc,
    mort_acc,
    pub_rec_bankruptcies,
    round(((issue_date - earliest_credit_date) / 365.25)::numeric, 2)::double precision
        as credit_history_years

from finished
