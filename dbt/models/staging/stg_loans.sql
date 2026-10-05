-- Staging: one row per loan, with clean types and consistent values.
with source as (
    select * from {{ source('raw', 'lending_club_loans') }}
)

select
    trim(id)                                   as loan_id,
    {{ to_month('issue_d') }}                  as issue_date,
    trim(loan_status)                          as loan_status,

    {{ to_num('loan_amnt') }}                  as loan_amnt,
    {{ to_num('term') }}::int                  as term_months,
    {{ to_num('int_rate') }}                   as int_rate,
    {{ to_num('installment') }}                as installment,
    nullif(trim(grade), '')                    as grade,

    case
        when emp_length like '10+%' then 10
        when emp_length like '<%' then 0
        else {{ to_num('emp_length') }}::int
    end                                        as emp_length_years,

    upper(nullif(trim(home_ownership), ''))    as home_ownership,
    {{ to_num('annual_inc') }}                 as annual_inc,
    nullif(trim(verification_status), '')      as verification_status,
    nullif(trim(purpose), '')                  as purpose,
    nullif(trim(addr_state), '')               as addr_state,
    nullif(trim(application_type), '')         as application_type,

    {{ to_num('dti') }}                        as dti,
    {{ to_num('delinq_2yrs') }}                as delinq_2yrs,
    {{ to_num('fico_range_low') }}             as fico_range_low,
    {{ to_num('fico_range_high') }}            as fico_range_high,
    {{ to_num('inq_last_6mths') }}             as inq_last_6mths,
    {{ to_num('open_acc') }}                   as open_acc,
    {{ to_num('pub_rec') }}                    as pub_rec,
    {{ to_num('revol_bal') }}                  as revol_bal,
    {{ to_num('revol_util') }}                 as revol_util,
    {{ to_num('total_acc') }}                  as total_acc,
    {{ to_num('mort_acc') }}                   as mort_acc,
    {{ to_num('pub_rec_bankruptcies') }}       as pub_rec_bankruptcies,
    {{ to_month('earliest_cr_line') }}         as earliest_credit_date

from source
