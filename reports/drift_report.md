# Data drift report (PSI)

Reference: loans issued up to 2015 (829,355 loans). Current: loans issued after 2015 (518,744 loans).

| feature | type | PSI | status |
|---|---|---|---|
| application_type | categorical | 0.226 | moderate |
| revol_util | numeric | 0.098 | stable |
| int_rate | numeric | 0.084 | stable |
| purpose | categorical | 0.044 | stable |
| addr_state | categorical | 0.038 | stable |
| installment_to_income | numeric | 0.028 | stable |
| loan_to_income | numeric | 0.022 | stable |
| installment | numeric | 0.022 | stable |
| revol_bal | numeric | 0.019 | stable |
| mort_acc | numeric | 0.016 | stable |
| loan_amnt | numeric | 0.015 | stable |
| inq_last_6mths | numeric | 0.014 | stable |
| verification_status | categorical | 0.014 | stable |
| home_ownership | categorical | 0.011 | stable |
| pub_rec | numeric | 0.010 | stable |
| fico_avg | numeric | 0.010 | stable |
| annual_inc | numeric | 0.009 | stable |
| pub_rec_bankruptcies | numeric | 0.008 | stable |
| grade | categorical | 0.008 | stable |
| dti | numeric | 0.007 | stable |
| emp_length_years | numeric | 0.006 | stable |
| total_acc | numeric | 0.006 | stable |
| open_acc | numeric | 0.006 | stable |
| credit_history_years | numeric | 0.004 | stable |
| delinq_2yrs | numeric | 0.000 | stable |
| term_months | numeric | 0.000 | stable |
