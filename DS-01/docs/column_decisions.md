# Column decisions

## Features I'm keeping (19)

| Feature | Why I kept it |
|---|---|
| gender | Known at signup. Probably weak, but I'll let the data decide |
| senior_citizen | Older customers may react differently to price |
| partner | Customers with a partner may be more settled |
| dependents | Households with dependents tend to stay |
| tenure_months | New customers leave the most. It's also my duration for survival analysis |
| phone_service | Known service detail. Almost everyone has it, so likely weak |
| multiple_lines | Known service detail |
| internet_service | Fiber and DSL customers behave differently |
| online_security | Add-on services usually mean a stickier customer |
| online_backup | Same as online_security |
| device_protection | Same as online_security |
| tech_support | Customers with support tend to stay |
| streaming_tv | Shows how engaged the customer is |
| streaming_movies | Same as streaming_tv |
| contract | Month-to-month customers can leave anytime, so this should matter most |
| paperless_billing | Known billing preference |
| payment_method | Some methods are linked to higher churn |
| monthly_charges | The customer's current price |
| total_charges | What the customer has paid so far. Needs fixing first because it's stored as text |

## Columns I'm dropping (13)

| Column | Why I dropped it |
|---|---|
| count | Same value (1) in every row |
| country | Same value in every row |
| state | Same value in every row |
| customer_id | A unique ID. The model could memorize it instead of learning patterns |
| zip_code | About 4 customers per zip, too few to learn from |
| city | Too many cities with too few customers each |
| latitude | Just the zip code written as a number |
| longitude | Same as latitude |
| lat_long | Latitude and longitude joined into one text column |
| churn_label | The same thing as churn_value, so it would give away the answer |
| churn_reason | Only filled in for customers who already left, so it's leakage. I'll use it for EDA only |
| churn_score | IBM's own churn prediction, so it's another model's answer |
| cltv | IBM's own lifetime value. I'll compute my own and use this to compare |

## Target

| Column | Role |
|---|---|
| churn_value | 1 if the customer left, 0 if they stayed |



### Your feature ranking
## Tier	          
 Columns             What it means
Strong (spread 26-40 pts)	contract, internet_service, payment_method, dependents	The model will lean on these
Moderate (13-18 pts)	senior_citizen, paperless_billing, online_security, tech_support, partner	Useful, and they will probably overlap with the strong ones
Useless (under 1 pt)	gender	Male and female churn at nearly the same rate



Kaplan-Meier shows month-to-month customers drop fastest (about 13% remain at 72 months vs 94% for two-year). The Cox model confirms contract, dependents, auto-pay, and add-on services as protective, and fiber and multiple lines as risk factors. Gender and (after adjustment) monthly price and electronic check show no independent effect. Tenure was excluded from the covariates because it is the time axis. Limitations: contract is the one recorded at the snapshot (customers may have switched, which may flatter long contracts), and I did not test the proportional-hazards assumption.