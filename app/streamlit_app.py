from pathlib import Path
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Retention ROI simulator", layout="wide")
DATA = Path(__file__).resolve().parents[1] / "data" / "customer_scores.csv"

@st.cache_data
def load():
    return pd.read_csv(DATA)

df = load()
REF_DISCOUNT = 0.15   # discount level the simulated uplift represents

def money(x):
    return f"-${abs(x):,.0f}" if x < 0 else f"${x:,.0f}"

st.title("Retention offer ROI simulator")
st.caption("Uplift comes from a simulated campaign (synthetic treatment), so results demonstrate "
           "the method, not real customer behaviour.")

with st.sidebar:
    discount = st.slider("Discount offered (%)", 5, 40, 15) / 100
    months = st.slider("Months the discount lasts", 1, 12, 3)
    share = st.slider("Share of customers to target (%)", 5, 100, 20) / 100
    strategy = st.radio("Who to target", ["Highest uplift", "Highest churn risk", "Random"])

def pick(strategy, share):
    n = max(1, int(len(df) * share))
    if strategy == "Highest uplift":
        return df.nlargest(n, "uplift_cal")
    if strategy == "Highest churn risk":
        return df.nlargest(n, "churn_prob")
    return df.sample(n, random_state=0)

def evaluate(target, discount, months):
    scale = min(discount / REF_DISCOUNT, 2.0)      # assumption: response grows with discount, capped at 2x
    p_saved = (target["uplift_cal"].clip(lower=0) * scale).clip(upper=1)   # negative uplift is noise
    benefit = (p_saved * target["clv_12m"]).sum()   # expected retained profit
    cost = (discount * target["monthly_charges"] * months).sum()   # paid to EVERY targeted customer
    return p_saved.sum(), benefit, cost, benefit - cost

saved, benefit, cost, net = evaluate(pick(strategy, share), discount, months)
roi = net / cost if cost else 0
breakeven = months * benefit / cost if cost else 0   # months at which net return reaches zero

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Customers expected to be saved", f"{saved:,.0f}")
c2.metric("Benefit (retained profit)", money(benefit))
c3.metric("Cost of discounts", money(cost))
c4.metric("Net return", money(net))
c5.metric("ROI (net / cost)", f"{roi:.0%}")
st.caption(f"Under the model's assumptions, this campaign breaks even at about {breakeven:.1f} months of discount.")

st.subheader("Same number of customers, different targeting")
rows = []
for s in ["Highest uplift", "Highest churn risk", "Random"]:
    sv, b, c, nt = evaluate(pick(s, share), discount, months)
    rows.append({"strategy": s, "customers saved": round(sv), "benefit": round(b),
                 "cost": round(c), "net return": round(nt)})
st.dataframe(pd.DataFrame(rows), hide_index=True)

st.subheader("Net return by discount size")
grid = list(range(5, 41, 5))
st.line_chart(pd.DataFrame({"discount %": grid,
    "net return": [evaluate(pick(strategy, share), d / 100, months)[3] for d in grid]}).set_index("discount %"))

st.subheader("Net return by months the discount lasts")
st.line_chart(pd.DataFrame({"months": list(range(1, 13)),
    "net return": [evaluate(pick(strategy, share), discount, m)[3] for m in range(1, 13)]}).set_index("months"))

st.subheader("Net return by share of customers targeted")
sgrid = [5, 10, 15, 20, 30, 40, 60, 80, 100]
st.line_chart(pd.DataFrame({"share %": sgrid,
    "net return": [evaluate(pick(strategy, s / 100), discount, months)[3] for s in sgrid]}).set_index("share %"))

st.subheader("Customer segments")
st.dataframe(df.groupby("segment").agg(customers=("customer_id", "count"),
    churn_risk=("churn_prob", "mean"), clv_12m=("clv_12m", "mean"),
    uplift=("uplift", "mean")).round(3))

st.caption("Assumptions: 30% profit margin; response grows linearly with discount (capped at 2x the 15% "
           "reference); customers saved do not depend on how many months the discount lasts; negative "
           "predicted uplift is treated as zero. Uplift is simulated and calibrated to the held-out "
           "effect in the top 20%.")