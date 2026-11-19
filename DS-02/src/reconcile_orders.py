import numpy as np
import pandas as pd
from pathlib import Path
from darts import TimeSeries
from darts.dataprocessing.transformers import MinTReconciliator
from ortools.linear_solver import pywraplp

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results"
stores = ["CA_1", "CA_2", "TX_1", "TX_2"]

fc = pd.read_parquet(OUT / "forecasts.parquet")
pro = pd.read_parquet(OUT / "prophet_store.parquet")

# ---------- 1) hierarchical reconciliation ----------
store_lgb = fc.groupby(["date", "store_id"]).lgbm.sum().unstack()[stores]
state_lgb = fc.groupby(["date", "state_id"]).lgbm.sum().unstack()[["CA", "TX"]]
total_lgb = fc.groupby("date").lgbm.sum().rename("total")
store_pro = pro.pivot(index="date", columns="store_id", values="prophet")[stores]

base = pd.concat([store_pro, state_lgb, total_lgb], axis=1)
base.columns.name = None
hier = {"CA_1": ["CA"], "CA_2": ["CA"], "TX_1": ["TX"], "TX_2": ["TX"], "CA": ["total"], "TX": ["total"]}
ts = TimeSeries.from_dataframe(base, freq="D", hierarchy=hier)
rec = MinTReconciliator(method="wls_struct")
rec.fit(ts)
out = rec.transform(ts).to_dataframe().clip(lower=0)

act_store = fc.groupby(["date", "store_id"]).sales.sum().unstack()[stores]
act_state = fc.groupby(["date", "state_id"]).sales.sum().unstack()[["CA", "TX"]]
wape = lambda y, p: float(np.abs(y.values - p.values).sum() / y.values.sum())
print("Store WAPE  before:", round(wape(act_store, base[stores]), 4), " after:", round(wape(act_store, out[stores]), 4))
print("State WAPE  before:", round(wape(act_state, base[["CA", "TX"]]), 4), " after:", round(wape(act_state, out[["CA", "TX"]]), 4))
gap_b = (base[["CA_1", "CA_2"]].sum(axis=1) - base["CA"]).abs().max()
gap_a = (out[["CA_1", "CA_2"]].sum(axis=1) - out["CA"]).abs().max()
print(f"Coherence gap (CA stores vs CA state) before: {gap_b:.2f}  after: {gap_a:.6f}")
out.to_csv(OUT / "reconciled_hierarchy.csv")

# push reconciled store totals down to item level
ratio = (out[stores] / store_lgb).clip(0.5, 1.5).stack().rename("ratio").reset_index()
ratio.columns = ["date", "store_id", "ratio"]
fc = fc.merge(ratio, on=["date", "store_id"], how="left")
for c, n in (("lgbm", "rec"), ("lgbm_p10", "rec_p10"), ("lgbm_p90", "rec_p90")):
    fc[n] = fc[c] * fc.ratio
print("Item-level WAPE  lgbm:", round(np.abs(fc.sales - fc.lgbm).sum() / fc.sales.sum(), 4),
      " reconciled:", round(np.abs(fc.sales - fc.rec).sum() / fc.sales.sum(), 4))
fc.to_parquet(OUT / "forecasts_reconciled.parquet", index=False)

# ---------- 2) OR-Tools order optimization (per store, 28-day horizon) ----------
agg = fc.groupby(["store_id", "item_id"]).agg(
    mid=("rec", "sum"), lo=("rec_p10", "sum"), hi=("rec_p90", "sum"),
    price=("sell_price", "last")).reset_index()
agg["price"] = agg.price.fillna(agg.price.median())
agg["lo"] = np.minimum(agg.lo, agg.mid)
agg["hi"] = np.maximum(agg.hi, agg.mid)
agg["on_hand"] = np.round(agg.mid * 3 / 28)          # assumption: 3 days of stock on hand

SCEN = (("lo", 0.25), ("mid", 0.50), ("hi", 0.25))   # demand scenarios and weights
orders = []
for s, g in agg.groupby("store_id"):
    solver = pywraplp.Solver.CreateSolver("GLOP")
    inf = solver.infinity()
    q, obj = {}, solver.Objective()
    for i, r in g.iterrows():
        q[i] = solver.NumVar(0, inf, f"q{i}")
        cost = 0.7 * r.price
        hold, short = 0.10 * cost, r.price - cost    # holding cost / lost margin per unit
        for name, w in SCEN:
            over, under = solver.NumVar(0, inf, ""), solver.NumVar(0, inf, "")
            solver.Add(r.on_hand + q[i] - r[name] == over - under)
            obj.SetCoefficient(over, w * hold)
            obj.SetCoefficient(under, w * short)
    cap = 1.05 * float((g.mid - g.on_hand).clip(lower=0).sum())   # store capacity limit
    solver.Add(solver.Sum(q.values()) <= cap)
    obj.SetMinimization()
    status = solver.Solve()
    assert status == pywraplp.Solver.OPTIMAL, f"solver failed for {s}"
    for i, r in g.iterrows():
        orders.append((s, r.item_id, round(r.mid, 1), r.on_hand,
                       int(np.ceil(q[i].solution_value() - 1e-6)), round(r.price, 2)))

orders = pd.DataFrame(orders, columns=["store_id", "item_id", "forecast_28d", "on_hand", "order_qty", "unit_price"])
orders["order_cost"] = (orders.order_qty * orders.unit_price * 0.7).round(2)
orders.to_csv(OUT / "orders.csv", index=False)
print(orders.groupby("store_id")[["forecast_28d", "order_qty", "order_cost"]].sum().round(0))
print(orders.head())