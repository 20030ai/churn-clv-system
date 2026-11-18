import logging
import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path
from prophet import Prophet

logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
logging.getLogger("prophet").setLevel(logging.ERROR)

ROOT = Path(__file__).resolve().parent.parent
DATA, OUT = ROOT / "data", ROOT / "results"
OUT.mkdir(exist_ok=True)
H = 28  # forecast horizon (days)

df = pd.read_parquet(DATA / "processed.parquet").sort_values(["id", "date"]).reset_index(drop=True)
cut = df.date.max() - pd.Timedelta(days=H)

# features usable for a 28-day-ahead forecast (all lags >= 28, so no leakage)
g = df.groupby("id")["sales"]
df["lag_28"] = g.shift(28)
df["lag_35"] = g.shift(35)
df["lag_42"] = g.shift(42)
df["rm7_28"] = g.transform(lambda s: s.shift(28).rolling(7).mean())
df["rm28_28"] = g.transform(lambda s: s.shift(28).rolling(28).mean())
df["dow"] = df.date.dt.dayofweek
df["month"] = df.date.dt.month
df["dom"] = df.date.dt.day
for c in ["item_id", "dept_id", "cat_id", "store_id", "state_id"]:
    df[c + "_c"] = df[c].astype("category").cat.codes

feats = ["lag_28", "lag_35", "lag_42", "rm7_28", "rm28_28", "dow", "month", "dom",
         "sell_price", "promo", "snap", "is_holiday",
         "item_id_c", "dept_id_c", "cat_id_c", "store_id_c", "state_id_c"]

train = df[df.date <= cut].dropna(subset=feats)
test = df[df.date > cut].copy()

# 1) seasonal naive: same weekday from the last training week
last = df[df.date <= cut].groupby("id").tail(7)[["id", "dow", "sales"]].rename(columns={"sales": "naive"})
test = test.merge(last, on=["id", "dow"], how="left")

# 2) LightGBM point forecast + 80% prediction interval
base = dict(learning_rate=0.05, num_leaves=63, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.8, verbose=-1, n_jobs=-1)
point = lgb.LGBMRegressor(objective="tweedie", tweedie_variance_power=1.2, n_estimators=400, **base)
point.fit(train[feats], train.sales)
test["lgbm"] = np.clip(point.predict(test[feats]), 0, None)
for name, alpha in (("lgbm_p10", 0.1), ("lgbm_p90", 0.9)):
    q = lgb.LGBMRegressor(objective="quantile", alpha=alpha, n_estimators=200, **base)
    q.fit(train[feats], train.sales)
    test[name] = np.clip(q.predict(test[feats]), 0, None)
lo = np.minimum(test.lgbm_p10, test.lgbm)
hi = np.maximum(test.lgbm_p90, test.lgbm)
test["lgbm_p10"], test["lgbm_p90"] = lo, hi

# 3) Prophet at store level (4 series)
st_tr = df[df.date <= cut].groupby(["store_id", "date"], as_index=False).sales.sum()
rows = []
for s, d in st_tr.groupby("store_id"):
    m = Prophet(weekly_seasonality=True, yearly_seasonality=True, daily_seasonality=False)
    m.fit(d.rename(columns={"date": "ds", "sales": "y"})[["ds", "y"]])
    fc = m.predict(m.make_future_dataframe(periods=H)).tail(H)[["ds", "yhat"]]
    fc["store_id"] = s
    rows.append(fc)
prophet = pd.concat(rows).rename(columns={"ds": "date", "yhat": "prophet"})
prophet["prophet"] = prophet.prophet.clip(lower=0)

# metrics
wape = lambda y, p: np.abs(y - p).sum() / y.sum()
rmse = lambda y, p: float(np.sqrt(np.mean((y - p) ** 2)))

res = []
for n in ("naive", "lgbm"):
    res.append(("item-store", n, wape(test.sales, test[n]), rmse(test.sales, test[n])))
st = test.groupby(["store_id", "date"], as_index=False)[["sales", "naive", "lgbm"]].sum().merge(
    prophet, on=["store_id", "date"])
for n in ("naive", "prophet", "lgbm"):
    res.append(("store", n, wape(st.sales, st[n]), rmse(st.sales, st[n])))
metrics = pd.DataFrame(res, columns=["level", "model", "WAPE", "RMSE"])
cov = ((test.sales >= test.lgbm_p10) & (test.sales <= test.lgbm_p90)).mean()

print(metrics.round(4).to_string(index=False))
print(f"LightGBM 80% interval coverage: {cov:.1%}")

metrics.to_csv(OUT / "metrics.csv", index=False)
keep = ["id", "item_id", "store_id", "state_id", "date", "sales", "naive", "lgbm", "lgbm_p10", "lgbm_p90", "sell_price"]
test[keep].to_parquet(OUT / "forecasts.parquet", index=False)
prophet.to_parquet(OUT / "prophet_store.parquet", index=False)