import numpy as np
import pandas as pd
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"
STORES = ["CA_1", "CA_2", "TX_1", "TX_2"]
N_ITEMS = 150
LAST_DAYS = 730

sales = pd.read_csv(DATA / "sales_train_evaluation.csv")
sales = sales[sales.store_id.isin(STORES)]
day_cols = [c for c in sales.columns if c.startswith("d_")]

top_items = sales.groupby("item_id")[day_cols].sum().sum(axis=1).nlargest(N_ITEMS).index
sales = sales[sales.item_id.isin(top_items)]

id_cols = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
df = sales.melt(id_vars=id_cols, value_vars=day_cols[-LAST_DAYS:],
                var_name="d", value_name="sales")

cal = pd.read_csv(DATA / "calendar.csv", parse_dates=["date"])
df = df.merge(cal, on="d", how="left")
prices = pd.read_csv(DATA / "sell_prices.csv")
df = df.merge(prices, on=["store_id", "item_id", "wm_yr_wk"], how="left")
df = df.sort_values(["id", "date"]).reset_index(drop=True)

df["sell_price"] = df.groupby("id")["sell_price"].ffill()
df["sell_price"] = df.groupby("id")["sell_price"].bfill()
df["price_max_28"] = df.groupby("id")["sell_price"].transform(lambda s: s.rolling(28, min_periods=1).max())
df["promo"] = (df["sell_price"] < 0.95 * df["price_max_28"]).astype(int)

df["snap"] = np.select(
    [df.state_id == "CA", df.state_id == "TX"],
    [df.snap_CA, df.snap_TX], default=df.snap_WI)
df["is_holiday"] = df.event_name_1.notna().astype(int)

g = df.groupby("id")["sales"]
for lag in (7, 14, 28):
    df[f"lag_{lag}"] = g.shift(lag)
for w in (7, 28):
    df[f"roll_mean_{w}"] = g.transform(lambda s: s.shift(1).rolling(w).mean())

df = df.drop(columns=["event_name_2", "event_type_2", "snap_CA", "snap_TX",
                      "snap_WI", "price_max_28"])
df.to_parquet(DATA / "processed.parquet", index=False)
print(df.shape, df.date.min(), df.date.max())