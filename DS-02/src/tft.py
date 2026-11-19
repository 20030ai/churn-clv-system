import warnings
import numpy as np
import pandas as pd
from pathlib import Path
import lightning.pytorch as pl
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
DATA, OUT = ROOT / "data", ROOT / "results"
H, ENC, N_SERIES, DAYS = 28, 56, 60, 365

df = pd.read_parquet(DATA / "processed.parquet")
top = df.groupby("id").sales.sum().nlargest(N_SERIES).index
df = df[df.id.isin(top)]
df = df[df.date > df.date.max() - pd.Timedelta(days=DAYS)].sort_values(["id", "date"]).copy()
df["time_idx"] = (df.date - df.date.min()).dt.days
df["sales"] = df.sales.astype(float)
df["sell_price"] = df.sell_price.fillna(0).astype(float)
for c in ["snap", "promo", "is_holiday"]:
    df[c] = df[c].astype(float)
for c in ["id", "store_id", "item_id"]:
    df[c] = df[c].astype(str)
df["dow"] = df.date.dt.dayofweek.astype(str)
df["month"] = df.date.dt.month.astype(str)
df = df[["id", "store_id", "item_id", "date", "time_idx", "sales", "sell_price",
         "snap", "promo", "is_holiday", "dow", "month"]]

cutoff = df.time_idx.max() - H
training = TimeSeriesDataSet(
    df[df.time_idx <= cutoff], time_idx="time_idx", target="sales", group_ids=["id"],
    max_encoder_length=ENC, max_prediction_length=H,
    static_categoricals=["store_id", "item_id"],
    time_varying_known_categoricals=["dow", "month"],
    time_varying_known_reals=["time_idx", "sell_price", "snap", "promo", "is_holiday"],
    time_varying_unknown_reals=["sales"],
    target_normalizer=GroupNormalizer(groups=["id"], transformation="softplus"),
    add_relative_time_idx=True, add_target_scales=True, add_encoder_length=True,
    allow_missing_timesteps=True)
validation = TimeSeriesDataSet.from_dataset(training, df, predict=True, stop_randomization=True)
train_dl = training.to_dataloader(train=True, batch_size=64, num_workers=0)
val_dl = validation.to_dataloader(train=False, batch_size=128, num_workers=0)

tft = TemporalFusionTransformer.from_dataset(
    training, learning_rate=0.03, hidden_size=16, attention_head_size=2, dropout=0.1,
    hidden_continuous_size=8, loss=QuantileLoss(quantiles=[0.1, 0.5, 0.9]), optimizer="adam")

trainer = pl.Trainer(max_epochs=5, accelerator="cpu", gradient_clip_val=0.1,
                     limit_train_batches=60, logger=False, enable_checkpointing=False)
trainer.fit(tft, train_dl, val_dl)

res = tft.predict(val_dl, mode="quantiles", return_index=True,
                  trainer_kwargs=dict(accelerator="cpu"))
q = res.output.cpu().numpy()
idx = res.index.reset_index(drop=True)
tmap = df.drop_duplicates("time_idx").set_index("time_idx").date

rows = []
for i, (sid, t0) in enumerate(zip(idx["id"], idx["time_idx"])):
    for h in range(H):
        rows.append((sid, tmap[t0 + h], q[i, h, 0], q[i, h, 1], q[i, h, 2]))
pred = pd.DataFrame(rows, columns=["id", "date", "tft_p10", "tft_p50", "tft_p90"])
for c in ["tft_p10", "tft_p50", "tft_p90"]:
    pred[c] = pred[c].clip(lower=0)

fc = pd.read_parquet(OUT / "forecasts.parquet")
cmp = fc.merge(pred, on=["id", "date"])
wape = lambda y, p: np.abs(y - p).sum() / y.sum()
rmse = lambda y, p: float(np.sqrt(np.mean((y - p) ** 2)))
res_rows = [("item-store (60 series)", "lgbm", wape(cmp.sales, cmp.lgbm), rmse(cmp.sales, cmp.lgbm)),
            ("item-store (60 series)", "tft", wape(cmp.sales, cmp.tft_p50), rmse(cmp.sales, cmp.tft_p50))]
m = pd.DataFrame(res_rows, columns=["level", "model", "WAPE", "RMSE"])
cov = ((cmp.sales >= cmp.tft_p10) & (cmp.sales <= cmp.tft_p90)).mean()
print(m.round(4).to_string(index=False))
print(f"TFT 80% interval coverage: {cov:.1%}")

old = pd.read_csv(OUT / "metrics.csv")
old = old[~old.level.str.contains("60 series")]
pd.concat([old, m]).to_csv(OUT / "metrics.csv", index=False)
pred.to_parquet(OUT / "tft_forecasts.parquet", index=False)