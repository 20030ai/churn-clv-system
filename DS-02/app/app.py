import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
from dash import Dash, dcc, html, dash_table, Input, Output

R = Path(__file__).resolve().parent.parent / "results"
fc = pd.read_parquet(R / "forecasts_reconciled.parquet")
orders = pd.read_csv(R / "orders.csv")
metrics = pd.read_csv(R / "metrics.csv").round(4)
stores = sorted(fc.store_id.unique())

app = Dash(__name__)
app.title = "DS-02 Demand Forecasting"
app.layout = html.Div(style={"maxWidth": "1100px", "margin": "auto", "fontFamily": "sans-serif"}, children=[
    html.H2("Nationwide Demand Forecasting and Inventory Optimization"),
    html.Div(style={"display": "flex", "gap": "16px"}, children=[
        dcc.Dropdown(stores, stores[0], id="store", clearable=False, style={"width": "200px"}),
        dcc.Dropdown(id="item", clearable=False, style={"width": "300px"}),
    ]),
    dcc.Graph(id="chart"),
    html.Div(id="kpi", style={"margin": "8px 0", "fontWeight": "bold"}),
    html.H3("Model accuracy (last 28 days)"),
    dash_table.DataTable(metrics.to_dict("records"), [{"name": c, "id": c} for c in metrics.columns]),
    html.H3("Order recommendations"),
    dash_table.DataTable(id="orders", page_size=10, sort_action="native"),
])

@app.callback(Output("item", "options"), Output("item", "value"), Input("store", "value"))
def items(store):
    it = sorted(fc[fc.store_id == store].item_id.unique())
    return it, it[0]

@app.callback(Output("chart", "figure"), Output("kpi", "children"),
              Output("orders", "data"), Output("orders", "columns"),
              Input("store", "value"), Input("item", "value"))
def update(store, item):
    d = fc[(fc.store_id == store) & (fc.item_id == item)].sort_values("date")
    f = go.Figure()
    f.add_scatter(x=d.date, y=d.rec_p90, line=dict(width=0), showlegend=False)
    f.add_scatter(x=d.date, y=d.rec_p10, fill="tonexty", line=dict(width=0),
                  fillcolor="rgba(0,120,255,0.2)", name="80% interval")
    f.add_scatter(x=d.date, y=d.sales, name="Actual", line=dict(color="black"))
    f.add_scatter(x=d.date, y=d.rec, name="Forecast (reconciled)", line=dict(color="blue"))
    f.update_layout(title=f"{item} at {store}", height=380, margin=dict(t=40, b=20))
    o = orders[orders.store_id == store].sort_values("order_cost", ascending=False)
    row = o[o.item_id == item].iloc[0]
    kpi = (f"Recommended order for {item}: {int(row.order_qty)} units "
           f"(28-day forecast {row.forecast_28d:.0f}, on hand {row.on_hand:.0f})")
    return f, kpi, o.to_dict("records"), [{"name": c, "id": c} for c in o.columns]

if __name__ == "__main__":
    app.run(debug=False)