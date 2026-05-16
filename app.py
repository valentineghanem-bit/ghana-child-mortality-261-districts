#!/usr/bin/env python3
"""
Subnational Child Mortality — Ghana 261 Districts
Interactive Dash dashboard: spatial distribution, ML predictors, district explorer.
Run: python app.py  →  http://127.0.0.1:8050
"""
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import dash
from dash import dcc, html, Input, Output, dash_table
import dash_bootstrap_components as dbc

DATA = os.path.join(os.path.dirname(__file__), "data",
                    "Ghana_ChildMortality_261District_MasterDataset.csv")
df = pd.read_csv(DATA)

METRICS = {
    "u5mr_per1000lb": "Under-5 Mortality Rate (per 1,000 LB)",
    "nmr_per1000lb":  "Neonatal Mortality Rate (per 1,000 LB)",
    "imr_per1000lb":  "Infant Mortality Rate (per 1,000 LB)",
}
PREDICTORS = {
    "early_breastfeeding_pct":  "Early Breastfeeding (%)",
    "diarrhea_prevalence_pct":  "Diarrhoea Prevalence (%)",
    "fully_vaccinated_pct":     "Full Vaccination Coverage (%)",
    "improved_water_pct":       "Improved Water Access (%)",
    "poverty_rate_pct":         "Poverty Rate (%)",
    "illiteracy_rate_pct":      "Illiteracy Rate (%)",
    "open_defecation_pct":      "Open Defecation (%)",
}

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY],
                title="Ghana Child Mortality — 261 Districts")
server = app.server

def metric_card(label, value, color="primary"):
    return dbc.Card(dbc.CardBody([
        html.P(label, className="text-muted mb-1", style={"fontSize": "0.75rem"}),
        html.H5(value, className=f"text-{color} mb-0"),
    ]), className="mb-2")

app.layout = dbc.Container(fluid=True, children=[
    dbc.Row(dbc.Col(html.H4(
        "Subnational Distribution & Ensemble ML Predictors of Child Mortality — Ghana 261 Districts",
        className="text-center text-light py-3 mb-0"))),

    # Summary cards
    dbc.Row([
        dbc.Col(metric_card("Districts", "261", "info"), md=2),
        dbc.Col(metric_card("U5MR Range", f"{df.u5mr_per1000lb.min():.0f}–{df.u5mr_per1000lb.max():.0f} /1,000 LB", "warning"), md=3),
        dbc.Col(metric_card("High-Risk Districts", f"{df.u5mr_high_risk.sum()} (≥48/1,000 LB)", "danger"), md=3),
        dbc.Col(metric_card("Global Moran's I", "0.778 (p<0.001)", "success"), md=2),
        dbc.Col(metric_card("Stacked Ensemble AUC", "1.00 (ecological)", "secondary"), md=2),
    ], className="mb-3"),

    dbc.Tabs([
        # ── Tab 1: Spatial Distribution ──────────────────────────────────────
        dbc.Tab(label="Spatial Distribution", children=[
            dbc.Row([
                dbc.Col([
                    html.Label("Outcome metric:", className="text-light mt-3"),
                    dcc.Dropdown(id="map-metric",
                                 options=[{"label": v, "value": k} for k, v in METRICS.items()],
                                 value="u5mr_per1000lb", clearable=False,
                                 style={"color": "#000"}),
                ], md=4),
            ]),
            dbc.Row([
                dbc.Col(dcc.Graph(id="scatter-map"), md=8),
                dbc.Col(dcc.Graph(id="region-box"), md=4),
            ]),
        ]),

        # ── Tab 2: Predictor Analysis ─────────────────────────────────────
        dbc.Tab(label="Predictor Analysis", children=[
            dbc.Row([
                dbc.Col([
                    html.Label("X-axis predictor:", className="text-light mt-3"),
                    dcc.Dropdown(id="pred-x",
                                 options=[{"label": v, "value": k} for k, v in PREDICTORS.items()],
                                 value="early_breastfeeding_pct", clearable=False,
                                 style={"color": "#000"}),
                ], md=4),
                dbc.Col([
                    html.Label("Outcome:", className="text-light mt-3"),
                    dcc.Dropdown(id="pred-y",
                                 options=[{"label": v, "value": k} for k, v in METRICS.items()],
                                 value="u5mr_per1000lb", clearable=False,
                                 style={"color": "#000"}),
                ], md=4),
            ]),
            dbc.Row([
                dbc.Col(dcc.Graph(id="scatter-pred"), md=8),
                dbc.Col(dcc.Graph(id="gini-bar"), md=4),
            ]),
        ]),

        # ── Tab 3: District Explorer ─────────────────────────────────────
        dbc.Tab(label="District Explorer", children=[
            dbc.Row(dbc.Col([
                html.Label("Filter by Region:", className="text-light mt-3"),
                dcc.Dropdown(id="region-filter",
                             options=[{"label": r, "value": r} for r in sorted(df.region.unique())],
                             multi=True, placeholder="All regions",
                             style={"color": "#000"}),
            ], md=5)),
            dbc.Row(dbc.Col(dash_table.DataTable(
                id="district-table",
                columns=[
                    {"name": "District", "id": "district"},
                    {"name": "Region",   "id": "region"},
                    {"name": "U5MR",     "id": "u5mr_per1000lb"},
                    {"name": "NMR",      "id": "nmr_per1000lb"},
                    {"name": "Poverty %","id": "poverty_rate_pct"},
                    {"name": "Vacc. %",  "id": "fully_vaccinated_pct"},
                    {"name": "High-Risk","id": "u5mr_high_risk"},
                ],
                page_size=20, sort_action="native", filter_action="native",
                style_table={"overflowX": "auto"},
                style_header={"backgroundColor": "#2c3e50", "color": "white"},
                style_data={"backgroundColor": "#1a1a2e", "color": "white"},
                style_data_conditional=[{
                    "if": {"filter_query": "{u5mr_high_risk} = 1"},
                    "color": "#e74c3c", "fontWeight": "bold"
                }],
            ))),
        ]),
    ]),
], style={"backgroundColor": "#111827", "minHeight": "100vh"})


@app.callback(Output("scatter-map", "figure"), Output("region-box", "figure"),
              Input("map-metric", "value"))
def update_map(metric):
    label = METRICS[metric]
    fig_map = px.scatter(df, x="longitude", y="latitude", color=metric,
                         size="total_pop", hover_name="district",
                         hover_data={"region": True, metric: ":.1f",
                                     "longitude": False, "latitude": False},
                         color_continuous_scale="RdYlGn_r", size_max=18,
                         title=f"{label} by District",
                         labels={metric: label})
    fig_map.update_layout(paper_bgcolor="#1a1a2e", plot_bgcolor="#1a1a2e",
                          font_color="white", margin=dict(t=40, b=10))

    fig_box = px.box(df, x="region", y=metric, color="region",
                     title=f"{label} by Region", labels={metric: label, "region": ""},
                     color_discrete_sequence=px.colors.qualitative.Set2)
    fig_box.update_xaxes(tickangle=45)
    fig_box.update_layout(paper_bgcolor="#1a1a2e", plot_bgcolor="#1a1a2e",
                          font_color="white", showlegend=False,
                          margin=dict(t=40, b=80))
    return fig_map, fig_box


@app.callback(Output("scatter-pred", "figure"), Output("gini-bar", "figure"),
              Input("pred-x", "value"), Input("pred-y", "value"))
def update_predictor(px_col, py_col):
    fig = px.scatter(df, x=px_col, y=py_col, color="region",
                     hover_name="district", trendline="ols",
                     title=f"{PREDICTORS[px_col]} vs {METRICS[py_col]}",
                     labels={px_col: PREDICTORS[px_col], py_col: METRICS[py_col]},
                     color_discrete_sequence=px.colors.qualitative.Vivid)
    fig.update_layout(paper_bgcolor="#1a1a2e", plot_bgcolor="#1a1a2e",
                      font_color="white", margin=dict(t=40, b=10))

    gini_data = {k: v for k, v in PREDICTORS.items() if k in df.columns}
    importance = {"early_breastfeeding_pct": 0.224, "diarrhea_prevalence_pct": 0.182,
                  "fully_vaccinated_pct": 0.136, "poverty_rate_pct": 0.098,
                  "improved_water_pct": 0.087, "illiteracy_rate_pct": 0.071,
                  "open_defecation_pct": 0.063}
    gimp = pd.DataFrame({"feature": [PREDICTORS[k] for k in importance],
                          "importance": list(importance.values())}).sort_values("importance")
    fig_imp = px.bar(gimp, x="importance", y="feature", orientation="h",
                     title="RF Gini Feature Importance",
                     color="importance", color_continuous_scale="Blues")
    fig_imp.update_layout(paper_bgcolor="#1a1a2e", plot_bgcolor="#1a1a2e",
                          font_color="white", showlegend=False,
                          margin=dict(t=40, l=10))
    return fig, fig_imp


@app.callback(Output("district-table", "data"),
              Input("region-filter", "value"))
def update_table(regions):
    d = df if not regions else df[df.region.isin(regions)]
    return d[["district", "region", "u5mr_per1000lb", "nmr_per1000lb",
              "poverty_rate_pct", "fully_vaccinated_pct", "u5mr_high_risk"]].to_dict("records")


if __name__ == "__main__":
    print("Dashboard: http://127.0.0.1:8050")
    app.run(debug=False, port=8050)
