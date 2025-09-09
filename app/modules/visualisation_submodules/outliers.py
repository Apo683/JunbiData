# app/modules/visualisation_parts/outliers.py
import dash
import dash_bootstrap_components as dbc
from dash import html, dcc, Input, Output, State
import pandas as pd
import numpy as np
import plotly.express as px

try:
    from pyspark.sql import functions as F
    from pyspark.sql.functions import col
    from pyspark.sql.types import (IntegerType, LongType, FloatType, DoubleType, ShortType, DecimalType, BooleanType, DateType, TimestampType)
    HAS_SPARK = True
except Exception:
    F = None
    col = None  # type: ignore
    HAS_SPARK = False

from .common import load_df, _get_common_layout, _generate_content, format_warning

NUMERIC_KINDS = ("i", "u", "f")  # int, uint, float
# constants en haut du module outliers.py (ou import depuis un common si tu préfères)
CHECKLIST_STYLE = {
    "display": "flex",
    "flexWrap": "wrap",
    "justifyContent": "flex-start",
}
CHECKLIST_INPUT_STYLE = {"marginRight": "6px"}
CHECKLIST_LABEL_STYLE = {
    "width": "230px",
    "textOverflow": "ellipsis",
    "overflow": "hidden",
    "whiteSpace": "nowrap",
    "display": "inline-block",
    "marginRight": "12px",
}


def _numeric_columns(df, is_spark):
    if is_spark:
        allowed = (IntegerType, LongType, FloatType, DoubleType, ShortType, DecimalType)
        excluded = (BooleanType, DateType, TimestampType)
        cols = []
        for f in df.schema.fields:
            if isinstance(f.dataType, allowed):
                cols.append(f.name)
            # on ignore explicitement bool/date/ts
        return cols
    else:
        cols = []
        for c in df.columns:
            dt = df[c].dtype
            if (pd.api.types.is_numeric_dtype(dt)
                and not pd.api.types.is_bool_dtype(dt)
                and not pd.api.types.is_datetime64_any_dtype(dt)):
                cols.append(c)
        return cols


def get_layout(STYLE_DROPDOWN, OPTIONS_DROPDOWN):
    return dbc.Tab(tab_id="outliers", label="Outliers", children=[
        html.H6("Vue globale des outliers par colonne :"),
        dbc.Row([
            dbc.Col(
                dcc.RadioItems(
                    id="outliers-method",
                    options=[
                        {"label": "IQR (Tukey)", "value": "iqr"},
                        {"label": "Z-score", "value": "z"},
                        {"label": "MAD", "value": "mad"},
                    ],
                    value="iqr",
                    inline=True,
                    style={"display": "flex", "flexWrap": "wrap", "gap": "8px 14px", "alignItems": "center"},
                    inputStyle={"marginRight": "6px"},
                    labelStyle={"marginRight": "0px", "padding": "2px 4px"}
                ), xs=12, md=6
            ),
        ]),
        dbc.Row([
            dbc.Col(
                dbc.InputGroup([
                    dbc.InputGroupText("k (IQR)"),
                    dbc.Input(id="outliers-iqr-k", type="number", value=1.5, step=0.1, min=0.1)
                ], className="me-2", style={"maxWidth": "220px", "width": "auto"}
                ), xs="auto",
            ),
            dbc.Col(
                dbc.InputGroup([
                    dbc.InputGroupText("Seuil |z|"),
                    dbc.Input(id="outliers-z-th", type="number", value=3.0, step=0.1, min=0.5)
                ], className="me-2", style={"maxWidth": "220px", "width": "auto"}
                ), xs="auto",
            ),
            dbc.Col(
                dbc.InputGroup([
                    dbc.InputGroupText("Seuil MAD"),
                    dbc.Input(id="outliers-mad-th", type="number", value=3.5, step=0.1, min=0.5)
                ], className="me-2", style={"maxWidth": "220px", "width": "auto"}
                ), xs="auto",
            ),
        ], className="g-2", style={"margin": 0, "marginBottom": "10px"}),
        dbc.Row([
            dbc.Col(
                dcc.Dropdown(
                    id="outliers-display-mode",
                    options=OPTIONS_DROPDOWN,
                    value="graph_descending",
                    style=STYLE_DROPDOWN,
                    clearable=False,
                ), xs=12, md=6
            )
        ], className="g-2", style={"margin": 0}),
        html.Div(id="outliers-overview-container", style={"marginBottom": "20px"}),

        html.H6("Détails par colonne :"),
        dbc.Row([
            dcc.Checklist(
                id="outliers-columns-checklist",
                options=[],
                value=[],
                inline=True,
                style=CHECKLIST_STYLE,
                inputStyle=CHECKLIST_INPUT_STYLE,
                labelStyle=CHECKLIST_LABEL_STYLE,
            )
        ], className="g-2", style={"margin": 0}),
        dbc.Row([
                dbc.Col(
                    dbc.InputGroup(
                        [
                            dbc.InputGroupText("Sample N"),
                            dbc.Input(
                                id="outliers-sample-n",
                                type="number",
                                value=5000,
                                min=100,
                                step=100,
                                style={"width": "90px"},  # compact
                            ),
                        ],
                        style={"maxWidth": "200px", "width": "auto"},  # un peu plus lisible
                        className="d-flex align-items-center",
                    ),
                    xs="auto",
                    className="d-flex align-items-center",
                ),
            ],
            className="g-2 align-items-end",  # petit gap + align bas pour un rendu propre
            style={"margin": 0, "marginBottom": "4px"},
        ),
        dbc.Row([
                dbc.Col(
                    dcc.Checklist(
                        id="outliers-log-scale",
                        options=[{"label": "Échelle log", "value": "log"}],
                        value=[],
                        inline=True,
                        style={"display": "flex", "alignItems": "center"},
                        inputStyle={"marginRight": "6px"},
                        labelStyle={"margin": 0},  # pas d'écart superflu
                    ),
                    xs="auto",
                    className="d-flex align-items-center",
                ),
            ],
            className="g-2 align-items-center",
            style={"margin": 0},
        ),
        html.Div(id="outliers-details-container")
    ])

def _approx_quantiles_spark(df, col_name, probs=(0.25, 0.5, 0.75)):
    # relError ~ 1e-3: bon compromis perf/précision
    return df.approxQuantile(col_name, list(probs), 1e-3)

def _overview_metrics(df, is_spark, method="iqr", iqr_k=1.5, z_th=3.0, mad_th=3.5):
    cols = _numeric_columns(df, is_spark)
    if not cols:
        return pd.DataFrame(columns=["Colonne","Nb outliers","% outliers"])
    total_rows = df.count() if is_spark else len(df)

    records = []
    if is_spark:
        for c in cols:
            # coercition + cast float pour éviter les bools déguisés
            s = pd.to_numeric(df[c], errors="coerce").astype("float64")
            s = s[~pd.isna(s)]
            if s.empty or s.nunique(dropna=True) <= 1:
                records.append((c, 0, 0.0))
                continue
            if method == "iqr":
                q1, q3 = s.quantile(0.25), s.quantile(0.75)
                iqr = float(q3 - q1)
                if iqr <= 0:
                    out = 0
                else:
                    low, high = q1 - iqr_k * iqr, q3 + iqr_k * iqr
                    out = int(((s < low) | (s > high)).sum())
            elif method == "z":
                m, st = float(s.mean()), float(s.std(ddof=0))
                out = 0 if st <= 0 else int((np.abs((s - m) / st) > z_th).sum())
            else:  # MAD
                med = float(s.median())
                mad = float(np.median(np.abs(s - med)))
                scale = 1.4826 * mad
                out = 0 if scale <= 0 else int((np.abs(s - med) / scale > mad_th).sum())
            pct = round(100.0 * out / max(1, len(df)), 2)
            records.append((c, out, pct))
    else:
        for c in cols:
            s = pd.to_numeric(df[c], errors="coerce")
            s = s[~s.isna()]
            if s.empty:
                records.append((c, 0, 0.0)); continue
            if method == "iqr":
                q1, q3 = s.quantile(0.25), s.quantile(0.75)
                iqr = q3 - q1
                if iqr == 0:
                    out = 0
                else:
                    low, high = q1 - iqr_k*iqr, q3 + iqr_k*iqr
                    out = int(((s < low) | (s > high)).sum())
            elif method == "z":
                m, st = s.mean(), s.std(ddof=0)
                if st == 0:
                    out = 0
                else:
                    out = int((np.abs((s - m) / st) > z_th).sum())
            else:
                med = s.median()
                mad = np.median(np.abs(s - med))
                if mad == 0:
                    out = 0
                else:
                    scale = 1.4826 * mad
                    out = int((np.abs(s - med) / scale > mad_th).sum())
            pct = round(100.0 * out / max(1, len(df)), 2)
            records.append((c, out, pct))

    out_df = pd.DataFrame(records, columns=["Colonne","Nb outliers","% outliers"])
    return out_df

def _sample_for_plot(df, col_name, is_spark, sample_n=5000):
    if is_spark:
        # Prend jusqu’à sample_n non nuls
        # Option 1: fraction approx
        total = df.count()
        frac = min(1.0, max(0.0001, float(sample_n) / max(1, total)))
        sdf = df.select(col(col_name).alias(col_name)).sample(False, frac, seed=42)
        # Garde une limite dure au cas où
        sdf = sdf.limit(sample_n)
        return sdf.toPandas()
    else:
        s = df[col_name]
        if len(s) > sample_n:
            s = s.sample(n=sample_n, random_state=42)
        return pd.DataFrame({col_name: s})

def register_callbacks(app):
    # Alimente la checklist des colonnes numériques
    @app.callback(
        Output("outliers-columns-checklist", "options"),
        Input("parquet-path-store", "data")
    )
    def fill_columns(parquet_path):
        df, is_spark = load_df(parquet_path)
        if df is None:
            return []
        cols = _numeric_columns(df, is_spark)
        return [{"label": c, "value": c} for c in cols]

    # Vue globale
    @app.callback(
        Output("outliers-overview-container", "children"),
        Input("parquet-path-store", "data"),
        Input("outliers-method", "value"),
        Input("outliers-iqr-k", "value"),
        Input("outliers-z-th", "value"),
        Input("outliers-mad-th", "value"),
        Input("outliers-display-mode", "value")
    )
    def update_overview(parquet_path, method, iqr_k, z_th, mad_th, display_mode):
        df, is_spark = load_df(parquet_path)
        if df is None:
            return format_warning("Aucun dataset chargé.")
        metrics = _overview_metrics(df, is_spark, method or "iqr", float(iqr_k or 1.5), float(z_th or 3.0), float(mad_th or 3.5))

        if metrics.empty:
            return format_warning("Aucune colonne numérique détectée.")

        # Graph nb outliers
        layout_n = _get_common_layout("Nombre d’outliers par colonne", "Colonne", "Nb outliers", height=400)
        layout_n["xaxis_showticklabels"] = False
        graph_n = _generate_content(
            metrics, display_mode or "graph_descending",
            "Colonne", "Nb outliers",
            "Nombre d’outliers par colonne",
            "Colonne", "Nb outliers",
            sort_key="Nb outliers",
            height=400, custom_layout=layout_n
        )
        # Graph % outliers
        layout_p = _get_common_layout("Pourcentage d’outliers par colonne", "Colonne", "% outliers", height=400)
        layout_p["xaxis_showticklabels"] = False
        graph_p = _generate_content(
            metrics, display_mode or "graph_descending",
            "Colonne", "% outliers",
            "Pourcentage d’outliers par colonne",
            "Colonne", "% outliers",
            sort_key="% outliers",
            height=400, custom_layout=layout_p
        )

        return html.Div([
            dbc.Row(
                [
                    dbc.Col(graph_n, xs=12, md=6, className="mb-3 mb-md-0"),
                    dbc.Col(graph_p, xs=12, md=6),
                ],
                className="g-2 flex-wrap",
                style={"margin": 0}
            )
        ])

    # Détails (plots par colonne sélectionnée)
    @app.callback(
        Output("outliers-details-container", "children"),
        Input("parquet-path-store", "data"),
        Input("outliers-columns-checklist", "value"),
        Input("outliers-method", "value"),
        Input("outliers-iqr-k", "value"),
        Input("outliers-z-th", "value"),
        Input("outliers-mad-th", "value"),
        Input("outliers-sample-n", "value"),
        Input("outliers-log-scale", "value")
    )
    def update_details(parquet_path, selected_cols, method, iqr_k, z_th, mad_th, sample_n, log_scale):
        df, is_spark = load_df(parquet_path)
        if df is None:
            return format_warning("Aucun dataset chargé.")
        if not selected_cols:
            return html.Div("Aucune colonne sélectionnée.")

        iqr_k = float(iqr_k or 1.5); z_th = float(z_th or 3.0); mad_th = float(mad_th or 3.5)
        sample_n = int(sample_n or 5000)
        out_graphs = []

        for c in selected_cols:
            pdf = _sample_for_plot(df, c, is_spark, sample_n=sample_n)
            s = pd.to_numeric(pdf[c], errors="coerce").dropna()
            if s.empty:
                out_graphs.append(html.Div(f"{c}: pas de données numériques à tracer.")); continue

            # Seuils selon méthode
            if method == "iqr":
                q1, q3 = s.quantile(0.25), s.quantile(0.75)
                iqr = q3 - q1
                low, high = (q1 - iqr_k*iqr, q3 + iqr_k*iqr) if iqr > 0 else (None, None)
                title_suffix = f"(IQR k={iqr_k})"
            elif method == "z":
                m, st = s.mean(), s.std(ddof=0)
                low, high = (m - z_th*st, m + z_th*st) if st > 0 else (None, None)
                title_suffix = f"(Z |z|>{z_th})"
            else:
                med = s.median()
                mad = np.median(np.abs(s - med))
                scale = 1.4826 * mad
                low, high = (med - mad_th*scale, med + mad_th*scale) if scale > 0 else (None, None)
                title_suffix = f"(MAD>{mad_th})"

            # Boxplot
            fig_box = px.box(pdf, y=c, points="outliers", title=f"{c} — Boxplot {title_suffix}", height=320)
            layout_box = _get_common_layout(fig_box.layout.title.text, "", c, height=320)
            if log_scale: layout_box["yaxis_type"] = "log"
            fig_box.update_layout(**layout_box)

            # Histogramme avec zones outliers
            fig_hist = px.histogram(pdf, x=c, nbins=50, title=f"{c} — Histogramme {title_suffix}", height=320)
            layout_hist = _get_common_layout(fig_hist.layout.title.text, c, "Fréquence", height=320)
            if log_scale: layout_hist["xaxis_type"] = "log"
            fig_hist.update_layout(**layout_hist)
            if low is not None and high is not None:
                fig_hist.add_vrect(x0=min(s.min(), low), x1=low, fillcolor="tomato", opacity=0.12, line_width=0)
                fig_hist.add_vrect(x0=high, x1=max(s.max(), high), fillcolor="tomato", opacity=0.12, line_width=0)
                fig_hist.add_vline(x=low, line=dict(color="tomato", width=1))
                fig_hist.add_vline(x=high, line=dict(color="tomato", width=1))

            out_graphs.append(
                dbc.Row(
                    [
                        dbc.Col(dcc.Graph(figure=fig_box), xs=12, md=6, className="mb-3 mb-md-0"),
                        dbc.Col(dcc.Graph(figure=fig_hist), xs=12, md=6),
                    ],
                    className="g-2 flex-wrap",
                    style={"margin": 0, "marginBottom": "12px"}
                )
            )

        return html.Div(out_graphs)