# app/modules/visualisation_parts/outliers.py
import math
import dash
import dash_bootstrap_components as dbc
from dash import html, dcc, Input, Output, State, no_update
import pandas as pd
import numpy as np
import plotly.express as px

try:
    from pyspark.sql import functions as F
    from pyspark.sql.functions import col
    from pyspark.sql.types import (
        IntegerType, LongType, FloatType, DoubleType, ShortType, DecimalType,
        BooleanType, DateType, TimestampType
    )
    HAS_SPARK = True
except Exception:
    F = None
    col = None  # type: ignore
    HAS_SPARK = False

from app.modules.common.io import load_df, format_warning
from app.modules.common.viz import _get_common_layout, _generate_content
# from ..common.io import load_df, _get_common_layout, _generate_content, format_warning

# ------------------------------
# Constantes UI
# ------------------------------
CHECKLIST_STYLE = {"display": "flex", "flexWrap": "wrap", "justifyContent": "flex-start"}
CHECKLIST_INPUT_STYLE = {"marginRight": "6px"}
CHECKLIST_LABEL_STYLE = {
    "width": "230px",
    "textOverflow": "ellipsis",
    "overflow": "hidden",
    "whiteSpace": "nowrap",
    "display": "inline-block",
    "marginRight": "12px",
}

# ------------------------------
# Aides colonnes numériques
# ------------------------------
def _numeric_columns(df, is_spark):
    if is_spark:
        allowed = (IntegerType, LongType, FloatType, DoubleType, ShortType, DecimalType)
        cols = [f.name for f in df.schema.fields if isinstance(f.dataType, allowed)]
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

# ------------------------------
# Aides statistiques Spark
# ------------------------------
def _approx_quantiles_spark(df, col_name, probs=(0.25, 0.5, 0.75), rel_error=1e-3):
    # percentiles approx scalables
    return df.approxQuantile(col_name, list(probs), rel_error)

def _spark_non_null_count(df, c):
    return df.where(col(c).isNotNull()).count()

def _spark_distinct_non_null_leq1(df, c):
    # Si <=1 valeur non nulle distincte → pas d'outliers
    row = df.select(F.approx_count_distinct(F.when(col(c).isNotNull(), col(c))).alias("d")).collect()[0]
    return (row["d"] or 0) <= 1

def _spark_mean_std(df, c):
    row = df.select(
        F.mean(col(c).cast("double")).alias("mu"),
        F.stddev(col(c).cast("double")).alias("sigma")
    ).collect()[0]
    mu = float(row["mu"]) if row["mu"] is not None else float("nan")
    sigma = float(row["sigma"]) if row["sigma"] is not None else float("nan")
    return mu, sigma

def _spark_median(df, c, rel_error=1e-3):
    med = df.approxQuantile(c, [0.5], rel_error)[0]
    return float(med) if med is not None else float("nan")

def _spark_mad(df, c, med, rel_error=1e-3):
    # MAD = median(|x - median|)
    x = F.abs(col(c).cast("double") - F.lit(float(med)))
    mad = df.select(x.alias("d")).approxQuantile("d", [0.5], rel_error)[0]
    return float(mad) if mad is not None else float("nan")

def _spark_iqr_bounds(df, c, k=1.5, rel_error=1e-3):
    q1, q3 = _approx_quantiles_spark(df.select(col(c).cast("double").alias(c)), c, (0.25, 0.75), rel_error)
    iqr = float(q3 - q1)
    if iqr <= 0:
        return None, None, 0.0, 0.0
    low, high = float(q1 - k * iqr), float(q3 + k * iqr)
    return low, high, float(q1), float(q3)

def _spark_outlier_counts(df, c, method, iqr_k=1.5, z_th=3.0, mad_th=3.5, rel_error=1e-3):
    """
    Retourne:
      count_out, pct_out, extras (dict avec quelques stats utiles pour l’affichage)
    Ne compte que sur les valeurs non nulles.
    """
    c_d = col(c).cast("double")
    base = df.where(c_d.isNotNull())
    n = base.count()
    if n == 0 or _spark_distinct_non_null_leq1(df, c):
        return 0, 0.0, {"q1": None, "q3": None, "median": None, "mean": None, "std": None, "mad": None}

    if method == "iqr":
        low, high, q1, q3 = _spark_iqr_bounds(df, c, k=iqr_k, rel_error=rel_error)
        if low is None:
            return 0, 0.0, {"q1": q1, "q3": q3, "median": None, "mean": None, "std": None, "mad": None}
        out = base.where((c_d < F.lit(low)) | (c_d > F.lit(high))).count()
        pct = round(100.0 * out / n, 2)
        return int(out), pct, {"q1": q1, "q3": q3, "median": None, "mean": None, "std": None, "mad": None}

    elif method == "z":
        mu, sigma = _spark_mean_std(df, c)
        if not (sigma and sigma > 0):
            return 0, 0.0, {"q1": None, "q3": None, "median": None, "mean": mu, "std": sigma, "mad": None}
        z = F.abs((c_d - F.lit(mu)) / F.lit(sigma))
        out = base.where(z > F.lit(float(z_th))).count()
        pct = round(100.0 * out / n, 2)
        return int(out), pct, {"q1": None, "q3": None, "median": None, "mean": mu, "std": sigma, "mad": None}

    else:  # MAD
        med = _spark_median(df, c, rel_error=rel_error)
        mad = _spark_mad(df, c, med, rel_error=rel_error)
        # échelle robuste (consistance gaussienne)
        scale = 1.4826 * mad if mad is not None else None
        if not (scale and scale > 0):
            return 0, 0.0, {"q1": None, "q3": None, "median": med, "mean": None, "std": None, "mad": mad}
        score = F.abs((c_d - F.lit(med)) / F.lit(scale))
        out = base.where(score > F.lit(float(mad_th))).count()
        pct = round(100.0 * out / n, 2)
        return int(out), pct, {"q1": None, "q3": None, "median": med, "mean": None, "std": None, "mad": mad}

# ------------------------------
# Calcul "overview" (scalable)
# ------------------------------
def _overview_metrics(df, is_spark, method="iqr", iqr_k=1.5, z_th=3.0, mad_th=3.5, rel_error=1e-3):
    cols = _numeric_columns(df, is_spark)
    if not cols:
        return pd.DataFrame(columns=["Colonne", "Nb outliers", "% outliers"])

    records = []
    if is_spark:
        # Astuce: on caste à la volée dans les fonctions pour éviter de matérialiser un DF intermédiaire colossal
        for c in cols:
            out, pct, _ = _spark_outlier_counts(
                df, c, method=method, iqr_k=iqr_k, z_th=z_th, mad_th=mad_th, rel_error=rel_error
            )
            records.append((c, int(out), float(pct)))
        return pd.DataFrame(records, columns=["Colonne", "Nb outliers", "% outliers"])
    else:
        # Pandas: conversion colonne par colonne (coerce), sans rapatrier tout en float d’un coup si inutile
        for c in cols:
            s = pd.to_numeric(df[c], errors="coerce").dropna()
            if s.empty or s.nunique(dropna=True) <= 1:
                records.append((c, 0, 0.0))
                continue
            if method == "iqr":
                q1, q3 = s.quantile(0.25), s.quantile(0.75)
                iqr = q3 - q1
                if iqr <= 0:
                    out = 0
                else:
                    low, high = q1 - iqr_k * iqr, q3 + iqr_k * iqr
                    out = int(((s < low) | (s > high)).sum())
            elif method == "z":
                m, st = float(s.mean()), float(s.std(ddof=0))
                out = 0 if st <= 0 else int((np.abs((s - m) / st) > z_th).sum())
            else:
                med = float(s.median())
                mad = float(np.median(np.abs(s - med)))
                scale = 1.4826 * mad
                out = 0 if scale <= 0 else int((np.abs(s - med) / scale > mad_th).sum())
            pct = round(100.0 * out / len(s), 2)
            records.append((c, out, pct))
        return pd.DataFrame(records, columns=["Colonne", "Nb outliers", "% outliers"])

# ------------------------------
# Sampling pour les graphiques
# ------------------------------
def _sample_for_plot(df, col_name, is_spark, sample_n=5000):
    if is_spark:
        # Prend jusqu’à sample_n non nuls, en évitant collect massif
        total_non_null = df.where(col(col_name).isNotNull()).count()
        if total_non_null == 0:
            return pd.DataFrame({col_name: []})
        frac = min(1.0, max(0.0001, float(sample_n) / float(total_non_null)))
        sdf = (
            df.select(col(col_name).cast("double").alias(col_name))
              .where(col(col_name).isNotNull())
              .sample(False, frac, seed=42)
              .limit(sample_n)
        )
        return sdf.toPandas()
    else:
        s = pd.to_numeric(df[col_name], errors="coerce").dropna()
        if len(s) > sample_n:
            s = s.sample(n=sample_n, random_state=42)
        return pd.DataFrame({col_name: s})

# ------------------------------
# Layout
# ------------------------------
def get_layout(STYLE_DROPDOWN, OPTIONS_DROPDOWN):
    return dbc.Tab(
        tab_id="outliers",
        label="Outliers",
        children=[
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
                        labelStyle={"marginRight": "0px", "padding": "2px 4px"},
                    ),
                    xs=12, md=6,
                ),
            ]),
            dbc.Row([
                dbc.Col(
                    dbc.InputGroup([
                        dbc.InputGroupText("k (IQR)"),
                        dbc.Input(id="outliers-iqr-k", type="number", value=1.5, step=0.1, min=0.1),
                    ], className="me-2", style={"maxWidth": "180px", "width": "auto"}),
                    xs="auto",
                ),
                dbc.Col(
                    dbc.InputGroup([
                        dbc.InputGroupText("Seuil |z|"),
                        dbc.Input(id="outliers-z-th", type="number", value=3.0, step=0.1, min=0.5),
                    ], className="me-2", style={"maxWidth": "180px", "width": "auto"}),
                    xs="auto",
                ),
                dbc.Col(
                    dbc.InputGroup([
                        dbc.InputGroupText("Seuil MAD"),
                        dbc.Input(id="outliers-mad-th", type="number", value=3.5, step=0.1, min=0.5),
                    ], className="me-2", style={"maxWidth": "180px", "width": "auto"}),
                    xs="auto",
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
                    ),
                    xs=12, md=6,
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
                            dbc.InputGroupText("Taille de l'échantillon N"),
                            dbc.Input(
                                id="outliers-sample-n",
                                type="number",
                                value=5000,
                                min=100,
                                step=100,
                                style={"width": "100px"},
                            ),
                        ],
                        style={"maxWidth": "600px", "width": "auto"},
                        className="d-flex align-items-center",
                    ),
                    xs="auto",
                    className="d-flex align-items-center",
                ),
            ], className="g-2 align-items-end", style={"margin": 0, "marginBottom": "4px"}),
            dbc.Row([
                dbc.Col(
                    dcc.Checklist(
                        id="outliers-log-scale",
                        options=[{"label": "Échelle log", "value": "log"}],
                        value=[],
                        inline=True,
                        style={"display": "flex", "alignItems": "center"},
                        inputStyle={"marginRight": "6px"},
                        labelStyle={"margin": 0},
                    ),
                    xs="auto",
                    className="d-flex align-items-center",
                ),
            ], className="g-2 align-items-center", style={"margin": 0}),
            html.Div(id="outliers-details-container"),
        ]
    )

# ------------------------------
# Callbacks
# ------------------------------
def register_callbacks(app):
    # 1) Alimente la checklist des colonnes numériques
    @app.callback(
        Output("outliers-columns-checklist", "options"),
        Input("original-parquet-path-store", "data"),
        prevent_initial_call=True,
    )
    def fill_columns(parquet_path):
        df, is_spark = load_df(parquet_path)
        if df is None:
            return []
        cols = _numeric_columns(df, is_spark)
        return [{"label": c, "value": c} for c in cols]

    # 2) Vue globale
    @app.callback(
        Output("outliers-overview-container", "children"),
        Input("original-parquet-path-store", "data"),
        Input("outliers-method", "value"),
        Input("outliers-iqr-k", "value"),
        Input("outliers-z-th", "value"),
        Input("outliers-mad-th", "value"),
        Input("outliers-display-mode", "value"),
        prevent_initial_call=True,
    )
    def update_overview(parquet_path, method, iqr_k, z_th, mad_th, display_mode):
        df, is_spark = load_df(parquet_path)
        if df is None:
            return format_warning("Aucun dataset chargé.")

        metrics = _overview_metrics(
            df, is_spark,
            method or "iqr",
            float(iqr_k or 1.5),
            float(z_th or 3.0),
            float(mad_th or 3.5),
            rel_error=1e-3,
        )

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
            height=400, custom_layout=layout_n,
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
            height=400, custom_layout=layout_p,
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

    # 3) Détails (plots par colonne sélectionnée)
    @app.callback(
        Output("outliers-details-container", "children"),
        Input("original-parquet-path-store", "data"),
        Input("outliers-columns-checklist", "value"),
        Input("outliers-method", "value"),
        Input("outliers-iqr-k", "value"),
        Input("outliers-z-th", "value"),
        Input("outliers-mad-th", "value"),
        Input("outliers-sample-n", "value"),
        Input("outliers-log-scale", "value"),
        prevent_initial_call=True,
    )
    def update_details(parquet_path, selected_cols, method, iqr_k, z_th, mad_th, sample_n, log_scale):
        df, is_spark = load_df(parquet_path)
        if df is None:
            return format_warning("Aucun dataset chargé.")
        if not selected_cols:
            return html.Div("Aucune colonne sélectionnée.")

        iqr_k = float(iqr_k or 1.5)
        z_th = float(z_th or 3.0)
        mad_th = float(mad_th or 3.5)
        sample_n = int(sample_n or 5000)
        log_scale = (log_scale or []) and ("log" in log_scale)

        out_graphs = []
        for c in selected_cols:
            pdf = _sample_for_plot(df, c, is_spark, sample_n=sample_n)
            s = pd.to_numeric(pdf.get(c, pd.Series([])), errors="coerce").dropna()
            if s.empty:
                out_graphs.append(html.Div(f"{c}: pas de données numériques à tracer."))
                continue

            # Seuils selon méthode (calculés sur l'échantillon affiché)
            if method == "iqr":
                q1, q3 = s.quantile(0.25), s.quantile(0.75)
                iqr = q3 - q1
                low, high = (q1 - iqr_k * iqr, q3 + iqr_k * iqr) if iqr > 0 else (None, None)
                title_suffix = f"(IQR k={iqr_k})"
            elif method == "z":
                m, st = s.mean(), s.std(ddof=0)
                low, high = (m - z_th * st, m + z_th * st) if st > 0 else (None, None)
                title_suffix = f"(Z |z|>{z_th})"
            else:
                med = s.median()
                mad = np.median(np.abs(s - med))
                scale = 1.4826 * mad
                low, high = (med - mad_th * scale, med + mad_th * scale) if scale > 0 else (None, None)
                title_suffix = f"(MAD>{mad_th})"

            # Boxplot
            fig_box = px.box(pdf, y=c, points="outliers", title=f"{c} — Boxplot {title_suffix}", height=320)
            layout_box = _get_common_layout(fig_box.layout.title.text, "", c, height=320)
            if log_scale:
                layout_box["yaxis_type"] = "log"
            fig_box.update_layout(**layout_box)

            # Histogramme + zones outliers
            fig_hist = px.histogram(pdf, x=c, nbins=50, title=f"{c} — Histogramme {title_suffix}", height=320)
            layout_hist = _get_common_layout(fig_hist.layout.title.text, c, "Fréquence", height=320)
            if log_scale:
                layout_hist["xaxis_type"] = "log"
            fig_hist.update_layout(**layout_hist)
            if low is not None and high is not None and len(s) > 0:
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
