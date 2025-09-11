# app/modules/nettoyage_submodules/outliers_clean.py
import dash_bootstrap_components as dbc
from dash import html, dcc, Input, Output, State
import numpy as np
import pandas as pd

from app.modules.common.io import load_df, format_warning
# from ..common.io import load_df, format_warning
try:
    from pyspark.sql import functions as F
    from pyspark.sql.functions import col, when
    HAS_SPARK = True
except Exception:
    HAS_SPARK = False

TAB_ID = "clean-outliers"

def get_tab():
    return dbc.Tab(tab_id=TAB_ID, label="Outliers")

def get_layout():
    return html.Div([
        dbc.Row([
            dbc.Col(dcc.Dropdown(id="out-cols", options=[], value=[], multi=True, placeholder="Colonnes numériques"), xs=12, md=6),
            dbc.Col(dcc.Dropdown(
                id="out-method", options=[
                    {"label": "IQR (Tukey)", "value": "iqr"},
                    {"label": "Z-score", "value": "z"},
                    {"label": "MAD", "value": "mad"},
                ], value="iqr", clearable=False
            ), xs=12, md=3),
            dbc.Col(dcc.Dropdown(
                id="out-action", options=[
                    {"label": "Supprimer les lignes outliers", "value": "drop"},
                    {"label": "Caper (winsoriser)", "value": "cap"},
                    {"label": "Marquer (is_outlier=1/0)", "value": "flag"},
                ], value="flag", clearable=False
            ), xs=12, md=4),
        ], className="g-2"),
        dbc.Row([
            dbc.Col(dbc.InputGroup([dbc.InputGroupText("k (IQR)"), dbc.Input(id="out-iqr-k", type="number", value=1.5, step=0.1)]), xs="auto"),
            dbc.Col(dbc.InputGroup([dbc.InputGroupText("|z|"), dbc.Input(id="out-z", type="number", value=3.0, step=0.1)]), xs="auto"),
            dbc.Col(dbc.InputGroup([dbc.InputGroupText("MAD th"), dbc.Input(id="out-mad", type="number", value=3.5, step=0.1)]), xs="auto"),
            dbc.Col(dbc.Button("Appliquer", id="out-apply", color="primary"), xs="auto"),
        ], className="g-2"),
        html.Div(id="out-feedback", className="mt-2")
    ])

def _numeric_cols(df, is_spark):
    if is_spark:
        return [f.name for f in df.schema.fields if f.dataType.simpleString() in ("int", "bigint", "float", "double", "smallint", "decimal")]
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

def register_callbacks(app):
    @app.callback(
        Output("out-cols", "options"),
        Input("parquet-path-store", "data")
    )
    def fill_cols(path):
        df, is_spark = load_df(path)
        if df is None: return []
        cols = _numeric_cols(df, is_spark)
        return [{"label": c, "value": c} for c in cols]

    @app.callback(
        Output("out-feedback", "children"),
        Input("out-apply", "n_clicks"),
        State("parquet-path-store", "data"),
        State("out-cols", "value"),
        State("out-method", "value"),
        State("out-action", "value"),
        State("out-iqr-k", "value"),
        State("out-z", "value"),
        State("out-mad", "value"),
        prevent_initial_call=True
    )
    def apply_outliers(n, path, cols, method, action, iqr_k, z_th, mad_th):
        df, is_spark = load_df(path)
        if df is None: return format_warning("Aucun dataset chargé.")
        if not cols: return format_warning("Sélectionne au moins une colonne.")
        iqr_k = float(iqr_k or 1.5); z_th = float(z_th or 3.0); mad_th = float(mad_th or 3.5)

        if is_spark:
            sdf = df
            for c in cols:
                if method == "iqr":
                    q1, q3 = sdf.approxQuantile(c, [0.25, 0.75], 1e-3)
                    iqr = q3 - q1
                    low, high = (q1 - iqr_k*iqr, q3 + iqr_k*iqr) if iqr > 0 else (None, None)
                elif method == "z":
                    stats = sdf.select(F.mean(c).alias("m"), F.stddev_pop(c).alias("s")).first()
                    m, s = stats["m"], stats["s"]
                    low, high = ((m - z_th*s, m + z_th*s) if (s and s > 0) else (None, None))
                else:
                    med = sdf.approxQuantile(c, [0.5], 1e-3)[0]
                    mad = sdf.select(F.expr(f"percentile_approx(abs({c} - {med}), 0.5, 10000) as mad")).first()["mad"]
                    scale = 1.4826 * mad if mad is not None else None
                    low, high = ((med - mad_th*scale, med + mad_th*scale) if (scale and scale > 0) else (None, None))
                if action == "flag":
                    flag_col = f"is_outlier_{c}"
                    if low is None or high is None:
                        sdf = sdf.withColumn(flag_col, F.lit(0))
                    else:
                        sdf = sdf.withColumn(flag_col, ( (F.col(c) < F.lit(low)) | (F.col(c) > F.lit(high)) ).cast("int"))
                elif action == "cap":
                    if low is not None and high is not None:
                        sdf = sdf.withColumn(c, when(col(c) < low, low).when(col(c) > high, high).otherwise(col(c)))
                else:  # drop
                    if low is not None and high is not None:
                        sdf = sdf.filter( (col(c) >= low) & (col(c) <= high) )
            new_df = sdf
        else:
            pdf = df.copy()
            for c in cols:
                s = pd.to_numeric(pdf[c], errors="coerce")
                if method == "iqr":
                    q1, q3 = s.quantile(0.25), s.quantile(0.75); iqr = q3 - q1
                    low, high = (q1 - iqr_k*iqr, q3 + iqr_k*iqr) if iqr > 0 else (None, None)
                elif method == "z":
                    m, st = s.mean(), s.std(ddof=0); low, high = ((m - z_th*st, m + z_th*st) if st > 0 else (None, None))
                else:
                    med = s.median(); mad = np.median(np.abs(s - med)); scale = 1.4826*mad
                    low, high = ((med - mad_th*scale, med + mad_th*scale) if scale > 0 else (None, None))
                if action == "flag":
                    pdf[f"is_outlier_{c}"] = ((s < low) | (s > high)).astype(int) if (low is not None) else 0
                elif action == "cap" and (low is not None):
                    pdf[c] = s.clip(low, high)
                elif action == "drop" and (low is not None):
                    pdf = pdf[(s >= low) & (s <= high)]
            new_df = pdf

        # TODO: persister new_df
        return dbc.Alert("Traitement outliers effectué (voir TODO persistance).", color="success")
