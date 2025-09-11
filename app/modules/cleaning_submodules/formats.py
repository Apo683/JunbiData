# app/modules/nettoyage_submodules/formats.py
import dash_bootstrap_components as dbc
from dash import html, dcc, Input, Output, State
import unicodedata

from app.modules.common.io import load_df, format_warning
# from ..common.io import load_df, format_warning
try:
    from pyspark.sql import functions as F
    HAS_SPARK = True
except Exception:
    HAS_SPARK = False

TAB_ID = "clean-formats"

def get_tab():
    return dbc.Tab(tab_id=TAB_ID, label="Formats")

def get_layout():
    return html.Div([
        dbc.Row([
            dbc.Col(dcc.Dropdown(id="fmt-cols", options=[], value=[], multi=True, placeholder="Colonnes"), xs=12, md=6),
            dbc.Col(dcc.Dropdown(
                id="fmt-action", options=[
                    {"label": "Cast en numérique", "value": "to_numeric"},
                    {"label": "Cast en date (yyyy-MM-dd)", "value": "to_date"},
                    {"label": "Trimming espaces", "value": "trim"},
                    {"label": "Normaliser texte (minuscules, accents)", "value": "normalize"},
                ], value="trim", clearable=False
            ), xs=12, md=4),
            dbc.Col(dbc.Button("Appliquer", id="fmt-apply", color="primary"), xs="auto"),
        ], className="g-2"),
        html.Div(id="fmt-feedback", className="mt-2"),
    ])

def register_callbacks(app):
    @app.callback(
        Output("fmt-cols", "options"),
        Input("parquet-path-store", "data")
    )
    def fill_cols(path):
        df, is_spark = load_df(path)
        if df is None: return []
        return [{"label": c, "value": c} for c in (df.columns if not is_spark else df.columns)]

    @app.callback(
        Output("fmt-feedback", "children"),
        Input("fmt-apply", "n_clicks"),
        State("parquet-path-store", "data"),
        State("fmt-cols", "value"),
        State("fmt-action", "value"),
        prevent_initial_call=True
    )
    def apply_fmt(n, path, cols, action):
        df, is_spark = load_df(path)
        if df is None: return format_warning("Aucun dataset chargé.")
        if not cols: return format_warning("Sélectionne au moins une colonne.")

        if is_spark:
            sdf = df
            if action == "to_numeric":
                for c in cols:
                    sdf = sdf.withColumn(c, F.regexp_replace(F.col(c).cast("string"), ",", ".").cast("double"))
            elif action == "to_date":
                for c in cols:
                    sdf = sdf.withColumn(c, F.to_date(F.col(c).cast("string"), "yyyy-MM-dd"))
            elif action == "trim":
                for c in cols:
                    sdf = sdf.withColumn(c, F.trim(F.col(c).cast("string")))
            else:  # normalize
                # simpliste: lower+remove accents
                # nécessite udf légère
                from pyspark.sql.types import StringType
                def _norm(s):
                    if s is None: return None
                    s = str(s).lower().strip()
                    s = "".join(ch for ch in unicodedata.normalize("NFKD", s) if not unicodedata.combining(ch))
                    return s
                norm_udf = F.udf(_norm, StringType())
                for c in cols:
                    sdf = sdf.withColumn(c, norm_udf(F.col(c)))
            new_df = sdf
        else:
            pdf = df.copy()
            if action == "to_numeric":
                for c in cols:
                    pdf[c] = (pdf[c].astype(str).str.replace(",", ".", regex=False))
                    pdf[c] = pd.to_numeric(pdf[c], errors="coerce")
            elif action == "to_date":
                for c in cols:
                    pdf[c] = pd.to_datetime(pdf[c], errors="coerce").dt.date
            elif action == "trim":
                for c in cols:
                    pdf[c] = pdf[c].astype(str).str.strip()
            else:
                def _norm(s):
                    if s is None: return None
                    s = str(s).lower().strip()
                    s = "".join(ch for ch in unicodedata.normalize("NFKD", s) if not unicodedata.combining(ch))
                    return s
                for c in cols:
                    pdf[c] = pdf[c].map(_norm)
            new_df = pdf

        # TODO: persister new_df
        return dbc.Alert("Formatage appliqué (voir TODO persistance).", color="success")
