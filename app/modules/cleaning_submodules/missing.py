# app/modules/nettoyage_submodules/missing.py
import dash_bootstrap_components as dbc
from dash import html, dcc, Input, Output, State
import pandas as pd

from app.modules.common.io import load_df, format_warning
from app.modules.common.ui import CHECKLIST_STYLE, CHECKLIST_INPUT_STYLE, CHECKLIST_LABEL_STYLE

try:
    from pyspark.sql import functions as F
    HAS_SPARK = True
except Exception:
    HAS_SPARK = False

TAB_ID = "clean-missing"

def get_tab():
    return dbc.Tab(tab_id=TAB_ID, label="Valeurs manquantes")

def get_layout():
    return html.Div([
        dbc.Row([
            dbc.Col(dcc.Checklist(
                id="miss-cols",
                    options=[],
                    value=[],  # aucune sélection par défaut
                    inline=True,
                    style=CHECKLIST_STYLE,
                    inputStyle=CHECKLIST_INPUT_STYLE,
                    labelStyle=CHECKLIST_LABEL_STYLE,
                ), xs=12),
        ], className="g-2"),
        dbc.Row([
            dbc.Col(dcc.Dropdown(
                id="miss-strategy",
                options=[
                    {"label": "Supprimer les lignes", "value": "drop"},
                    {"label": "Remplir par constante", "value": "const"},
                    {"label": "Imputer par moyenne", "value": "mean"},
                    {"label": "Imputer par médiane", "value": "median"},
                    {"label": "Imputer par mode", "value": "mode"},
                ], value="median", clearable=False
            ), xs=12, md=4),
            dbc.Col(dbc.Input(id="miss-const", type="text", placeholder="valeur constante"), xs=12, md=3),
            dbc.Col(dbc.Button("Appliquer", id="miss-apply", color="primary"), xs="auto"),
        ], className="g-2"),
        html.Div(id="miss-feedback", className="mt-2")
    ])

def register_callbacks(app):
    @app.callback(
        Output("miss-cols", "options"),
        Input("parquet-path-store", "data")
    )
    def fill_cols(parquet_path):
        df, is_spark = load_df(parquet_path)
        if df is None:
            return []
        cols = df.columns if not is_spark else df.columns
        return [{"label": c, "value": c} for c in cols]

    @app.callback(
        Output("miss-feedback", "children"),
        Input("miss-apply", "n_clicks"),
        State("parquet-path-store", "data"),
        State("miss-cols", "value"),
        State("miss-strategy", "value"),
        State("miss-const", "value"),
        prevent_initial_call=True
    )
    def apply_missing(n, parquet_path, cols, strat, const_val):
        df, is_spark = load_df(parquet_path)
        if df is None:
            return format_warning("Aucun dataset chargé.")
        if not cols:
            return format_warning("Sélectionne au moins une colonne.")

        if is_spark:
            sdf = df
            if strat == "drop":
                cond = None
                for c in cols:
                    cond = F.col(c).isNotNull() if cond is None else (cond & F.col(c).isNotNull())
                sdf = sdf.filter(cond)
            elif strat in ("const", "mean", "median", "mode"):
                for c in cols:
                    if strat == "const":
                        sdf = sdf.na.fill({c: const_val})
                    elif strat == "mean":
                        m = sdf.select(F.mean(F.col(c))).first()[0]
                        sdf = sdf.na.fill({c: m})
                    elif strat == "median":
                        med = sdf.approxQuantile(c, [0.5], 1e-3)[0]
                        sdf = sdf.na.fill({c: med})
                    else:  # mode
                        mode_row = (sdf.groupBy(c).count()
                                      .orderBy(F.col("count").desc(), F.col(c).asc())
                                      .first())
                        mode_val = None if mode_row is None else mode_row[0]
                        sdf = sdf.na.fill({c: mode_val})
            new_df = sdf
        else:
            pdf = df.copy()
            if strat == "drop":
                pdf = pdf.dropna(subset=cols, how="any")
            elif strat == "const":
                for c in cols:
                    pdf[c] = pdf[c].fillna(const_val)
            elif strat == "mean":
                for c in cols:
                    pdf[c] = pd.to_numeric(pdf[c], errors="ignore")
                    pdf[c] = pdf[c].fillna(pdf[c].mean())
            elif strat == "median":
                for c in cols:
                    pdf[c] = pd.to_numeric(pdf[c], errors="ignore")
                    pdf[c] = pdf[c].fillna(pdf[c].median())
            else:  # mode
                for c in cols:
                    mode_val = pdf[c].mode(dropna=True)
                    mode_val = mode_val.iloc[0] if not mode_val.empty else None
                    pdf[c] = pdf[c].fillna(mode_val)
            new_df = pdf

        # TODO: persister new_df (par ex. remplace/écrit parquet temporaire puis maj d’un store chemin)
        return dbc.Alert("Imputation terminée (voir TODO persistance).", color="success")
