# app/modules/nettoyage_submodules/duplicates_clean.py
import dash_bootstrap_components as dbc
from dash import html, dcc, Input, Output, State

from app.modules.common.io import load_df, format_warning
# from ..common.io import load_df, format_warning

TAB_ID = "clean-duplicates"

def get_tab():
    return dbc.Tab(tab_id=TAB_ID, label="Doublons")

def get_layout():
    return html.Div([
        dbc.Row([
            dbc.Col(dcc.Dropdown(id="dup-keys", options=[], value=[], multi=True, placeholder="Clés de déduplication"), xs=12, md=6),
            dbc.Col(dcc.Dropdown(
                id="dup-keep", options=[
                    {"label": "Garder le premier", "value": "first"},
                    {"label": "Garder le dernier", "value": "last"},
                    {"label": "Supprimer tous les doublons", "value": "none"},
                ], value="first", clearable=False
            ), xs=12, md=4),
            dbc.Col(dbc.Button("Dédupliquer", id="dup-apply", color="primary"), xs="auto"),
        ], className="g-2"),
        html.Div(id="dup-feedback", className="mt-2"),
    ])

def register_callbacks(app):
    @app.callback(
        Output("dup-keys", "options"),
        Input("parquet-path-store", "data")
    )
    def fill_cols(parquet_path):
        df, is_spark = load_df(parquet_path)
        if df is None: return []
        return [{"label": c, "value": c} for c in (df.columns if not is_spark else df.columns)]

    @app.callback(
        Output("dup-feedback", "children"),
        Input("dup-apply", "n_clicks"),
        State("parquet-path-store", "data"),
        State("dup-keys", "value"),
        State("dup-keep", "value"),
        prevent_initial_call=True
    )
    def apply_dedup(n, parquet_path, keys, keep):
        df, is_spark = load_df(parquet_path)
        if df is None: return format_warning("Aucun dataset chargé.")
        if not keys: return format_warning("Sélectionne au moins une clé.")

        if is_spark:
            if keep in ("first", "last"):
                # Astuce: utiliser row_number sur une clé d’ordre si “last”
                # Par défaut, Spark dropDuplicates garde la première occurrence selon l’ordre de partition.
                sdf = df.dropDuplicates(keys) if keep == "first" else df.sort(*keys, ascending=False).dropDuplicates(keys)
            else:
                # Supprimer toutes les clés dupliquées: filtrer les groupes count == 1
                from pyspark.sql import functions as F
                cnt = df.groupBy(*keys).count()
                uniq = cnt.filter(F.col("count") == 1).drop("count")
                sdf = uniq.join(df, on=keys, how="inner")
            new_df = sdf
        else:
            if keep == "first":
                new_df = df.drop_duplicates(subset=keys, keep="first")
            elif keep == "last":
                new_df = df.drop_duplicates(subset=keys, keep="last")
            else:
                dup_mask = df.duplicated(subset=keys, keep=False)
                new_df = df[~dup_mask]

        # TODO: persister new_df
        return dbc.Alert("Déduplication effectuée (voir TODO persistance).", color="success")
