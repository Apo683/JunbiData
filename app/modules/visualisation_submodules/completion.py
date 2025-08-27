# modules/visualisation_parts/completion.py
import dash
import dash_bootstrap_components as dbc
from dash import html, dcc, Input, Output
import numpy as np
import pandas as pd
from functools import reduce
from operator import add
import math

try:
    from pyspark.sql import functions as F
    from pyspark.sql.functions import col, when, monotonically_increasing_id, row_number, lit
    from pyspark.sql.window import Window
    HAS_SPARK = True
except Exception:
    F = None
    col = when = monotonically_increasing_id = row_number = lit = None  # type: ignore
    Window = None  # type: ignore
    HAS_SPARK = False

from .common import load_df, _get_common_layout, _generate_content, format_warning

# -------------------------
# Layout
# -------------------------
def get_layout(STYLE_DROPDOWN, OPTIONS_DROPDOWN):
    return dbc.Tab(tab_id="completion", label="Taux de remplissage", children=[
        html.Div([
            html.Div(id="completion-mean-container", style={"textAlign": "left", "marginBottom": "20px"}),

            html.P("Taux de remplissage par colonne :", style={"fontSize": "18px"}),
            dcc.Dropdown(
                id="completion-display-mode-cols",
                options=OPTIONS_DROPDOWN,
                value="graph_descending",
                style=STYLE_DROPDOWN
            ),
            html.Div(id="completion-cols-container", style={"marginBottom": "30px"}),

            html.P("Taux de remplissage par ligne :", style={"fontSize": "18px"}),
            dcc.Dropdown(
                id="completion-display-mode-rows",
                options=OPTIONS_DROPDOWN,
                value="graph_descending",
                style=STYLE_DROPDOWN
            ),
            html.Div([
                dbc.Button("Précédent", id="prev-page-rows", n_clicks=0, style={"marginRight": "10px"}),
                dbc.Button("Suivant", id="next-page-rows", n_clicks=0, style={"marginRight": "10px"}),
                dcc.Dropdown(
                    id="page-select-rows",
                    options=[],
                    value=0,
                    clearable=False,
                    style={"width": "180px", "display": "inline-grid", "backgroundColor": "#ffffff", "color": "#000", "borderRadius": "5px"}
                ),
            ], style={"marginBottom": "15px"}),
            html.Div(id="completion-rows-container", style={"marginBottom": "20px"})
        ])
    ])

# -------------------------
# Helpers
# -------------------------
def _format_mean(df, is_spark=False):
    if df is None:
        return format_warning("Aucun dataset chargé.")
    if is_spark:
        total_rows = df.count()
        if total_rows == 0:
            mean_pct = 0.0
        else:
            per_col = df.select(
                [(F.count(when(col(c).isNotNull(), c)) / total_rows * 100).alias(c) for c in df.columns]
            ).toPandas()
            mean_pct = float(per_col.mean().mean())
    else:
        if len(df) == 0:
            mean_pct = 0.0
        else:
            mean_pct = float((df.notna().sum(axis=1) / df.shape[1] * 100).mean())
    return html.Div([html.P(f"Taux de remplissage moyen : {mean_pct:.2f}%", style={"fontSize": "18px"})])

def _cols_content(spark_df, display_mode, is_spark=False):
    if spark_df is None:
        return format_warning("Aucun dataset chargé pour les colonnes.")
    if is_spark:
        total_rows = spark_df.count()
        if total_rows == 0:
            completion_percent = pd.DataFrame({"Colonne": [], "% de remplissage": []})
        else:
            cols = spark_df.columns
            exprs = [(F.count(when(col(c).isNotNull(), c)) / total_rows * 100).alias(c) for c in cols]
            pdf = spark_df.select(exprs).toPandas().T
            pdf.columns = ["% de remplissage"]
            completion_percent = pdf.reset_index().rename(columns={"index": "Colonne"})
    else:
        if len(spark_df) == 0:
            completion_percent = pd.DataFrame({"Colonne": [], "% de remplissage": []})
        else:
            pct = (spark_df.notna().sum(axis=0) / len(spark_df) * 100.0)
            completion_percent = pd.DataFrame({"Colonne": pct.index, "% de remplissage": pct.values})

    layout = _get_common_layout("Taux de remplissage par colonne", "Colonnes", "Pourcentage de remplissage (%)", height=400)
    layout["xaxis_showticklabels"] = False

    # Ici on laisse le tri au display_mode (graph_descending par défaut).
    return _generate_content(
        completion_percent,
        display_mode=display_mode or "graph_descending",
        x_col="Colonne",
        y_col="% de remplissage",
        title="Taux de remplissage par colonne",
        xaxis_title="Colonnes",
        yaxis_title="Pourcentage de remplissage (%)",
        sort_key="% de remplissage",
        height=400,
        custom_layout=layout
    )

def _rows_content(spark_df, display_mode, current_page=0, page_size=1000, is_spark=False):
    # Cas sans dataset
    if spark_df is None:
        empty = pd.DataFrame({"Ligne": [], "% de remplissage": []})
        layout = _get_common_layout("Taux de remplissage par ligne", "Numéro de ligne", "Pourcentage de remplissage (%)", height=400)
        layout["xaxis_showticklabels"] = False
        component = _generate_content(
            empty, "graph_raw",
            x_col="Ligne", y_col="% de remplissage",
            title="Taux de remplissage par ligne",
            xaxis_title="Numéro de ligne", yaxis_title="Pourcentage de remplissage (%)",
            height=400, custom_layout=layout,
            category_order=[], x_as_category=True
        )
        return component, [{"label": "Page 1/1", "value": 0}], 0

    # 1) % par ligne + id stable
    if is_spark:
        n_cols = len(spark_df.columns)
        if n_cols == 0:
            pdf_all = pd.DataFrame({"row_id": [], "pct": []})
        else:
            df_idx = spark_df.withColumn("_row_id", row_number().over(Window.orderBy(monotonically_increasing_id())) - 1)
            indicators = [when(col(c).isNotNull(), 1).otherwise(0) for c in spark_df.columns]
            filled = reduce(add, indicators)
            df_pct = df_idx.withColumn("pct", (filled / F.lit(n_cols)) * 100.0)
            pdf_all = df_pct.select("_row_id", "pct").toPandas().rename(columns={"_row_id": "row_id"})
    else:
        if spark_df.shape[1] == 0:
            pdf_all = pd.DataFrame({"row_id": [], "pct": []})
        else:
            pdf_all = pd.DataFrame(index=spark_df.index).assign(
                pct=(spark_df.notna().sum(axis=1) / spark_df.shape[1] * 100.0)
            ).reset_index().rename(columns={"index": "row_id"})[["row_id", "pct"]]

    # 2) Tri global (hors pagination)
    num_columns = spark_df.shape[1]
    total_rows = len(pdf_all)
    if num_columns == 0 or total_rows == 0:
        completion_percent_all = pd.Series([0.0] * total_rows, index=spark_df.index)
    else:
        completion_percent_all = (spark_df.notna().sum(axis=1) / num_columns) * 100

    # tri global (sans rien paginer ici)
    if display_mode in ("graph_ascending", "table"):
        ordered = completion_percent_all.sort_values(ascending=True, kind="mergesort")
    elif display_mode == "graph_descending":
        ordered = completion_percent_all.sort_values(ascending=False, kind="mergesort")
    else:
        ordered = completion_percent_all  # "graph_raw" => pas de tri ici

    # on garde l'index d'origine pour afficher "la vraie ligne"
    ordered_df = pd.DataFrame({
        "row_id": ordered.index,
        "% de remplissage": ordered.values
    }).reset_index(drop=True)

    # pagination
    total_pages = max(1, math.ceil((len(ordered_df) or 0) / page_size))
    current_page = min(max(int(current_page or 0), 0), total_pages - 1)
    start_idx = current_page * page_size
    end_idx = min(start_idx + page_size, len(ordered_df))
    page_df = ordered_df.iloc[start_idx:end_idx].copy()

    # X catégoriel texte, comme dans l’ancienne version:
    # "Ligne <row_id+1>" pour rester 1-based si tu le souhaites
    page_df["Ligne"] = page_df["row_id"].map(lambda r: f"Ligne {int(r)+1}")

    # Y arrondi à 2 décimales dans les données
    page_df["% de remplissage"] = page_df["% de remplissage"].round(2)

    df_percent = page_df[["Ligne", "% de remplissage"]]
    category_order = df_percent["Ligne"].tolist()

    # 5) Layout + bargap
    layout = _get_common_layout(
        "Taux de remplissage par ligne",
        "Numéro de ligne",
        "Pourcentage de remplissage (%)",
        height=400
    )
    layout["xaxis_showticklabels"] = False

    component = _generate_content(
        df_percent,
        display_mode="graph_raw",          # pas de tri interne
        x_col="Ligne",
        y_col="% de remplissage",
        title="Taux de remplissage par ligne",
        xaxis_title="Numéro de ligne",
        yaxis_title="Pourcentage de remplissage (%)",
        height=400,
        custom_layout=layout,
        bargap=0.40,
        category_order=category_order,     # ordre exact des X de la page
        x_as_category=True                 # axe X catégoriel
    )

    page_options = [{"label": f"Page {i+1}/{total_pages}", "value": i} for i in range(total_pages)]
    page_value = current_page
    return component, page_options, page_value

# -------------------------
# Callbacks
# -------------------------
def register_callbacks(app):
    # Moyenne
    @app.callback(
        Output("completion-mean-container", "children"),
        Input("parquet-path-store", "data")
    )
    def update_mean(parquet_path):
        spark_df, is_spark = load_df(parquet_path)
        return _format_mean(spark_df, is_spark=is_spark)

    # Colonnes
    @app.callback(
        Output("completion-cols-container", "children"),
        Input("parquet-path-store", "data"),
        Input("completion-display-mode-cols", "value"),
        prevent_initial_call=False
    )
    def update_cols(parquet_path, display_mode_cols):
        spark_df, is_spark = load_df(parquet_path)
        if parquet_path is None:
            return format_warning("Aucun dataset chargé.")
        return _cols_content(spark_df, display_mode_cols or "graph_descending", is_spark=is_spark)

    # Lignes + pagination (le dropdown devient la vérité)
    @app.callback(
        Output("completion-rows-container", "children"),
        Output("page-select-rows", "options"),
        Output("page-select-rows", "value"),
        Input("parquet-path-store", "data"),
        Input("completion-display-mode-rows", "value"),
        Input("prev-page-rows", "n_clicks"),
        Input("next-page-rows", "n_clicks"),
        Input("page-select-rows", "value"),
        prevent_initial_call=False
    )
    def update_rows(parquet_path, display_mode_rows, prev_clicks, next_clicks, page_select_value):
        # 0) dataset non chargé → état neutre
        if parquet_path is None:
            empty = html.Div("Aucun dataset chargé.")
            return empty, [{"label": "Page 1/1", "value": 0}], 0

        spark_df, is_spark = load_df(parquet_path)
        display_mode_rows = display_mode_rows or "graph_descending"
        current_page = int(page_select_value or 0)  # 0-based

        ctx = dash.callback_context
        if ctx.triggered:
            trig = ctx.triggered[0]["prop_id"]
            if "prev-page-rows" in trig:
                current_page = max(0, current_page - 1)
            elif "next-page-rows" in trig:
                current_page = current_page + 1
            elif "page-select-rows" in trig and page_select_value is not None:
                current_page = max(0, int(page_select_value))  # pas de -1 !

        component, page_options, page_value = _rows_content(
            spark_df, display_mode_rows, current_page, is_spark=is_spark
        )
        return component, page_options, page_value
