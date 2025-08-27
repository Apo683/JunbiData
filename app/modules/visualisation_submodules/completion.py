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

            html.H6("Taux de remplissage par colonne :"),
            dcc.Dropdown(
                id="completion-display-mode-cols",
                options=OPTIONS_DROPDOWN,
                value="graph_descending",
                style=STYLE_DROPDOWN
            ),
            html.Div(id="completion-cols-container", style={"marginBottom": "20px"}),

            html.H6("Taux de remplissage par ligne :"),
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
            ], style={"marginBottom": "10px"}),
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

def _auto_bargap(n_bars: int) -> float:
    if n_bars <= 0:
        return 0.30
    # piecewise simple et lisible
    if n_bars >= 800:
        return 0.48
    if n_bars >= 400:
        return 0.42
    if n_bars >= 200:
        return 0.36
    if n_bars >= 100:
        return 0.30
    if n_bars >= 50:
        return 0.22
    return 0.15

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
    if spark_df is None:
        empty = pd.DataFrame({"Ligne": [], "% de remplissage": []})
        layout = _get_common_layout("Taux de remplissage par ligne", "Numéro de ligne", "Pourcentage de remplissage (%)", height=400)
        layout["xaxis_showticklabels"] = False
        component = _generate_content(
            empty, "graph_raw",
            x_col="Ligne", y_col="% de remplissage",
            title="Taux de remplissage par ligne",
            xaxis_title="Numéro de ligne", yaxis_title="Pourcentage de remplissage (%)",
            height=400, custom_layout=layout, bargap=0.30, category_order=[]
        )
        return component, [{"label": "Page 1/1", "value": 0}], 0

    # 1) % par ligne + row_id
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
            )
            pdf_all = pdf_all.reset_index().rename(columns={"index": "row_id"})[["row_id", "pct"]]

    total_rows = len(pdf_all)
    total_pages = max(1, math.ceil((total_rows or 0) / page_size))

    # 2) Tri global (desc) une seule fois (stable)
    ordered = pdf_all.sort_values("pct", ascending=False, kind="mergesort").reset_index(drop=True)
    ordered["OrdreGlobal"] = np.arange(len(ordered), dtype=int)

    # 3) Pagination
    current_page = int(min(max(current_page or 0, 0), total_pages - 1))
    start_idx = current_page * page_size
    end_idx = min(start_idx + page_size, total_rows)
    page_df = ordered.iloc[start_idx:end_idx].copy()
    page_df["OrdrePage"] = np.arange(len(page_df), dtype=int)

    # 4) Data finale pour graphe
    df_percent = page_df.rename(columns={"OrdrePage": "Ligne", "pct": "% de remplissage"})[
        ["Ligne", "% de remplissage", "row_id", "OrdreGlobal"]
    ]

    # 5) Layout + bargap auto
    layout = _get_common_layout("Taux de remplissage par ligne", "Numéro de ligne", "Pourcentage de remplissage (%)", height=400)
    layout["xaxis_showticklabels"] = False
    bargap = _auto_bargap(len(df_percent))
    category_order = df_percent["Ligne"].tolist()  # x unique et ordonné

    component = _generate_content(
        df_percent,
        display_mode="graph_raw",  # ne pas re-trier après pagination
        x_col="Ligne",
        y_col="% de remplissage",
        title="Taux de remplissage par ligne",
        xaxis_title="Numéro de ligne",
        yaxis_title="Pourcentage de remplissage (%)",
        height=400,
        custom_layout=layout,
        bargap=bargap,
        category_order=category_order
    )

    # 6) Dropdown de pages (mettre “Page X/Y” dans le dropdown)
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
