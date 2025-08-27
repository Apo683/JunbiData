# modules/visualisation/common.py
import dash
import pandas as pd
import plotly.express as px
from dash import html, dcc, dash_table
from typing import Tuple, Optional, List
import math

# Spark optionnel
try:
    from pyspark.sql import functions as F
    from pyspark.sql.functions import col, when, monotonically_increasing_id, row_number
    from pyspark.sql.window import Window
    HAS_SPARK = True
except Exception:
    F = None
    col = when = monotonically_increasing_id = row_number = None  # type: ignore
    Window = None  # type: ignore
    HAS_SPARK = False

# spark_utils de l'app
try:
    from app.modules import spark_utils
    HAS_SPARK_UTILS = True
except Exception:
    spark_utils = None  # type: ignore
    HAS_SPARK_UTILS = False

def is_spark_active() -> bool:
    return bool(HAS_SPARK and HAS_SPARK_UTILS and getattr(spark_utils, "is_spark_active", lambda: False)())

def load_df(parquet_path: Optional[str]):
    if parquet_path is None:
        return None, False
    if is_spark_active():
        spark = spark_utils.get_spark_session()
        return spark.read.parquet(parquet_path), True
    else:
        return pd.read_parquet(parquet_path), False

def format_warning(msg: str):
    return html.I(f"⚠️ {msg}")

def _get_common_layout(title, xaxis_title, yaxis_title, height=400):
    return {
        "title": {"text": title, "x": 0.02, "xanchor": "left"},
        "height": height,
        "margin": {"l": 50, "r": 20, "t": 60, "b": 30},
        "xaxis_title": xaxis_title,
        "yaxis_title": yaxis_title,
        "template": "plotly_white",
    }

def _generate_content(
    df: pd.DataFrame,
    display_mode: str,
    x_col: str,
    y_col: str,
    title: str,
    xaxis_title: str,
    yaxis_title: str,
    sort_key: Optional[str] = None,
    color: Optional[str] = None,
    color_scale: str = "Plotly3",
    discrete_map: Optional[dict] = None,
    height: int = 400,
    custom_layout: Optional[dict] = None,
    xaxis_hide_ticks: bool = True,
    bargap: float = 0.4,
    category_order: Optional[list] = None,
):
    # Cas table
    if display_mode == "table":
        return dash_table.DataTable(
            data=(df if df is not None else pd.DataFrame()).to_dict("records"),
            columns=[{"name": c, "id": c} for c in (df.columns if df is not None else [])],
            style_table={"overflowX": "auto"},
            style_cell={"textAlign": "left", "minWidth": "120px"},
        )

    if df is None or df.empty:
        return html.Div("Aucune donnée à afficher.")

    # Tri (si l'appelant le souhaite). Pour “respecter l’ordre déjà calculé”, passer display_mode="graph_raw".
    key = sort_key or y_col
    if display_mode == "graph_ascending":
        df = df.sort_values(key, ascending=True)
    elif display_mode == "graph_descending":
        df = df.sort_values(key, ascending=False)
    # graph_raw = ordre tel quel

    fig = px.bar(
        df,
        x=x_col,
        y=y_col,
        title=title,
        labels={y_col: yaxis_title, x_col: xaxis_title},
        color=(color if color else y_col),
        color_continuous_scale=(color_scale if not discrete_map else None),
        color_discrete_map=discrete_map,
        height=height,
    )

    layout = custom_layout or _get_common_layout(title, xaxis_title, yaxis_title, height)
    if xaxis_hide_ticks:
        layout["xaxis_showticklabels"] = False
    if bargap is not None:
        layout["bargap"] = bargap
        layout["bargroupgap"] = 0.05  # un petit espace intra-groupe

    # Forcer l’ordre des catégories pour éviter toute réorganisation côté Plotly
    if category_order is not None:
        layout["xaxis"] = layout.get("xaxis", {})
        layout["xaxis"]["categoryorder"] = "array"
        layout["xaxis"]["categoryarray"] = category_order

    fig.update_layout(**layout)
    fig.update_traces(marker_line_width=0, hovertemplate=f"{x_col}: %{{x}}<br>{y_col}: %{{y}}")
    return dcc.Graph(figure=fig)
