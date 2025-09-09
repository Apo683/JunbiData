# modules/visualisation/common.py
import dash
from dash import html, dcc, dash_table
import pandas as pd
import math
import re
import plotly.express as px
import plotly.io as pio
from typing import Tuple, Optional, List

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

_PERCENT_HINTS = ("%", "pourcent", "taux", "rate", "ratio")
_COUNT_HINTS = ("compte", "count", "#", "n", "nb", "nombre", "occurrence", "fréquence")
SAFE_TEMPLATE = "junbi" if "junbi" in pio.templates else "plotly_white"

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

def _infer_hover_mode_from_ylabel(y_label: str) -> str:
    if not y_label:
        return "float2"
    yl = y_label.strip().lower()
    if any(h in yl for h in _PERCENT_HINTS):
        return "percent"
    if any(yl == h or h in yl for h in _COUNT_HINTS):
        return "int"
    # nombres entre parenthèses p.ex. "Valeur (%)"
    if re.search(r"%\)", yl):
        return "percent"
    return "float2"

def _build_hovertemplate(x_label: str, y_label: str, mode: str) -> str:
    if mode == "percent":
        return f"{x_label}: %{{x}}<br>{y_label}: %{{y:.2f}}%"
    if mode == "int":
        return f"{x_label}: %{{x}}<br>{y_label}: %{{y:.0f}}"
    if mode == "float2":
        return f"{x_label}: %{{x}}<br>{y_label}: %{{y:.2f}}"
    return f"{x_label}: %{{x}}<br>{y_label}: %{{y}}"

def _apply_smart_hover(fig, x_label: str, y_label: str):
    mode = _infer_hover_mode_from_ylabel(y_label)
    ht = _build_hovertemplate(x_label, y_label, mode)
    fig.update_traces(hovertemplate=ht)


def _get_common_layout(title, xaxis_title, yaxis_title, height=400):
    return {
        "title": {"text": title, "x": 0.5, "xanchor": "center"},
        "height": height,
        "margin": {"l": 50, "r": 20, "t": 60, "b": 30},
        "xaxis_title": xaxis_title,
        "yaxis_title": yaxis_title,
    }

def _generate_content(df, display_mode, x_col, y_col, title, xaxis_title, yaxis_title,
    sort_key=None, height=400, custom_layout=None, bargap=0.25,
    category_order=None, x_as_category=False,):
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
    data = df.copy()
    if display_mode == "graph_ascending" and sort_key:
        data = data.sort_values(sort_key, ascending=True, kind="mergesort")
    elif display_mode == "graph_descending" and sort_key:
        data = data.sort_values(sort_key, ascending=False, kind="mergesort")
    if x_as_category:
        data[x_col] = data[x_col].astype(str)
    # Création robuste de la figure avec template sûr + fallback
    try:
        fig = px.bar(
            data,
            x=x_col,
            y=y_col,
            title=title,
            height=height,
            color=y_col,
            color_continuous_scale="Bluered_r",
            template=SAFE_TEMPLATE,  # <- toujours passer un nom de template
        )
    except Exception as e:
        # Fallback au cas improbable où le template serait indisponible à cet instant
        fig = px.bar(
            data,
            x=x_col,
            y=y_col,
            title=title,
            height=height,
            color=y_col,
            color_continuous_scale="Bluered_r",
            template="plotly_white",
        )
    if custom_layout:
        fig.update_layout(**custom_layout)
    fig.update_layout(bargap=bargap)
    if category_order is not None:
        fig.update_xaxes(type="category", categoryorder="array", categoryarray=[str(v) for v in category_order])
    elif x_as_category:
        fig.update_xaxes(type="category")

    _apply_smart_hover(fig, x_col, y_col)

    return dcc.Graph(figure=fig)
