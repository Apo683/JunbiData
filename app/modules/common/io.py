# modules/common/io.py
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

def load_df_only(parquet_path: Optional[str]):
    if not parquet_path:
        return None
    res = load_df(parquet_path)
    if res is None:
        return None
    if isinstance(res, tuple):
        return res[0]
    return res

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
