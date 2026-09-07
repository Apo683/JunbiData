# modules/common/io.py
import dash
from dash import html, dcc, dash_table
import pandas as pd
import math
import re
import json
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
MAX_PREVIEW_ROWS = 10
MAX_PREVIEW_COLUMNS = 50
MAX_CELL_LENGTH = 100
MAX_JSON_SIZE = 5 * 1024 * 1024  # 5 MB

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

def _is_missing(value):
    if value is None:
        return True

    if isinstance(value, float):
        return math.isnan(value)

    return False

# Convertit une valeur JSON/Pandas en valeur compatible avec Dash DataTable.
def _to_datatable_value(value):
    if _is_missing(value):
        return None

    if isinstance(value, (list, tuple, dict)):
        return json.dumps(
            value,
            ensure_ascii=False,
            default=str
        )

    if isinstance(value, (date, datetime)):
        return value.isoformat()

    # Conversion des types numpy éventuels
    if hasattr(value, "item"):
        try:
            value = value.item()
        except (ValueError, TypeError):
            return str(value)

    if isinstance(value, (str, int, float, bool)) or value is None:
        return value

    return str(value)

def truncate_preview_value(value, max_length=MAX_CELL_LENGTH):
    if value is None:
        return None

    if isinstance(value, float) and math.isnan(value):
        return None

    if isinstance(value, (dict, list, tuple, set)):
        value = json.dumps(
            value,
            ensure_ascii=False,
            default=str
        )

    value = str(value)

    if len(value) > max_length:
        return value[:max_length] + "…"

    return value

def prepare_preview_dataframe(
    df,
    max_rows=MAX_PREVIEW_ROWS,
    max_columns=MAX_PREVIEW_COLUMNS,
    max_cell_length=MAX_CELL_LENGTH,
):
    preview_df = df.head(max_rows).iloc[:, :max_columns].copy()
    # Évite les problèmes liés aux noms de colonnes non sérialisables
    preview_df.columns = [str(column) for column in preview_df.columns]

    for column in preview_df.columns:
        preview_df[column] = preview_df[column].map(
            lambda value: truncate_preview_value(
                value,
                max_length=max_cell_length
            )
        )

    return preview_df

def prepare_adaptive_preview(
    df,
    max_rows=MAX_PREVIEW_ROWS,
    max_columns=MAX_PREVIEW_COLUMNS,
    max_cell_length=MAX_CELL_LENGTH,
    max_json_size=MAX_JSON_SIZE,
):
    for row_count in range(max_rows, 0, -1):
        preview_df = prepare_preview_dataframe(
            df,
            max_rows=row_count,
            max_columns=max_columns,
            max_cell_length=max_cell_length,
        )

        json_str = preview_df.to_json(
            orient="split",
            force_ascii=False
        )

        if len(json_str.encode("utf-8")) <= max_json_size:
            return json_str

    return None

def get_column_dtype(df, column, is_spark=False):
    if is_spark:
        field = next(
            (
                field
                for field in df.schema.fields
                if field.name == column
            ),
            None
        )

        if field is None:
            raise ValueError(
                f"La colonne '{column}' n'existe pas dans le dataset."
            )

        return field.dataType

    if column not in df.columns:
        raise ValueError(
            f"La colonne '{column}' n'existe pas dans le dataset."
        )

    return df[column].dtype

def sample_for_plot(df, column, is_spark=False, sample_n=5000):
    if is_spark:
        return (
            df.select(column)
            .dropna()
            .limit(sample_n)
            .toPandas()
        )

    if column not in df.columns:
        raise ValueError(f"Colonne inconnue : {column}")

    pdf = df[[column]].dropna()

    if len(pdf) > sample_n:
        pdf = pdf.sample(
            n=sample_n,
            random_state=42,
        )

    return pdf