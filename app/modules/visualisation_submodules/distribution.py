# modules/visualisation/distribution.py
import dash
import dash_bootstrap_components as dbc
from dash import html, dcc, Input, Output, State
import pandas as pd
import plotly.express as px

try:
    from pyspark.sql import functions as F
    from pyspark.sql.functions import col
    HAS_SPARK = True
except Exception:
    F = None
    col = None  # type: ignore
    HAS_SPARK = False

from .common import load_df, _get_common_layout, format_warning

def get_layout():
    return dbc.Tab(tab_id="distribution", label="Distribution", children=[
        html.Div([
            html.H6("Sélectionnez les colonnes pour afficher leur distribution :"),
            html.Div([
                dcc.Checklist(
                    id="column-selection-checklist",
                    options=[],
                    value=[],
                    inline=True,
                    style={"display": "flex", "flexWrap": "wrap", "justifyContent": "flex-start"},
                    inputStyle={"marginRight": "6px"},
                    labelStyle={
                        "width": "230px",
                        "textOverflow": "ellipsis",
                        "overflow": "hidden",
                        "whiteSpace": "nowrap",
                        "display": "inline-block",
                        "marginRight": "12px"
                    }
                )
            ], style={"marginBottom": "20px"}),
            html.Div(id="distribution-container", style={"marginTop": "20px"})
        ])
    ])

def _dist_graph_pd(df: pd.DataFrame, col_name: str):
    if pd.api.types.is_numeric_dtype(df[col_name]):
        centered = df[col_name] - df[col_name].mean()
        fig = px.histogram(centered, nbins=30, title=f"Distribution centrée de {col_name}", height=300)
        fig.update_layout(**_get_common_layout(f"Distribution centrée de {col_name}", col_name, "Compte", height=300))
        fig.update_layout(xaxis_showticklabels=False, bargap=0)
    else:
        value_counts = df[col_name].value_counts(dropna=False).reset_index()
        value_counts.columns = [col_name, "Compte"]
        value_counts = value_counts.sort_values("Compte", ascending=False)
        fig = px.bar(value_counts, x=col_name, y="Compte", title=f"Distribution de {col_name}", height=300)
        fig.update_layout(**_get_common_layout(f"Distribution de {col_name}", col_name, "Compte", height=300))
        fig.update_layout(xaxis_showticklabels=False, bargap=0)
    fig.update_traces(marker_line_width=0, hovertemplate=f"{col_name}: %{{x}}<br>Compte: %{{y}}")
    return fig

def _dist_graph_spark(df, col_name: str):
    dtype = dict(df.dtypes).get(col_name, "")
    is_numeric = any(dtype.startswith(t) for t in ["double", "int", "float", "long", "decimal", "short"])
    if is_numeric:
        min_val, max_val = df.select(F.min(col(col_name)), F.max(col(col_name))).collect()[0]
        if min_val is None or max_val is None or min_val == max_val:
            vc = df.groupBy(col_name).count().orderBy(F.desc("count")).limit(1000).toPandas()
            vc.columns = [col_name, "Compte"]
            fig = px.bar(vc, x=col_name, y="Compte", title=f"Distribution de {col_name}", height=300)
            fig.update_layout(**_get_common_layout(f"Distribution de {col_name}", col_name, "Compte", height=300))
        else:
            bucket_width = (max_val - min_val) / 30.0 if (max_val - min_val) != 0 else 1.0
            b = (((col(col_name) - F.lit(min_val)) / F.lit(bucket_width)).cast("int")).alias("bucket")
            buckets = df.select(b).groupBy("bucket").count().toPandas().sort_values("bucket")
            buckets["x"] = buckets["bucket"] * bucket_width + float(min_val)
            fig = px.bar(buckets, x="x", y="count", title=f"Distribution de {col_name}", height=300)
            fig.update_layout(**_get_common_layout(f"Distribution de {col_name}", col_name, "Compte", height=300))
    else:
        vc = df.groupBy(col_name).count().orderBy(F.desc("count")).limit(1000).toPandas()
        vc.columns = [col_name, "Compte"]
        fig = px.bar(vc, x=col_name, y="Compte", title=f"Distribution de {col_name}", height=300)
        fig.update_layout(**_get_common_layout(f"Distribution de {col_name}", col_name, "Compte", height=300))
    fig.update_layout(xaxis_showticklabels=False, bargap=0)
    fig.update_traces(marker_line_width=0, hovertemplate=f"{col_name}: %{{x}}<br>Compte: %{{y}}")
    return fig

def register_callbacks(app):
    # Alimente la checklist avec les colonnes
    @app.callback(
        Output("column-selection-checklist", "options"),
        Output("column-selection-checklist", "value"),
        Input("parquet-path-store", "data")
    )
    def fill_columns(parquet_path):
        df, is_spark = load_df(parquet_path)
        if df is None:
            return [], []
        cols = df.columns if is_spark else df.columns.tolist()
        opts = [{"label": c, "value": c} for c in cols]
        default = cols[:3] if len(cols) >= 3 else cols
        return opts, default

    # Graphs
    @app.callback(
        Output("distribution-container", "children"),
        Input("parquet-path-store", "data"),
        Input("column-selection-checklist", "value")
    )
    def update_distribution(parquet_path, selected_columns):
        df, is_spark = load_df(parquet_path)
        if df is None:
            return format_warning("Aucun dataset chargé.")
        if not selected_columns:
            return html.Div("Aucune colonne sélectionnée.")
        graphs = []
        for c in selected_columns:
            if is_spark:
                fig = _dist_graph_spark(df, c)
            else:
                if c not in df.columns:
                    continue
                fig = _dist_graph_pd(df, c)
            graphs.append(dcc.Graph(figure=fig))
        return html.Div(graphs)
