# modules/visualisation/distribution.py
import dash
import dash_bootstrap_components as dbc
from dash import html, dcc, Input, Output
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

from app.modules.common.io import load_df, format_warning
from app.modules.common.viz import _get_common_layout
from app.modules.common.ui import CHECKLIST_STYLE, CHECKLIST_INPUT_STYLE, CHECKLIST_LABEL_STYLE, get_dynamic_checklist_label_style

BARGAP = 0.30  # gap souhaité entre barres
MAX_BARS_PANDAS = 2000  # si <=, on peut afficher toutes les modalités
TOP_K = 50              # sinon on affiche Top K et on regroupe le reste en "Autres"

def get_layout():
    return dbc.Tab(
        tab_id="distribution",
        label="Distribution",
        children=[
            html.Div([
                html.H4("Sélectionnez les colonnes pour afficher leur distribution :"),
                html.Div([
                    dcc.Checklist(
                        id="column-selection-checklist",
                        options=[],
                        value=[],  # aucune sélection par défaut
                        inline=True,
                        style=CHECKLIST_STYLE,
                        inputStyle=CHECKLIST_INPUT_STYLE,
                        labelStyle=CHECKLIST_LABEL_STYLE,
                    )
                ], style={"marginBottom": "20px"}),
                html.Div(id="distribution-container", style={"marginTop": "20px"}),
            ])
        ],
    )


def _dist_graph_pd(df: pd.DataFrame, col_name: str):
    # NUMÉRIQUE: histogramme brut (non centré) pour la 1re analyse
    if pd.api.types.is_numeric_dtype(df[col_name]):
        fig = px.histogram(
            df[col_name],
            nbins=30,
            title=f"Distribution de {col_name}",
            height=300,
        )
        try:
            _get_common_layout(f"Distribution de {col_name}", col_name, "Compte", height=300, fig=fig)
        except TypeError:
            fig.update_layout(**_get_common_layout(f"Distribution de {col_name}", col_name, "Compte", height=300))
        fig.update_layout(xaxis_showticklabels=False, bargap=BARGAP)

    # CATÉGORIEL: Top K + "Autres" si beaucoup de modalités
    else:
        vc = df[col_name].value_counts(dropna=False)
        if vc.shape[0] <= MAX_BARS_PANDAS:
            value_counts = vc.reset_index()
            value_counts.columns = [col_name, "Compte"]
            value_counts = value_counts.sort_values("Compte", ascending=False, kind="mergesort")
            title = f"Distribution de {col_name}"
        else:
            topk = vc.iloc[:TOP_K]
            count_other = int(vc.iloc[TOP_K:].sum())
            value_counts = topk.reset_index()
            value_counts.columns = [col_name, "Compte"]
            if count_other > 0:
                value_counts = pd.concat(
                    [value_counts, pd.DataFrame({col_name: ["Autres"], "Compte": [count_other]})],
                    ignore_index=True
                )
            title = f"Distribution de {col_name} (Top {TOP_K} + Autres)"

        fig = px.bar(value_counts, x=col_name, y="Compte", title=title, height=300)
        try:
            _get_common_layout(title, col_name, "Compte", height=300, fig=fig)
        except TypeError:
            fig.update_layout(**_get_common_layout(title, col_name, "Compte", height=300))
        fig.update_layout(xaxis_showticklabels=False, bargap=BARGAP)

    fig.update_traces(marker_line_width=0)

    return fig



def _dist_graph_spark(df, col_name: str):
    dtype = dict(df.dtypes).get(col_name, "")
    is_numeric = any(dtype.startswith(t) for t in ["double", "int", "float", "long", "decimal", "short"])

    if is_numeric:
        # Histogramme distribué sur TOUT le dataset
        min_val, max_val = df.select(F.min(col(col_name)), F.max(col(col_name))).collect()[0]
        if min_val is None or max_val is None:
            # Rien d’exploitable
            vc = df.groupBy(col_name).count().orderBy(F.desc("count")).toPandas()
            vc.columns = [col_name, "Compte"]
            fig = px.bar(vc, x=col_name, y="Compte", title=f"Distribution de {col_name}", height=300)
            fig.update_layout(**_get_common_layout(f"Distribution de {col_name}", col_name, "Compte", height=300))
        elif min_val == max_val:
            # Colonne constante -> une seule barre, pas de limit nécessaire
            single = df.count()
            vc = pd.DataFrame({col_name: [min_val], "Compte": [single]})
            fig = px.bar(vc, x=col_name, y="Compte", title=f"Distribution de {col_name}", height=300)
            fig.update_layout(**_get_common_layout(f"Distribution de {col_name}", col_name, "Compte", height=300))
        else:
            # 30 bacs sur toute la colonne (bucketing DataFrame, pas de limit)
            span = float(max_val) - float(min_val)
            bucket_width = (span / 30.0) if span != 0 else 1.0
            b = (((col(col_name) - F.lit(float(min_val))) / F.lit(bucket_width)).cast("int")).alias("bucket")
            buckets = df.select(b).groupBy("bucket").count().toPandas().sort_values("bucket")
            buckets["x"] = buckets["bucket"] * bucket_width + float(min_val)
            fig = px.bar(buckets, x="x", y="count", title=f"Distribution de {col_name}", height=300)
            fig.update_layout(**_get_common_layout(f"Distribution de {col_name}", col_name, "Compte", height=300))
    else:
        # Catégoriel: distribution complète, puis adaptation d'affichage
        ndv = df.select(F.approx_count_distinct(col(col_name))).collect()[0][0]
        # Comptes exacts
        vc_spark = df.groupBy(col_name).count()
        vc_spark = vc_spark.orderBy(F.desc("count"))

        if ndv <= MAX_BARS_PANDAS:
            vc = vc_spark.toPandas()
            vc.columns = [col_name, "Compte"]
            title = f"Distribution de {col_name}"
        else:
            # Top K + Autres (somme du reste) pour l'affichage
            topk_pdf = vc_spark.limit(TOP_K).toPandas()
            topk_pdf.columns = [col_name, "Compte"]
            total_count = df.count()
            count_topk = int(topk_pdf["Compte"].sum())
            count_other = total_count - count_topk
            vc = topk_pdf
            if count_other > 0:
                vc = pd.concat([vc, pd.DataFrame({col_name: ["Autres"], "Compte": [count_other]})], ignore_index=True)
            title = f"Distribution de {col_name} (Top {TOP_K} + Autres)"

        fig = px.bar(vc, x=col_name, y="Compte", title=title, height=300)
        fig.update_layout(**_get_common_layout(title, col_name, "Compte", height=300))

    fig.update_layout(xaxis_showticklabels=False, bargap=BARGAP)
    fig.update_traces(marker_line_width=0, hovertemplate=f"{col_name}: %{{x}}<br>Compte: %{{y}}")
    return fig

def register_callbacks(app):
    # Alimente la checklist avec les colonnes (aucune sélection par défaut)
    @app.callback(
        Output("column-selection-checklist", "options"),
        Output("column-selection-checklist", "value"),
        Output("column-selection-checklist", "labelStyle"),
        Input("original-parquet-path-store", "data"),
    )
    def fill_columns(parquet_path):
        df, is_spark = load_df(parquet_path)
        if df is None:
            return [], [], CHECKLIST_LABEL_STYLE
        cols = df.columns if is_spark else df.columns.tolist()
        options = [{"label": c, "value": c} for c in cols]
        label_style = get_dynamic_checklist_label_style(options)
        return options, [], label_style  # valeur vide => pas de sélection auto

    # Graphs
    @app.callback(
        Output("distribution-container", "children"),
        Input("original-parquet-path-store", "data"),
        Input("column-selection-checklist", "value"),
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