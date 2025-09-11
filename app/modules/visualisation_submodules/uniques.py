# app/modules/visualisation_parts/uniques.py
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
from app.modules.common.viz import _get_common_layout, _generate_content
# from ..common.io import load_df, _get_common_layout, _generate_content, format_warning

def get_layout(STYLE_DROPDOWN, OPTIONS_DROPDOWN):
    return dbc.Tab(tab_id="unique_values", label="Valeurs Uniques", children=[
        html.H6("Nombre de valeurs uniques par colonne :"),
        dcc.Dropdown(
            id="unique-values-display-mode",
            options=OPTIONS_DROPDOWN,
            value="graph_descending",
            style=STYLE_DROPDOWN
        ),
        html.Div(id="unique-values-container", style={"marginBottom": "20px"}),

        html.H6("Détails des valeurs uniques :"),
        dcc.Dropdown(
            id="unique-values-column-selector",
            style=STYLE_DROPDOWN,
            placeholder="Sélectionner une colonne"
        ),
        html.Div(id="unique-values-details-container")
    ])

def _unique_counts(df, is_spark=False):
    if is_spark:
        unique_counts = df.select([F.countDistinct(col(c)).alias(c) for c in df.columns]) \
                          .toPandas().melt(var_name='Colonne', value_name='Nb de valeurs uniques')
        total_rows = df.count()
    else:
        unique_counts = pd.DataFrame({
            'Colonne': df.columns,
            'Nb de valeurs uniques': [df[c].nunique(dropna=False) for c in df.columns]
        })
        total_rows = len(df)
    unique_counts['% de valeurs uniques'] = (
        (unique_counts['Nb de valeurs uniques'] / max(1, total_rows)) * 100
    ).round(2)
    return unique_counts, total_rows

def register_callbacks(app):
    # Remplit le sélecteur de colonne pour les détails
    @app.callback(
        Output("unique-values-column-selector", "options"),
        Input("parquet-path-store", "data")
    )
    def fill_unique_selector(parquet_path):
        df, is_spark = load_df(parquet_path)
        if df is None:
            return []
        cols = df.columns if is_spark else df.columns.tolist()
        return [{"label": c, "value": c} for c in cols]

    # Graphiques Nombre + Pourcentage
    @app.callback(
        Output("unique-values-container", "children"),
        Input("parquet-path-store", "data"),
        Input("unique-values-display-mode", "value")
    )
    def update_unique_graphs(parquet_path, display_mode):
        df, is_spark = load_df(parquet_path)
        if df is None:
            return format_warning("Aucun dataset chargé.")
        counts, _ = _unique_counts(df, is_spark=is_spark)

        layout_n = _get_common_layout("Nombre de valeurs uniques par colonne",
                                    "Colonne", "Nb de valeurs uniques", height=400)
        layout_n["xaxis_showticklabels"] = False
        graph_n = _generate_content(
            counts, display_mode or "graph_descending",
            "Colonne", "Nb de valeurs uniques",
            "Nombre de valeurs uniques par colonne",
            "Colonne", "Nb de valeurs uniques",
            sort_key="Nb de valeurs uniques",
            height=400, custom_layout=layout_n
        )

        layout_p = _get_common_layout("Pourcentage de valeurs uniques par colonne",
                                    "Colonne", "% de valeurs uniques", height=400)
        layout_p["xaxis_showticklabels"] = False
        graph_p = _generate_content(
            counts, display_mode or "graph_descending",
            "Colonne", "% de valeurs uniques",
            "Pourcentage de valeurs uniques par colonne",
            "Colonne", "% de valeurs uniques",
            sort_key="% de valeurs uniques",
            height=400, custom_layout=layout_p
        )

        return html.Div([
            dbc.Row(
                [
                    dbc.Col(graph_n, xs=12, md=6),
                    dbc.Col(graph_p, xs=12, md=6),
                ],
                className="g-2",
                style={"width": "100%", "margin": 0}
            )
        ])

    # Détails d’une colonne (top 50)
    @app.callback(
        Output("unique-values-details-container", "children"),
        Input("parquet-path-store", "data"),
        Input("unique-values-column-selector", "value")
    )
    def update_unique_details(parquet_path, selected_column):
        df, is_spark = load_df(parquet_path)
        if df is None:
            return format_warning("Aucun dataset chargé.")
        if not selected_column:
            return html.Div("Aucune colonne sélectionnée.")

        if is_spark:
            vc = df.groupBy(selected_column).count().orderBy(col("count").desc()).limit(50).toPandas()
            total_rows = df.count()
        else:
            vc = df[selected_column].value_counts(dropna=False).reset_index()
            vc.columns = [selected_column, "count"]
            vc = vc.head(50)
            total_rows = len(df)

        vc["%"] = (vc["count"] / max(1, total_rows) * 100).round(2)
        fig = px.bar(vc, x=selected_column, y="count",
                     title=f"Détails des valeurs uniques — {selected_column}",
                     labels={"count": "Compte", selected_column: selected_column},
                     height=380)
        fig.update_layout(**_get_common_layout(f"Détails des valeurs uniques — {selected_column}",
                                               selected_column, "Compte", height=380))
        fig.update_layout(xaxis_showticklabels=False, bargap=0.25)
        fig.update_traces(marker_line_width=0, hovertemplate=f"{selected_column}: %{{x}}<br>Compte: %{{y}}")
        return dcc.Graph(figure=fig)
