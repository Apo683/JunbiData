# app/modules/visualisation_parts/duplicates.py
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
from app.modules.common.ui import STYLE_DROPDOWN, OPTIONS_DROPDOWN
from app.modules.common.viz import _get_common_layout, _generate_content
# from ..common.io import load_df, _get_common_layout, _generate_content, format_warning

def get_layout(STYLE_DROPDOWN, OPTIONS_DROPDOWN):
    # OPTIONS_DROPDOWN: réutilise celles de uniques (graph_descending, graph_ascending, table, etc.)
    return dbc.Tab(tab_id="duplicates", label="Doublons", children=[
        html.H6("Vue globale des doublons par colonne :"),
        dcc.Dropdown(
            id="duplicates-display-mode",
            options=OPTIONS_DROPDOWN,
            value="graph_descending",
            style=STYLE_DROPDOWN
        ),
        html.Div(id="duplicates-overview-container", style={"marginBottom": "20px"}),

        html.H6("Détails des valeurs dupliquées :"),
        dcc.Dropdown(
            id="duplicates-column-selector",
            style=STYLE_DROPDOWN,
            placeholder="Sélectionner une colonne"
        ),
        html.Div(id="duplicates-details-container")
    ])

def _duplicate_metrics(df, is_spark=False):
    """
    Retourne:
      metrics: DataFrame [Colonne, Nb de valeurs dupliquées, Nb de lignes dupliquées, % de lignes dupliquées]
      total_rows: int
    Définitions:
      - Nb de valeurs dupliquées: nb de valeurs distinctes de la colonne dont count >= 2
      - Nb de lignes dupliquées: somme des counts de ces valeurs (i.e., lignes appartenant à des doublons)
    """
    if df is None:
        return pd.DataFrame(columns=[
            "Colonne", "Nb de valeurs dupliquées", "Nb de lignes dupliquées", "% de lignes dupliquées"
        ]), 0

    if is_spark:
        total_rows = df.count()
        cols = df.columns
        vals_dupes = []
        rows_dupes = []
        # Une agrégation par colonne (coûte O(#colonnes), mais scalable par colonne)
        for c in cols:
            gb = df.groupBy(c).agg(F.count(F.lit(1)).alias("cnt"))
            dupes = gb.where(col("cnt") >= 2)
            # nb de valeurs (distinctes) dupliquées
            n_vals = dupes.count()
            # nb de lignes impliquées (somme des counts des valeurs dupliquées)
            n_rows = dupes.agg(F.sum("cnt")).collect()[0][0] or 0
            vals_dupes.append(int(n_vals))
            rows_dupes.append(int(n_rows))

        metrics = pd.DataFrame({
            "Colonne": list(cols),
            "Nb de valeurs dupliquées": vals_dupes,
            "Nb de lignes dupliquées": rows_dupes
        })
    else:
        total_rows = len(df)
        cols = df.columns.tolist()
        vals_dupes = []
        rows_dupes = []
        for c in cols:
            vc = df[c].value_counts(dropna=False)
            dupes = vc[vc >= 2]
            vals_dupes.append(int((dupes >= 2).sum()))
            rows_dupes.append(int(dupes.sum()) if not dupes.empty else 0)
        metrics = pd.DataFrame({
            "Colonne": cols,
            "Nb de valeurs dupliquées": vals_dupes,
            "Nb de lignes dupliquées": rows_dupes
        })

    metrics["% de lignes dupliquées"] = (
        metrics["Nb de lignes dupliquées"] / max(1, total_rows) * 100.0
    ).round(2)
    return metrics, total_rows

def register_callbacks(app):
    # Remplir le sélecteur de colonne pour les détails
    @app.callback(
        Output("duplicates-column-selector", "options"),
        Input("parquet-path-store", "data")
    )
    def fill_duplicates_selector(parquet_path):
        df, is_spark = load_df(parquet_path)
        if df is None:
            return []
        cols = df.columns if is_spark else df.columns.tolist()
        return [{"label": c, "value": c} for c in cols]

    # Graphiques overview (Nb valeurs dupliquées / % de lignes dupliquées)
    @app.callback(
        Output("duplicates-overview-container", "children"),
        Input("parquet-path-store", "data"),
        Input("duplicates-display-mode", "value")
    )
    def update_duplicates_overview(parquet_path, display_mode):
        df, is_spark = load_df(parquet_path)
        if df is None:
            return format_warning("Aucun dataset chargé.")
        metrics, _ = _duplicate_metrics(df, is_spark=is_spark)

        # Graph 1: Nb de valeurs dupliquées par colonne
        layout_n = _get_common_layout("Nombre de valeurs dupliquées par colonne",
                                      "Colonne", "Nb de valeurs dupliquées", height=400)
        layout_n["xaxis_showticklabels"] = False
        graph_n = _generate_content(
            metrics, display_mode or "graph_descending",
            "Colonne", "Nb de valeurs dupliquées",
            "Nombre de valeurs dupliquées par colonne",
            "Colonne", "Nb de valeurs dupliquées",
            sort_key="Nb de valeurs dupliquées",
            height=400, custom_layout=layout_n
        )

        # Graph 2: % de lignes dupliquées par colonne
        layout_p = _get_common_layout("% de lignes dupliquées par colonne",
                                      "Colonne", "% de lignes dupliquées", height=400)
        layout_p["xaxis_showticklabels"] = False
        graph_p = _generate_content(
            metrics, display_mode or "graph_descending",
            "Colonne", "% de lignes dupliquées",
            "% de lignes dupliquées par colonne",
            "Colonne", "% de lignes dupliquées",
            sort_key="% de lignes dupliquées",
            height=400, custom_layout=layout_p
        )

        return html.Div([
            dbc.Row(
                [
                    dbc.Col(graph_n, xs=12, md=6, className="mb-3 mb-md-0"),
                    dbc.Col(graph_p, xs=12, md=6),
                ],
                className="g-2 flex-wrap",
                style={"margin": 0}
            )
        ])

    # Détails d'une colonne (valeurs dupliquées, top 50)
    @app.callback(
        Output("duplicates-details-container", "children"),
        Input("parquet-path-store", "data"),
        Input("duplicates-column-selector", "value")
    )
    def update_duplicates_details(parquet_path, selected_column):
        df, is_spark = load_df(parquet_path)
        if df is None:
            return format_warning("Aucun dataset chargé.")
        if not selected_column:
            return html.Div("Aucune colonne sélectionnée.")

        if is_spark:
            # counts triés desc, filtrés sur cnt >= 2, limit 50
            vc = (df.groupBy(selected_column)
                    .count()
                    .where(col("count") >= 2)
                    .orderBy(col("count").desc())
                    .limit(50)
                    .toPandas())
            total_rows = df.count()
        else:
            s = df[selected_column]
            vc = s.value_counts(dropna=False)
            vc = vc[vc >= 2].reset_index()
            vc.columns = [selected_column, "count"]
            vc = vc.head(50)
            total_rows = len(df)

        if len(vc) == 0:
            return format_warning("Aucun doublon détecté pour cette colonne.")

        vc["% lignes (base toutes lignes)"] = (vc["count"] / max(1, total_rows) * 100).round(2)

        # Barres sur le compte (comme uniques détails)
        fig = px.bar(
            vc, x=selected_column, y="count",
            title=f"Doublons — détails: {selected_column}",
            labels={"count": "Compte", selected_column: selected_column},
            height=380
        )
        fig.update_layout(**_get_common_layout(f"Doublons — détails: {selected_column}",
                                               selected_column, "Compte", height=380))
        fig.update_layout(xaxis_showticklabels=False, bargap=0.25)
        fig.update_traces(
            marker_line_width=0,
            hovertemplate=f"{selected_column}: %{{x}}<br>Compte: %{{y}}"
        )
        return dcc.Graph(figure=fig)
