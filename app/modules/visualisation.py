import dash
from dash import html, dcc, dash_table, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
from functools import reduce
from operator import add

try:
    import pyspark  # noqa: F401
    from pyspark.sql import functions as F
    from pyspark.sql.functions import col, count, when, monotonically_increasing_id, row_number
    from pyspark.sql.window import Window
    HAS_SPARK = True
except Exception:
    F = None
    col = count = when = monotonically_increasing_id = row_number = None
    Window = None
    HAS_SPARK = False

from app.modules import spark_utils

STYLE_DROPDOWN = {
    "width": "250px",
    "backgroundColor": "#ffffff",
    "color": "#000",
    "borderRadius": "5px",
    "marginBottom": "15px"
}
OPTIONS_DROPDOWN = [
    {"label": "Graphique (brut)", "value": "graph_raw"},
    {"label": "Graphique (tri croissant)", "value": "graph_ascending"},
    {"label": "Graphique (tri décroissant)", "value": "graph_descending"},
    {"label": "Tableau", "value": "table"}
]


def get_content():
    return html.Div([
        dcc.Store(id="display-mode-store-cols", data="graph_descending"),
        dcc.Store(id="display-mode-store-rows", data="graph_descending"),
        dcc.Store(id="selected-columns-store", data=[]),
        dcc.Store(id="row-page-store", data=0),
        html.H5("📊 Visualisons la qualité des données :", style={"marginBottom": "15px"}),
        dbc.Tabs(id="subtabs-visu", active_tab="completion", children=[
            dbc.Tab(label="Taux de remplissage", tab_id="completion", children=[
                html.Div([
                    html.Div(id="completion-mean-container", style={"textAlign": "left", "marginBottom": "20px"})
                ]),
                html.Div([
                    html.H6("Taux de remplissage par colonne :"),
                    dcc.Dropdown(
                        id="completion-display-mode-cols",
                        options=OPTIONS_DROPDOWN,
                        value="graph_descending",
                        style=STYLE_DROPDOWN
                    ),
                    html.Div(id="completion-cols-container", style={"marginBottom": "20px"})
                ]),
                html.Div([
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
                            value=None,
                            placeholder="Aller à une page…",
                            style = {"width": "150px", "display": "inline-grid", "backgroundColor": "#ffffff", "color": "#000", "borderRadius": "5px"}
                        )
                    ], style={"marginBottom": "10px"}),
                    html.Div(id="completion-rows-container", style={"marginBottom": "20px"})
                ])
            ]),
            dbc.Tab(label="Distribution", tab_id="distribution", children=[
                html.Div([
                    html.H6("Sélectionnez les colonnes pour afficher leur distribution :"),
                    html.Div([
                        dbc.Checklist(
                            id="column-selection-checklist",
                            options=[],
                            value=[],
                            inline=True,
                            style={
                                "display": "flex",
                                "flexWrap": "wrap",
                                "justifyContent": "flex-start"
                            },
                            labelStyle={
                                "width": "230px",
                                "textOverflow": "ellipsis",
                                "overflow": "hidden",
                                "whiteSpace": "nowrap",
                                "display": "inline-block"
                            }
                        )
                    ], style={"marginBottom": "20px"}),
                    html.Div(id="distribution-container", style={"marginTop": "20px"})
                ])
            ]),
            dbc.Tab(label="Valeurs uniques", tab_id="uniques"),
            dbc.Tab(label="Doublons", tab_id="doublons"),
            dbc.Tab(label="Valeurs aberrantes", tab_id="outliers"),
        ], style={"marginBottom": "20px"}),
        html.Div(id="visu-content-container", style={"marginTop": "20px"})
    ])


def _get_common_layout(title, xaxis_title, yaxis_title, height=400, width=None, is_distribution=False):
    layout = {
        "title_x": 0.5,
        "margin": {"l": 30, "r": 30, "t": 50, "b": 100},
        "xaxis_tickangle": 30,
        "showlegend": False,
        "plot_bgcolor": "#f2f2f2",
        "paper_bgcolor": "#f2f2f2",
        "font_color": "#111",
        "font_size": 12,
        "yaxis": {"range": [0, 100] if "Pourcentage" in yaxis_title else None},
        "xaxis_title": xaxis_title,
        "yaxis_title": yaxis_title,
        "title": title,
        "height": height,
        "width": width
    }
    if is_distribution:
        layout["xaxis_showticklabels"] = False
    return layout


def _generate_content(df, display_mode, x_col, y_col, title, xaxis_title, yaxis_title, sort_key=None, color=None, color_scale="Plotly3", discrete_map=None, height=400, width=None, custom_layout=None):
    if display_mode == "table":
        return dash_table.DataTable(
            data=df.to_dict("records"),
            columns=[{"name": i, "id": i} for i in [x_col, y_col]],
            style_table={"overflowX": "auto"},
            style_cell={"textAlign": "left", "fontSize": "14px", "backgroundColor": "#f2f2f2", "color": "#111"},
            style_header={"backgroundColor": "#e0e0e0", "fontWeight": "bold", "color": "#000"},
            page_size=10
        )
    else:
        df_sorted = df.copy()
        if sort_key and display_mode != "graph_raw":
            df_sorted = df_sorted.sort_values(sort_key, ascending=(display_mode == "graph_ascending"))

        fig = px.bar(
            df_sorted,
            x=x_col,
            y=y_col,
            title=title,
            labels={y_col: yaxis_title, x_col: xaxis_title},
            color=color if color else y_col,
            color_continuous_scale=color_scale if not discrete_map else None,
            color_discrete_map=discrete_map,
            height=height
        )
        fig.update_layout(**custom_layout if custom_layout else _get_common_layout(title, xaxis_title, yaxis_title, height, width))
        fig.update_traces(
            hovertemplate=f"{x_col}: %{{x}}<br>{y_col}: %{{y:.2f}}%"
        )
        return dcc.Graph(figure=fig)


def _generate_cols_content(spark_df, display_mode, is_spark=False):
    if is_spark:
        total_rows = spark_df.count()
        nan_counts = spark_df.select([count(when(col(c).isNull(), c)).alias(c) for c in spark_df.columns])
        completion_percent = nan_counts.select(
            [(100 * (total_rows - col(c)) / total_rows).alias(c) for c in spark_df.columns]
        ).toPandas().melt(var_name='Colonne', value_name='% de remplissage')
    else:
        nan_percent = spark_df.isna().mean() * 100
        completion_percent = 100 - nan_percent
        completion_percent = pd.DataFrame({
            'Colonne': spark_df.columns,
            '% de remplissage': completion_percent.values
        })

    layout = _get_common_layout("Taux de remplissage par colonne", "Colonnes", "Pourcentage de remplissage (%)", height=400)
    layout["xaxis_showticklabels"] = False
    return _generate_content(completion_percent, display_mode, "Colonne", "% de remplissage", "Taux de remplissage par colonne", "Colonnes", "Pourcentage de remplissage (%)", "% de remplissage", height=400, custom_layout=layout)


def _generate_rows_content(spark_df, display_mode, page=0, page_size=1000, is_spark=False):
    start_idx = page * page_size
    if is_spark:
        total_rows = spark_df.count()
        if total_rows == 0:
            df_percent = pd.DataFrame({"Ligne": [], "% de remplissage": []})
        else:
            cols = [c for c in spark_df.columns]
            num_columns = len(cols)
            if num_columns == 0:
                df_percent = pd.DataFrame({"Ligne": [], "% de remplissage": []})
            else:
                df_idx = spark_df.withColumn("_orig_id", monotonically_increasing_id())
                indicators = [when(col(c).isNotNull(), 1).otherwise(0) for c in cols]
                filled_count_expr = reduce(add, indicators)
                df_pct = df_idx.withColumn("% de remplissage", (filled_count_expr / F.lit(num_columns)) * 100)

                if display_mode == "graph_ascending" or display_mode == "table":
                    order_cols = [col("% de remplissage").asc(), col("_orig_id").asc()]
                elif display_mode == "graph_descending":
                    order_cols = [col("% de remplissage").desc(), col("_orig_id").asc()]
                else:
                    order_cols = [col("_orig_id").asc()]

                w = Window.orderBy(*order_cols)
                df_ranked = df_pct.select(col("_orig_id"), col("% de remplissage")).withColumn("rn", row_number().over(w))
                df_page = df_ranked.where((col("rn") > start_idx) & (col("rn") <= start_idx + page_size)).orderBy(col("rn").asc())
                pdf = df_page.select("rn", "% de remplissage").toPandas()
                pdf["Ligne"] = pdf["rn"].apply(lambda x: f"Ligne {int(x)}")
                df_percent = pdf[["Ligne", "% de remplissage"]]
    else:
        total_rows = len(spark_df)
        if total_rows == 0:
            df_percent = pd.DataFrame({"Ligne": [], "% de remplissage": []})
        else:
            num_columns = spark_df.shape[1]
            if num_columns == 0:
                completion_percent_all = pd.Series([0.0] * total_rows)
            else:
                completion_percent_all = (spark_df.notna().sum(axis=1) / num_columns) * 100

            if display_mode == "graph_ascending" or display_mode == "table":
                order = completion_percent_all.sort_values(ascending=True)
            elif display_mode == "graph_descending":
                order = completion_percent_all.sort_values(ascending=False)
            else:
                order = completion_percent_all

            ordered_df = pd.DataFrame({"% de remplissage": order})
            start = start_idx
            end = min(start_idx + page_size, total_rows)
            page_df = ordered_df.iloc[start:end].copy()
            page_df["Ligne"] = [f"Ligne {i}" for i in range(start + 1, start + 1 + len(page_df))]
            df_percent = page_df[["Ligne", "% de remplissage"]]

    layout = _get_common_layout("Taux de remplissage par ligne", "Numéro de ligne", "Pourcentage de remplissage (%)", height=400)
    layout["xaxis_showticklabels"] = False
    return _generate_content(
        df_percent, display_mode, "Ligne", "% de remplissage",
        "Taux de remplissage par ligne", "Numéro de ligne",
        "Pourcentage de remplissage (%)", "% de remplissage", height=400, custom_layout=layout
    )


def _generate_mean_content(spark_df, is_spark=False):
    if is_spark:
        total_rows = spark_df.count()
        completion_percent_cols = spark_df.select(
            [(count(when(col(c).isNotNull(), c)) / total_rows * 100).alias(c) for c in spark_df.columns]
        ).toPandas().mean().mean()
    else:
        num_rows = len(spark_df)
        completion_percent_cols = (spark_df.notna().sum() / num_rows) * 100
        completion_percent_cols = completion_percent_cols.mean()
    return html.Div([
        html.P(f"Taux de remplissage moyen : {completion_percent_cols:.2f}%", style={"fontSize": "18px"})
    ])


def _generate_distribution_content(spark_df, selected_columns, display_mode="graph_raw", is_spark=False):
    if not selected_columns or (not is_spark and getattr(spark_df, 'empty', False)):
        return html.Div("Aucune colonne sélectionnée ou dataset vide.")

    graphs = []
    for column_name in selected_columns:
        if column_name in (spark_df.columns if not is_spark else spark_df.schema.names):
            if is_spark:
                dtype = spark_df.select(column_name).dtypes[0][1]
                if dtype.startswith('double') or dtype.startswith('int'):
                    min_val, max_val = spark_df.select(column_name).agg(F.min(column_name), F.max(column_name)).collect()[0]
                    bucket_width = (max_val - min_val) / 30 if (max_val - min_val) != 0 else 1
                    buckets = spark_df.select(
                        ((col(column_name) - min_val) / bucket_width).cast("int").alias("bucket")
                    ).groupBy("bucket").count()
                    df_plot = buckets.toPandas()
                    df_plot['bucket'] = df_plot['bucket'] * bucket_width + min_val
                    fig = px.histogram(
                        df_plot,
                        x='bucket',
                        y='count',
                        title=f"Distribution centrée de {column_name}",
                        nbins=30,
                        height=300
                    )
                    fig.update_layout(_get_common_layout(f"Distribution centrée de {column_name}", column_name, "Compte", is_distribution=True))
                else:
                    value_counts = spark_df.groupBy(column_name).count().orderBy(F.desc("count")).toPandas()
                    value_counts.columns = [column_name, "Compte"]
                    fig = px.bar(
                        value_counts,
                        x=column_name,
                        y="Compte",
                        title=f"Distribution de {column_name}",
                        height=300
                    )
                    fig.update_layout(_get_common_layout(f"Distribution de {column_name}", column_name, "Compte", is_distribution=True))
            else:
                if pd.api.types.is_numeric_dtype(spark_df[column_name]):
                    df_plot = pd.DataFrame({column_name: spark_df[column_name]})
                    fig = px.histogram(
                        df_plot,
                        x=column_name,
                        title=f"Distribution centrée de {column_name}",
                        nbins=30,
                        height=300
                    )
                    fig.update_layout(_get_common_layout(f"Distribution centrée de {column_name}", column_name, "Compte", is_distribution=True))
                else:
                    value_counts = spark_df[column_name].value_counts().reset_index()
                    value_counts.columns = [column_name, "Compte"]
                    value_counts = value_counts.sort_values("Compte", ascending=False)
                    fig = px.bar(
                        value_counts,
                        x=column_name,
                        y="Compte",
                        title=f"Distribution de {column_name}",
                        height=300
                    )
                    fig.update_layout(_get_common_layout(f"Distribution de {column_name}", column_name, "Compte", is_distribution=True))
            fig.update_traces(hovertemplate=f"{column_name}: %{{x}}<br>Compte: %{{y}}")
            graphs.append(dcc.Graph(figure=fig))

    return html.Div(graphs) if graphs else html.Div("Aucune distribution générée.")


def register_callbacks_visualisation(app):
    @app.callback(
        [Output("completion-cols-container", "children"),
         Output("completion-rows-container", "children"),
         Output("completion-mean-container", "children"),
         Output("distribution-container", "children"),
         Output("column-selection-checklist", "options"),
         Output("module-cache", "data", allow_duplicate=True),
         Output("page-select-rows", "options"),
         Output("page-select-rows", "value"),
         Output("row-page-store", "data")],
        [Input("parquet-path-store", "data"),
         Input("active-module", "data"),
         Input("refresh-state", "data"),
         Input("column-selection-checklist", "value"),
         Input("prev-page-rows", "n_clicks"),
         Input("next-page-rows", "n_clicks"),
         Input("page-select-rows", "value")],
        [State("module-cache", "data"),
         State("module-status", "data"),
         State("display-mode-store-cols", "data"),
         State("display-mode-store-rows", "data"),
         State("selected-columns-store", "data"),
         State("row-page-store", "data")],
        prevent_initial_call=True
    )
    def update_content(parquet_path, active_module, refresh, selected_columns, prev_clicks, next_clicks, page_select_value, module_cache, module_status, display_mode_cols, display_mode_rows, stored_selected_columns, current_page):
        cache = module_cache.copy() if module_cache else {}
        validated = module_status.get("visualisation", False) if module_status else False

        if active_module != "visualisation":
            return (html.I("⚠️ Module visualisation non actif."),) * 4 + ([], cache, [], None, 0)

        display_mode_cols = display_mode_cols or "graph_descending"
        display_mode_rows = display_mode_rows or "graph_descending"
        selected_columns = selected_columns or stored_selected_columns
        page_size = 1000

        ctx = dash.callback_context
        if ctx.triggered and "prev-page-rows" in ctx.triggered[0]["prop_id"]:
            current_page = max(0, (current_page or 0) - 1)
        elif ctx.triggered and "next-page-rows" in ctx.triggered[0]["prop_id"]:
            current_page = (current_page or 0) + 1
        elif ctx.triggered and "page-select-rows" in ctx.triggered[0]["prop_id"]:
            if page_select_value is not None:
                current_page = max(0, int(page_select_value) - 1)

        if parquet_path is None:
            return (html.I("⚠️ Aucun dataset chargé."),) * 4 + ([], cache, [], None, 0)

        is_spark = False
        if isinstance(parquet_path, str) and "large_dataset_parquet" in parquet_path and HAS_SPARK:
            spark = spark_utils.get_spark_session()
            df = spark.read.parquet(parquet_path)
            is_spark = True
        else:
            df = pd.read_parquet(parquet_path)

        total_rows = df.count() if is_spark else len(df)
        total_pages = (total_rows + page_size - 1) // page_size
        current_page = min(max(0, current_page or 0), max(0, total_pages - 1))

        cache_key_cols = f"visualisation_completion_cols_{display_mode_cols}"
        cache_key_rows = f"visualisation_completion_rows_{display_mode_rows}_{current_page}"

        if validated and cache_key_cols in cache:
            content_cols = cache[cache_key_cols]
        else:
            content_cols = _generate_cols_content(df, display_mode_cols, is_spark)
            if validated:
                cache[cache_key_cols] = content_cols

        if validated and cache_key_rows in cache:
            content_rows = cache[cache_key_rows]
        else:
            content_rows = _generate_rows_content(df, display_mode_rows, current_page, page_size, is_spark)
            if validated:
                cache[cache_key_rows] = content_rows

        content_mean = _generate_mean_content(df, is_spark)
        content_distribution = _generate_distribution_content(df, selected_columns, is_spark=is_spark)
        column_options = [{"label": col, "value": col} for col in (df.columns if not is_spark else df.schema.names)]

        page_options = [{"label": f"Page {i}", "value": i} for i in range(1, total_pages + 1)]
        page_value = (current_page + 1) if total_pages > 0 else None

        return content_cols, content_rows, content_mean, content_distribution, column_options, cache, page_options, page_value, current_page

    @app.callback(
        [Output("completion-cols-container", "children", allow_duplicate=True),
         Output("display-mode-store-cols", "data", allow_duplicate=True),
         Output("module-cache", "data", allow_duplicate=True)],
        Input("completion-display-mode-cols", "value"),
        [State("parquet-path-store", "data"),
         State("active-module", "data"),
         State("subtabs-visu", "active_tab"),
         State("module-cache", "data"),
         State("module-status", "data")],
        prevent_initial_call=True
    )
    def update_cols_display_mode(display_mode, parquet_path, active_module, active_tab, module_cache, module_status):
        return _update_display_mode_helper(display_mode, parquet_path, active_module, active_tab, module_cache, module_status, "cols")

    @app.callback(
        [Output("completion-rows-container", "children", allow_duplicate=True),
         Output("display-mode-store-rows", "data", allow_duplicate=True),
         Output("module-cache", "data", allow_duplicate=True),
         Output("row-page-store", "data", allow_duplicate=True)],
        Input("completion-display-mode-rows", "value"),
        [State("parquet-path-store", "data"),
         State("active-module", "data"),
         State("subtabs-visu", "active_tab"),
         State("module-cache", "data"),
         State("module-status", "data"),
         State("row-page-store", "data")],
        prevent_initial_call=True
    )
    def update_rows_display_mode(display_mode, parquet_path, active_module, active_tab, module_cache, module_status, current_page):
        content, display_mode, cache = _update_display_mode_helper(display_mode, parquet_path, active_module, active_tab, module_cache, module_status, "rows", current_page)
        return content, display_mode, cache, current_page


def _update_display_mode_helper(display_mode, parquet_path, active_module, active_tab, module_cache, module_status, index, current_page=0):
    cache = module_cache.copy() if module_cache else {}
    if active_module != "visualisation" or active_tab != "completion" or parquet_path is None:
        raise dash.exceptions.PreventUpdate
    page_size = 1000
    is_spark = False
    if isinstance(parquet_path, str) and "large_dataset_parquet" in parquet_path and HAS_SPARK:
        spark = spark_utils.get_spark_session()
        df = spark.read.parquet(parquet_path)
        is_spark = True
    else:
        df = pd.read_parquet(parquet_path)
    if index == "cols":
        content = _generate_cols_content(df, display_mode, is_spark)
        cache_key = f"visualisation_completion_cols_{display_mode}"
    elif index == "rows":
        content = _generate_rows_content(df, display_mode, current_page, page_size, is_spark)
        cache_key = f"visualisation_completion_rows_{display_mode}_{current_page}"
    else:
        raise dash.exceptions.PreventUpdate
    cache[cache_key] = content
    return content, display_mode, cache
