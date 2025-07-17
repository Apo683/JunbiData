import dash
from dash import html, dcc, dash_table, Input, Output, State, callback, callback_context
import dash_bootstrap_components as dbc
import pandas as pd
import io
import plotly.express as px
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
        # Stores pour persister les modes d'affichage
        dcc.Store(id="display-mode-store-cols", data="graph_descending"),
        dcc.Store(id="display-mode-store-rows", data="graph_descending"),
        dcc.Store(id="display-mode-store-mean", data="graph"),
        
        html.H5("📊 Visualisons la qualité des données :", style={"marginBottom": "15px"}),
        dbc.Tabs(id="subtabs-visu", active_tab="completion", children=[
            dbc.Tab(label="Taux de remplissage", tab_id="completion", children=[
                html.Div([
                    html.H6("Taux de remplissage moyen (lignes vs colonnes) :"),
                    dcc.Dropdown(
                        id="completion-display-mode-mean",
                        options=[
                            {"label": "Graphique", "value": "graph"},
                            {"label": "Tableau", "value": "table"}
                        ],
                        value="graph",
                        style=STYLE_DROPDOWN
                    ),
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
                    html.Div(id="completion-rows-container", style={"marginBottom": "20px"})
                ])
            ]),
            dbc.Tab(label="Distribution", tab_id="distribution"),
            dbc.Tab(label="Valeurs uniques", tab_id="uniques"),
            dbc.Tab(label="Doublons", tab_id="doublons"),
            dbc.Tab(label="Valeurs aberrantes", tab_id="outliers"),
        ], style={"marginBottom": "20px"}),
        html.Div(id="visu-content-container", style={"marginTop": "20px"})
    ])

# 🎯 Fonction commune pour générer le layout des graphiques
def _get_common_layout(title, xaxis_title, yaxis_title, height=400, width=None):
    return {
        "title_x": 0.5,
        "margin": {"l": 30, "r": 30, "t": 50, "b": 100},
        "xaxis_tickangle": 30,
        "showlegend": False,
        "plot_bgcolor": "#f2f2f2",
        "paper_bgcolor": "#f2f2f2",
        "font_color": "#111",
        "yaxis": {"range": [0, 100]},
        "xaxis_title": xaxis_title,
        "yaxis_title": yaxis_title,
        "title": title,
        "height": height,
        "width": width
    }

# 🎯 Génération générique du contenu
def _generate_content(df, display_mode, x_col, y_col, title, xaxis_title, yaxis_title, sort_key=None, color=None, color_scale="Blues", discrete_map=None, height=400, width=None, custom_layout=None):
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
        # Créer une copie pour éviter de modifier l'original
        df_sorted = df.copy()
        # Appliquer le tri si demandé
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

# 🎯 Génération spécifique pour chaque type de visualisation
def _generate_cols_content(df, display_mode):
    nan_percent = df.isna().mean() * 100
    completion_percent = 100 - nan_percent
    df_percent = pd.DataFrame({
        'Colonne': df.columns,
        '% de remplissage': completion_percent.values
    })
    return _generate_content(df_percent, display_mode, "Colonne", "% de remplissage", "Taux de remplissage par colonne", "Colonnes", "Pourcentage de remplissage (%)", "% de remplissage")

def _generate_rows_content(df, display_mode):
    nan_percent = df.isna().mean(axis=1) * 100
    completion_percent = 100 - nan_percent
    df_percent = pd.DataFrame({
        'Ligne': [f"Ligne {i}" for i in range(len(df))],  # Conversion en string
        '% de remplissage': completion_percent.values
    })
    layout = _get_common_layout("Taux de remplissage par ligne", "Numéro de ligne", "Pourcentage de remplissage (%)", height=400)
    layout["xaxis_showticklabels"] = False
    return _generate_content(
        df_percent, display_mode, "Ligne", "% de remplissage", 
        "Taux de remplissage par ligne", "Numéro de ligne", 
        "Pourcentage de remplissage (%)", "% de remplissage", height=400, custom_layout=layout
    )

def _generate_mean_content(df, display_mode):
    # Taux de remplissage moyen par colonne : pourcentage de lignes non-NA par colonne, puis moyenne
    num_rows = len(df)
    completion_percent_cols = (df.notna().sum() / num_rows) * 100
    mean_completion_cols = completion_percent_cols.mean()
    
    # Taux de remplissage moyen par ligne : pourcentage de colonnes non-NA par ligne, puis moyenne
    num_cols = len(df.columns)
    completion_percent_rows = (df.notna().sum(axis=1) / num_cols) * 100
    mean_completion_rows = completion_percent_rows.mean()
    
    print(f"DEBUG - Moyenne des colonnes: {mean_completion_cols:.2f}%")
    print(f"DEBUG - Moyenne des lignes: {mean_completion_rows:.2f}%")

    df_mean = pd.DataFrame({
        'Type': ['Colonnes', 'Lignes'],
        '% de remplissage': [mean_completion_cols, mean_completion_rows]
    })
    
    # Layout personnalisé pour afficher la légende différemment
    layout = _get_common_layout("Comparaison des taux de remplissage", "Type de calcul", "Pourcentage de remplissage (%)", height=400, width=500)
    layout["showlegend"] = True
    layout["xaxis_tickangle"] = 0
    return _generate_content(
        df_mean, display_mode, "Type", "% de remplissage", 
        "Comparaison des taux de remplissage", 
        "Type de calcul", "Pourcentage de remplissage (%)", 
        None, "Type", None, 
        {"Colonnes": "#87cefa", "Lignes": "#4682b4"}, 
        400, 500, custom_layout=layout
    )

def register_callbacks_visualisation(app):
    # Callback principal pour la mise à jour du contenu
    @app.callback(
        [Output("completion-cols-container", "children"),
         Output("completion-rows-container", "children"),
         Output("completion-mean-container", "children"),
         Output("module-cache", "data", allow_duplicate=True)],
        [Input("df-store", "data"),
         Input("active-module", "data"),
         Input("refresh-state", "data")],
        [State("module-cache", "data"),
         State("module-status", "data"),
         State("display-mode-store-cols", "data"),
         State("display-mode-store-rows", "data"),
         State("display-mode-store-mean", "data")],
        prevent_initial_call='initial_duplicate'
    )
    def update_completion_content(df_json, active_module, refresh, module_cache, module_status, display_mode_cols, display_mode_rows, display_mode_mean):
        print(f"DEBUG - update_completion_content: active_module={active_module}, df_json={df_json is not None if df_json else 'None'}")
        
        cache = module_cache.copy() if module_cache else {}
        validated = module_status.get("visualisation", False) if module_status else False
        
        if active_module != "visualisation":
            return html.I("⚠️ Module visualisation non actif."), html.I("⚠️ Module visualisation non actif."), html.I("⚠️ Module visualisation non actif."), cache
        
        display_mode_cols = display_mode_cols if display_mode_cols else "graph_descending"
        display_mode_rows = display_mode_rows if display_mode_rows else "graph_descending"
        display_mode_mean = display_mode_mean if display_mode_mean else "graph"

        cache_key_cols = f"visualisation_completion_cols_{display_mode_cols}"
        cache_key_rows = f"visualisation_completion_rows_{display_mode_rows}"
        cache_key_mean = f"visualisation_completion_mean_{display_mode_mean}"

        if df_json is None:
            content_cols = html.I("⚠️ Aucun dataset chargé.")
            content_rows = html.I("⚠️ Aucun dataset chargé.")
            content_mean = html.I("⚠️ Aucun dataset chargé.")
        else:
            df = pd.read_json(io.StringIO(df_json), orient="split")
            if validated and cache_key_cols in cache:
                content_cols = cache[cache_key_cols]
            else:
                content_cols = _generate_cols_content(df, display_mode_cols)
                if validated:
                    cache[cache_key_cols] = content_cols
            
            if validated and cache_key_rows in cache:
                content_rows = cache[cache_key_rows]
            else:
                content_rows = _generate_rows_content(df, display_mode_rows)
                if validated:
                    cache[cache_key_rows] = content_rows
            
            if validated and cache_key_mean in cache:
                content_mean = cache[cache_key_mean]
            else:
                content_mean = _generate_mean_content(df, display_mode_mean)
                if validated:
                    cache[cache_key_mean] = content_mean

        return content_cols, content_rows, content_mean, cache

    # Callback pour gérer les changements des dropdowns
    @app.callback(
        Output("completion-cols-container", "children", allow_duplicate=True),
        Output("display-mode-store-cols", "data", allow_duplicate=True),
        Output("module-cache", "data", allow_duplicate=True),
        Input("completion-display-mode-cols", "value"),
        State("df-store", "data"),
        State("active-module", "data"),
        State("subtabs-visu", "active_tab"),
        State("module-cache", "data"),
        State("module-status", "data"),
        prevent_initial_call=True
    )
    def update_cols_display_mode(display_mode, df_json, active_module, active_tab, module_cache, module_status):
        print(f"DEBUG - update_cols_display_mode: display_mode={display_mode}")
        return _update_display_mode_helper(display_mode, df_json, active_module, active_tab, module_cache, module_status, "cols")

    @app.callback(
        Output("completion-rows-container", "children", allow_duplicate=True),
        Output("display-mode-store-rows", "data", allow_duplicate=True),
        Output("module-cache", "data", allow_duplicate=True),
        Input("completion-display-mode-rows", "value"),
        State("df-store", "data"),
        State("active-module", "data"),
        State("subtabs-visu", "active_tab"),
        State("module-cache", "data"),
        State("module-status", "data"),
        prevent_initial_call=True
    )
    def update_rows_display_mode(display_mode, df_json, active_module, active_tab, module_cache, module_status):
        print(f"DEBUG - update_rows_display_mode: display_mode={display_mode}")
        return _update_display_mode_helper(display_mode, df_json, active_module, active_tab, module_cache, module_status, "rows")

    @app.callback(
        Output("completion-mean-container", "children", allow_duplicate=True),
        Output("display-mode-store-mean", "data", allow_duplicate=True),
        Output("module-cache", "data", allow_duplicate=True),
        Input("completion-display-mode-mean", "value"),
        State("df-store", "data"),
        State("active-module", "data"),
        State("subtabs-visu", "active_tab"),
        State("module-cache", "data"),
        State("module-status", "data"),
        prevent_initial_call=True
    )
    def update_mean_display_mode(display_mode, df_json, active_module, active_tab, module_cache, module_status):
        print(f"DEBUG - update_mean_display_mode: display_mode={display_mode}")
        return _update_display_mode_helper(display_mode, df_json, active_module, active_tab, module_cache, module_status, "mean")

def _update_display_mode_helper(display_mode, df_json, active_module, active_tab, module_cache, module_status, index):
    cache = module_cache.copy() if module_cache else {}
    
    if active_module != "visualisation" or active_tab != "completion" or df_json is None:
        raise dash.exceptions.PreventUpdate
    
    df = pd.read_json(io.StringIO(df_json), orient="split")
    if index == "cols":
        content = _generate_cols_content(df, display_mode)
        cache_key = f"visualisation_completion_cols_{display_mode}"
    elif index == "rows":
        content = _generate_rows_content(df, display_mode)
        cache_key = f"visualisation_completion_rows_{display_mode}"
    elif index == "mean":
        content = _generate_mean_content(df, display_mode)
        cache_key = f"visualisation_completion_mean_{display_mode}"
    else:
        raise dash.exceptions.PreventUpdate
    
    cache[cache_key] = content
    return content, display_mode, cache