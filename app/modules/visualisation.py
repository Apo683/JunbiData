import dash
from dash import html, dcc, dash_table, Input, Output, State, callback, callback_context
import dash_bootstrap_components as dbc
import pandas as pd
import io
import plotly.express as px

def get_content():
    return html.Div([
        html.H5("📊 Visualisons la qualité des données :", style={"marginBottom": "15px"}),
        dbc.Tabs(id="subtabs-visu", active_tab="completion", children=[
            dbc.Tab(label="Taux de remplissage", tab_id="completion"),
            dbc.Tab(label="Distribution", tab_id="distribution"),
            dbc.Tab(label="Valeurs uniques", tab_id="uniques"),
            dbc.Tab(label="Doublons", tab_id="doublons"),
            dbc.Tab(label="Valeurs aberrantes", tab_id="outliers"),
        ]),
        html.Div(id="visu-content-container", style={"marginTop": "20px"})
    ])

# 🎯 Génération du contenu pour le sous-onglet "Taux de complétion"
def _generate_completion_content(df, display_mode):
    # Calcul du pourcentage de valeurs manquantes
    nan_percent = df.isna().mean() * 100
    # Calcul du taux de complétion (inverse des valeurs manquantes)
    completion_percent = 100 - nan_percent
    
    df_percent = pd.DataFrame({
        'Colonne': df.columns,
        '% de NaN': nan_percent.values,
        '% de complétion': completion_percent.values
    })
    
    # Dropdown pour le mode d'affichage
    dropdown = dcc.Dropdown(
        id="completion-display-mode",
        options=[
            {"label": "Graphique (brut)", "value": "graph_raw"},
            {"label": "Graphique (tri croissant)", "value": "graph_ascending"},
            {"label": "Graphique (tri décroissant)", "value": "graph_descending"},
            {"label": "Tableau", "value": "table"}
        ],
        value=display_mode,
        style={
            "width": "250px",
            "backgroundColor": "#ffffff",
            "color": "#000",
            "borderRadius": "5px",
            "marginBottom": "15px"
        }
    )
    
    # Contenu dynamique selon le mode d'affichageFG
    if display_mode == "table":
        dynamic_content = dash_table.DataTable(
            data=df_percent.to_dict("records"),
            columns=[{"name": i, "id": i} for i in ["Colonne", "% de NaN", "% de complétion"]],
            style_table={"overflowX": "auto"},
            style_cell={"textAlign": "left", "fontSize": "14px", "backgroundColor": "#f2f2f2", "color": "#111"},
            style_header={"backgroundColor": "#e0e0e0", "fontWeight": "bold", "color": "#000"},
            page_size=10
        )
    else:
        if display_mode == "graph_ascending":
            df_percent = df_percent.sort_values("% de complétion", ascending=True)
        elif display_mode == "graph_descending":
            df_percent = df_percent.sort_values("% de complétion", ascending=False)
        # Sinon, "graph_raw" : garder l'ordre brut
        
        fig = px.bar(
            df_percent,
            x="Colonne",
            y="% de complétion",
            title="Taux de complétion par colonne",
            labels={"% de complétion": "Pourcentage de complétion (%)"},
            color="% de complétion",
            color_continuous_scale="Blues",
            # color_continuous_scale="Greens",
            height=500
        )
        fig.update_layout(
            xaxis_title="Colonnes",
            yaxis_title="Pourcentage de complétion (%)",
            title_x=0.5,
            margin={"l": 40, "r": 40, "t": 60, "b": 100},
            xaxis_tickangle=45,
            showlegend=False,
            plot_bgcolor="#f2f2f2",
            paper_bgcolor="#f2f2f2",
            font_color="#111",
            yaxis=dict(range=[0, 100])  # Fixer l'échelle de 0 à 100%
        )
        fig.update_traces(
            hovertemplate="Colonne: %{x}<br>% de complétion: %{y:.2f}%"
        )
        dynamic_content = dcc.Graph(figure=fig)
    
    return html.Div([
        html.H6("📉 Taux de complétion par colonne :"),
        dropdown,
        dynamic_content
    ])

def register_callbacks_visualisation(app):
    # Callback principal pour la mise à jour du contenu (sans le dropdown)
    @app.callback(
        [Output("visu-content-container", "children"),
         Output("module-cache", "data", allow_duplicate=True)],
        [Input("subtabs-visu", "active_tab"),
         Input("df-store", "data"),
         Input("active-module", "data"),
         Input("refresh-state", "data")],
        [State("module-cache", "data"),
         State("module-status", "data"),
         State("display-mode-store", "data")],
        prevent_initial_call='initial_duplicate'
    )
    def update_visualisation_content(active_tab, df_json, active_module, refresh, module_cache, module_status, display_mode_store):
        print(f"DEBUG - update_visualisation_content: active_tab={active_tab}, active_module={active_module}, df_json={df_json is not None if df_json else 'None'}")
        
        cache = module_cache.copy() if module_cache else {}
        validated = module_status.get("visualisation", False) if module_status else False
        
        # Vérifier si le module visualisation est actif
        if active_module != "visualisation":
            return html.I("⚠️ Module visualisation non actif."), cache
        
        # Utiliser la valeur du store ou la valeur par défaut
        display_mode = display_mode_store if display_mode_store else "graph_descending"
        
        # Vérifier le cache
        cache_key = f"visualisation_{active_tab}_{display_mode}" if active_tab == "completion" else f"visualisation_{active_tab}"
        if validated and cache_key in cache:
            return cache[cache_key], cache
        
        # Si aucun dataset n'est chargé
        if df_json is None:
            content = html.I("⚠️ Aucun dataset chargé.")
        else:
            df = pd.read_json(io.StringIO(df_json), orient="split")
            
            if active_tab == "completion":
                content = _generate_completion_content(df, display_mode)
            elif active_tab == "distribution":
                content = html.Div("📊 Distribution à venir.")
            elif active_tab == "uniques":
                content = html.Div("📊 Valeurs uniques à venir.")
            elif active_tab == "doublons":
                content = html.Div("📊 Doublons à venir.")
            elif active_tab == "outliers":
                content = html.Div("📊 Valeurs aberrantes à venir.")
            else:
                content = html.Div(f"📊 {active_tab.capitalize()} à venir.")
        
        # Mise en cache
        if df_json is not None:
            cache[cache_key] = content
            
        return content, cache

    # Callback séparé pour gérer les changements du dropdown (Pattern-matching)
    @app.callback(
        [Output("visu-content-container", "children", allow_duplicate=True),
         Output("display-mode-store", "data", allow_duplicate=True),
         Output("module-cache", "data", allow_duplicate=True)],
        [Input("completion-display-mode", "value")],
        [State("df-store", "data"),
         State("active-module", "data"),
         State("subtabs-visu", "active_tab"),
         State("module-cache", "data"),
         State("module-status", "data")],
        prevent_initial_call=True
    )
    def update_completion_display_mode(display_mode, df_json, active_module, active_tab, module_cache, module_status):
        print(f"DEBUG - update_completion_display_mode: display_mode={display_mode}")
        
        cache = module_cache.copy() if module_cache else {}
        
        # Vérifications de sécurité
        if active_module != "visualisation" or active_tab != "completion" or df_json is None:
            raise dash.exceptions.PreventUpdate
        
        df = pd.read_json(io.StringIO(df_json), orient="split")
        content = _generate_completion_content(df, display_mode)
        
        # Mise à jour du cache
        cache_key = f"visualisation_completion_{display_mode}"
        cache[cache_key] = content
        
        return content, display_mode, cache