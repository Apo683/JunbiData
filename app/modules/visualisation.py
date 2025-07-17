import dash
from dash import html, dcc, dash_table, Input, Output, State, callback, callback_context
import dash_bootstrap_components as dbc
import pandas as pd
import io
import plotly.express as px

def get_content():
    return html.Div([
        dbc.Tabs(id="subtabs-visu", active_tab="completion", children=[
            dbc.Tab(label="Taux de complétion", tab_id="completion"),
            dbc.Tab(label="Distribution", tab_id="distribution"),
            dbc.Tab(label="Valeurs uniques", tab_id="uniques"),
            dbc.Tab(label="Doublons", tab_id="doublons"),
            dbc.Tab(label="Valeurs aberrantes", tab_id="outliers"),
        ]),
        html.Div([
            dcc.Dropdown(
                id="completion-display-mode",
                options=[
                    {"label": "Graphique (brut)", "value": "graph_raw"},
                    {"label": "Graphique (tri croissant)", "value": "graph_ascending"},
                    {"label": "Graphique (tri décroissant)", "value": "graph_descending"},
                    {"label": "Tableau", "value": "table"}
                ],
                value="graph_descending",
                style={
                    "display": "none",
                    "backgroundColor": "#ffffff",
                    "color": "#000",
                    "borderRadius": "5px",
                }
            ),
            html.Div(id="visu-tab-content", style={"marginTop": "20px"})
        ])
    ])

def register_callbacks_visualisation(app):
    @app.callback(
        [Output("visu-tab-content", "children"),
         Output("module-cache", "data", allow_duplicate=True),
         Output("completion-display-mode", "style"),
         Output("display-mode-store", "data"),  # Mettre à jour le store
         Output("refresh-state", "data")],
        [Input("subtabs-visu", "active_tab"),
         Input("df-store", "data"),
         Input("active-module", "data"),
         Input("completion-display-mode", "value"),
         Input("display-mode-store", "data"),  # Utiliser le store comme fallback
         Input("refresh-state", "data")],
        [State("module-cache", "data"),
         State("module-status", "data")],
        prevent_initial_call='initial_duplicate'
    )
    def update_visualisation_tab(active_tab, df_json, active_module, display_mode_input, display_mode_store, refresh, module_cache, module_status):
        print(f"DEBUG - update_visualisation_tab: active_tab={active_tab}, active_module={active_module}, df_json={df_json is not None if df_json else 'None'}, display_mode_input={display_mode_input}, display_mode_store={display_mode_store}, refresh={refresh}")
        cache = module_cache.copy() if module_cache else {}
        validated = module_status.get("visualisation", False) if module_status else False

        # Vérifier si le module visualisation est actif
        if active_module != "visualisation":
            return html.I("⚠️ Module visualisation non actif."), cache, {"display": "none"}, display_mode_store, False

        # Afficher le dropdown seulement pour le sous-onglet "completion"
        dropdown_style = {
            "width": "250px",
            "backgroundColor": "#ffffff",
            "color": "#000",
            "borderRadius": "5px",
            "display": "block" if active_tab == "completion" else "none"
        }

        # Déterminer display_mode en utilisant le store comme fallback
        ctx = callback_context
        if not ctx.triggered:
            display_mode = display_mode_store
        else:
            display_mode = display_mode_input if display_mode_input is not None else display_mode_store

        # Vérifier si le cache contient le contenu pour le mode d'affichage actuel
        cache_key = f"visualisation_{active_tab}_{display_mode}" if active_tab == "completion" else f"visualisation_{active_tab}"
        if validated and cache_key in cache:
            return cache[cache_key], cache, dropdown_style, display_mode, False

        # Si aucun dataset n'est chargé
        if df_json is None:
            content = html.I("⚠️ Aucun dataset chargé.")
        else:
            df = pd.read_json(io.StringIO(df_json), orient="split")
            if active_tab == "completion":
                percent = df.isna().mean() * 100  # Sans sort_values pour garder les données brutes
                df_percent = percent.reset_index().rename(columns={"index": "Colonne", 0: "% de NaN"})

                # Contenu avec titre, dropdown, et contenu (graphique ou tableau)
                content = html.Div([
                    html.H6("📉 Taux de complétion par colonne :"),
                    dcc.Dropdown(
                        id="completion-display-mode",
                        options=[
                            {"label": "Graphique (brut)", "value": "graph_raw"},
                            {"label": "Graphique (tri croissant)", "value": "graph_ascending"},
                            {"label": "Graphique (tri décroissant)", "value": "graph_descending"},
                            {"label": "Tableau", "value": "table"}
                        ],
                        value=display_mode,
                        style=dropdown_style
                    ),
                    html.Div(id="dynamic-content")  # Placeholder pour le graphique ou tableau
                ])

                # Générer le contenu dynamique (graphique ou tableau)
                if display_mode == "table":
                    dynamic_content = dash_table.DataTable(
                        data=df_percent.to_dict("records"),
                        columns=[{"name": i, "id": i} for i in ["Colonne", "% de NaN"]],
                        style_table={"overflowX": "auto"},
                        style_cell={"textAlign": "left", "fontSize": "14px", "backgroundColor": "#f2f2f2", "color": "#111"},
                        style_header={"backgroundColor": "#e0e0e0", "fontWeight": "bold", "color": "#000"},
                        page_size=10
                    )
                else:
                    if display_mode == "graph_ascending":
                        df_percent = df_percent.sort_values("% de NaN", ascending=True)
                    elif display_mode == "graph_descending":
                        df_percent = df_percent.sort_values("% de NaN", ascending=False)
                    # Sinon, "graph_raw" : garder l'ordre brut

                    fig = px.bar(
                        df_percent,
                        x="Colonne",
                        y="% de NaN",
                        title="Taux de complétion par colonne",
                        labels={"% de NaN": "Pourcentage de valeurs manquantes (%)"},
                        color="% de NaN",
                        color_continuous_scale="Blues",
                        height=500
                    )
                    fig.update_layout(
                        xaxis_title="Colonnes",
                        yaxis_title="Pourcentage de valeurs manquantes (%)",
                        title_x=0.5,
                        margin={"l": 40, "r": 40, "t": 60, "b": 100},
                        xaxis_tickangle=45,
                        showlegend=False,
                        plot_bgcolor="#f2f2f2",
                        paper_bgcolor="#f2f2f2",
                        font_color="#111"
                    )
                    fig.update_traces(
                        hovertemplate="Colonne: %{x}<br>% de NaN: %{y:.2f}%"
                    )
                    dynamic_content = dcc.Graph(figure=fig)

                # Mettre à jour le contenu dynamique
                content = html.Div([
                    html.H6("📉 Taux de complétion par colonne :"),
                    dcc.Dropdown(
                        id="completion-display-mode",
                        options=[
                            {"label": "Graphique (brut)", "value": "graph_raw"},
                            {"label": "Graphique (tri croissant)", "value": "graph_ascending"},
                            {"label": "Graphique (tri décroissant)", "value": "graph_descending"},
                            {"label": "Tableau", "value": "table"}
                        ],
                        value=display_mode,
                        style=dropdown_style
                    ),
                    dynamic_content
                ])
            else:
                content = html.Div(f"📊 {active_tab.capitalize()} à venir.")

        if df_json is not None:
            cache[cache_key] = content
        return content, cache, dropdown_style, display_mode, False