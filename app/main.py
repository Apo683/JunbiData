import dash
from dash import dcc, html, Input, Output, State, callback, Dash
import dash_bootstrap_components as dbc
import dash_uploader as du
import os
from pathlib import Path

# Import des modules
from app.modules.chargement import register_callbacks_chargement as register_chargement, get_content as get_chargement
from app.modules.visualisation import register_callbacks_visualisation as register_visualisation, get_content as get_visualisation
from app.modules.cleaning import register_callbacks_cleaning as register_cleaning, get_content as get_cleaning

from app.modules.history import get_content as get_history, register_callbacks_history as register_history

# 🔧 Définition des modules
modules = {
    "chargement": "Chargement",
    "visualisation": "Visualisation",
    "nettoyage": "Nettoyage",
    "preparation": "Préparation",
    "historique": "Historique",
    "export": "Export",
}

# 🚀 Initialisation de l'application
# app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY], suppress_callback_exceptions=True)
BASE_DIR = Path(__file__).resolve().parent

app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY], suppress_callback_exceptions=True,
    extra_hot_reload_paths=[
        str(BASE_DIR),
        str("main.py"),
        str(BASE_DIR / "modules"),
    ],
)
server = app.server

# Configuration pour les gros fichiers
server.config['MAX_CONTENT_LENGTH'] = 1024 * 1024 * 1024  # 1 Go

# Configuration de dash-uploader
upload_folder = os.path.join(os.path.abspath(os.path.dirname(__file__)), "tmp")
os.makedirs(upload_folder, exist_ok=True)
du.configure_upload(app, upload_folder, use_upload_id=False)

app.title = "JunbiData"

# 🧱 Layout principal
def build_layout():
    return dbc.Container([
        html.Div([
            html.H1("JunbiData", style={"textAlign": "center", "marginTop": "40px"}),
            html.H3("準備 Data", style={"fontWeight": "300", "textAlign": "center", "marginBottom": "20px"}),
            html.Div([
                html.Span("🛠️", style={"fontSize": "20px", "marginRight": "10px"}),
                html.Span("Préparez votre jeu de données", style={"fontSize": "24px"})
            ], style={"textAlign": "left", "marginTop": "30px", "marginBottom": "15px"}),
            html.Hr(style={"borderTop": "1px solid #888", "marginBottom": "25px"})
        ]),

        # Stores partagés
        dcc.Store(id="filename-store", data=None),
        dcc.Store(id="active-module", data="chargement"),
        dcc.Store(id="module-status", data={k: False for k in modules}),
        dcc.Store(id="error-store", data={}),
        dcc.Store(id="show-upload", data=True),
        dcc.Store(id="module-cache", data={}),
        dcc.Store(id="display-mode-store", data="graph_descending"),
        dcc.Store(id="refresh-state", data=False),
        dcc.Store(id="original-parquet-path-store", data=None),
        dcc.Store(id="df-json-store", data=None),
        dcc.Store(id="pipeline-store", data=[]),

        # Navigation
        html.Div(id="step-navigation", style={"marginBottom": "25px"}),
        html.Hr(style={"borderTop": "1px solid #888", "marginBottom": "25px"}),

        # Modules
        html.Div(id="content", children=[
            html.Div(id="module-chargement", children=get_chargement(), style={"display": "block"}),
            html.Div(id="module-visualisation", children=get_visualisation(), style={"display": "none"}),
            html.Div(id="module-nettoyage", children=get_cleaning(), style={"display": "none"}),
            # html.Div(id="module-nettoyage", children=html.Div("🧹 Module Nettoyage"), style={"display": "none"}),
            html.Div(id="module-preparation", children=html.Div("🛠️ Module Préparation"), style={"display": "none"}),
            html.Div(id="module-historique", children=get_history(), style={"display": "none"}),
            html.Div(id="module-export", children=html.Div("📦 Module Export"), style={"display": "none"}),
        ], style={"minHeight": "300px", "paddingBottom": "40px"})
    ], fluid=True)

app.layout = build_layout

# 🔄 Affichage dynamique des modules visibles
@callback(
    [Output(f"module-{key}", "style") for key in modules],
    Input("active-module", "data")
)
def show_active_module(active_module):
    return [{"display": "block"} if k == active_module else {"display": "none"} for k in modules]

# 🔄 Navigation principale
@callback(
    Output("step-navigation", "children"),
    [Input("active-module", "data"),
     Input("module-status", "data")]
)
def update_navigation(active_key, status):
    nav = []
    for i, (key, label) in enumerate(modules.items()):
        validated = status.get(key, False)
        color = "success" if validated else ("primary" if key == active_key else "light")
        nav.append(
            dbc.Button(f"{i+1}. {label}", id={"type": "main-btn", "index": key}, color=color,
                       style={"marginTop": "5px", "fontWeight": "bold"})
        )
        if i < len(modules) - 1:
            nav.append(html.Span("→", style={"fontSize": "20px", "margin": "0 8px"}))
    return html.Div(nav, style={"display": "flex", "justifyContent": "center", "alignItems": "center", "flexWrap": "wrap"})

# 🔄 Mise à jour du contenu du module chargement
@callback(
    [Output("module-chargement", "children"),
     Output("module-cache", "data"),
     Output("df-json-store", "data")],
    [Input("show-upload", "data"),
     Input("original-parquet-path-store", "data"),
     Input("filename-store", "data"),
     Input("df-json-store", "data")],
    [State("module-cache", "data"),
     State("module-status", "data"),
     State("error-store", "data")]
)
def update_chargement_module(show_upload, parquet_path, filename, df_json, module_cache, module_status, error_store):
    print(f"DEBUG - update_chargement_module: show_upload={show_upload}, parquet_path={parquet_path}, filename={filename}, df_json={df_json is not None if df_json else 'None'}, module_status={module_status}")
    if not isinstance(module_cache, dict):
        module_cache = {}
    if not isinstance(module_status, dict):
        module_status = {"chargement": False}
    if not isinstance(error_store, dict):
        error_store = {}

    cache = module_cache.copy()
    validated = module_status.get("chargement", False)
    content = get_chargement(show_upload=show_upload, parquet_path=parquet_path, filename=filename, df_json=df_json, error=error_store,cache=cache)

    return content, cache, df_json

# 🔄 Mise à jour du module actif
@callback(
    Output("active-module", "data"),
    [Input({"type": "main-btn", "index": key}, "n_clicks") for key in modules],
    [State("active-module", "data"),
     State("module-status", "data")]
)
def switch_module(*args):
    current = args[-2]
    status = args[-1]
    
    if not isinstance(status, dict):
        status = {"chargement": False}
        
    ctx_id = dash.callback_context.triggered_id
    if ctx_id:
        selected = ctx_id["index"]
        if selected != current and (selected == "chargement" or status.get("chargement", False)):
            return selected
    raise dash.exceptions.PreventUpdate

### Intégration des modules
# Module chargement du dataset
register_chargement(app)
# Module visualisation des données du dataset
register_visualisation(app)
# Module nettoyage des données
register_cleaning(app)

# Module historique
register_history(app)