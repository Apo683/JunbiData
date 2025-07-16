import dash
from dash import dcc, html, Input, Output, State, callback
import dash_bootstrap_components as dbc

# Imports des modules
from app.modules.chargement import register_callbacks_chargement as register_chargement, get_content as get_chargement
from app.modules.visualisation import register_callbacks_visualisation as register_visualisation, get_content as get_visualisation

# 🔧 Définition des modules
modules = {
    "chargement": "Chargement",
    "visualisation": "Visualisation",
    "modifications": "Modifications",
    "export": "Export",
    "historique": "Historique"
}

# 🚀 Initialisation de l’application
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY], suppress_callback_exceptions=True)
server = app.server
app.title = "JunbiData"

# 🧱 Layout principal
def build_layout():
    return dbc.Container([
        html.Div([
            html.H1("JunbiData", style={"textAlign": "center", "marginTop": "40px", "marginBottom": "30px"}),
            html.Div([
                html.Span("🛠️", style={"fontSize": "20px", "marginRight": "10px"}),
                html.Span("Préparez votre jeu de données", style={"fontSize": "24px"})
            ], style={"textAlign": "left", "marginTop": "30px", "marginBottom": "15px"}),
            html.Hr(style={"borderTop": "1px solid #888", "marginBottom": "25px"})
        ]),

        # 📦 Stores partagés
        dcc.Store(id="df-store", data=None),
        dcc.Store(id="filename-store", data=None),
        dcc.Store(id="active-module", data="chargement"),
        dcc.Store(id="module-status", data={k: False for k in modules}),
        dcc.Store(id="show-upload", data=True),
        dcc.Store(id="upload-refresh", data=0),
        html.Button("Init", id="dummy-init", style={"display": "none"}),

        # 🔁 Navigation
        html.Div(id="step-navigation", style={"marginBottom": "25px"}),
        html.Hr(style={"borderTop": "1px solid #888", "marginBottom": "25px"}),

        # 📂 Modules (visibles selon le contexte)
        html.Div(id="content", children=[
            html.Div(id="module-chargement", children=get_chargement(), style={"display": "block"}),
            html.Div(id="module-visualisation", children=get_visualisation(), style={"display": "none"}),
            html.Div(id="module-modifications", children=html.Div("🔧 Module Modifications"), style={"display": "none"}),
            html.Div(id="module-export", children=html.Div("📦 Module Export"), style={"display": "none"}),
            html.Div(id="module-historique", children=html.Div("📜 Module Historique"), style={"display": "none"}),
        ], style={"minHeight": "300px", "paddingBottom": "40px"})
    ], fluid=True)

app.layout = build_layout

# 🔁 Affichage dynamique des modules visibles
@callback(
    [Output(f"module-{key}", "style") for key in modules],
    Input("active-module", "data")
)
def show_active_module(active_module):
    return [{"display": "block"} if k == active_module else {"display": "none"} for k in modules]

# 🧭 Navigation principale
@callback(
    Output("step-navigation", "children"),
    Input("active-module", "data"),
    Input("dummy-init", "n_clicks"),
    Input("module-status", "data")
)
def update_navigation(active_key, _, status):
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

@callback(
    Output("module-chargement", "children"),
    Input("show-upload", "data"),
    Input("upload-refresh", "data"),
    State("df-store", "data"),
    State("filename-store", "data")
)
def update_chargement_module(show_upload, _, df_json, filename):
    return get_chargement(show_upload=show_upload, df_json=df_json, filename=filename)

# 🔀 Changement de module actif
@callback(
    Output("active-module", "data"),
    [Input({"type": "main-btn", "index": key}, "n_clicks") for key in modules],
    State("active-module", "data")
)
def switch_module(*args):
    current = args[-1]
    clicks = args[:-1]
    ctx_id = dash.callback_context.triggered_id
    if ctx_id:
        selected = ctx_id["index"]
        if selected != current:
            return selected
    raise dash.exceptions.PreventUpdate

# 🧩 Enregistrement des callbacks propres à chaque module
register_chargement(app)
register_visualisation(app)
