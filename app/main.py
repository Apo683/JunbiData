import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
from app.modules.chargement import register_callbacks_chargement as register_chargement

# Définition des modules (clé = id, valeur = label)
modules = {
    "chargement": "Chargement",
    "visualisation": "Visualisation",
    "modifications": "Modifications",
    "export": "Export",
    "historique": "Historique"
}

# Initialisation de l’app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY], suppress_callback_exceptions=True)
server = app.server
app.title = "JunbiData"

# Stores globaux
stores = [
    dcc.Store(id="df-store", data=None),
    dcc.Store(id="active-module", data="chargement"),
    dcc.Store(id="upload-status", data=False)
]

# Navigation principale
@app.callback(
    output=dash.Output("step-navigation", "children"),
    inputs=dash.Input("active-module", "data")
)
def update_navigation(active_key):
    nav = []
    for i, (key, label) in enumerate(modules.items()):
        color = "success" if key == active_key else "light"
        btn = dbc.Button(
            f"{i+1}. {label}",
            id={"type": "main-btn", "index": key},
            color=color,
            style={"margin-top": "5px", "fontWeight": "bold"}
        )
        nav.append(btn)
        # Ajouter une flèche entre les boutons sauf après le dernier
        if i < len(modules) - 1:
            nav.append(html.Span("→", style={"fontSize": "20px", "margin": "0 8px"}))
    return html.Div(
        nav,
        style={"display": "flex", "justifyContent": "center", "alignItems": "center", "flexWrap": "wrap"}
    )

# Layout global
app.layout = dbc.Container([
    html.Div([
        html.H1("JunbiData", style={"textAlign": "center", "marginTop": "40px", "marginBottom": "30px"}),
        html.Div([
            html.Span("🛠️", style={"fontSize": "20px", "marginRight": "10px"}),
            html.Span("Préparez votre jeu de données", style={"fontSize": "24px"})
        ], style={"textAlign": "left", "marginTop": "30px", "marginBottom": "15px"}),
        html.Hr(style={"borderTop": "1px solid #888", "marginBottom": "25px"})
    ]),
    *stores,
    html.Div(update_navigation("chargement"), id="step-navigation", style={"marginBottom": "25px"}),
    html.Hr(style={"borderTop": "1px solid #888", "marginBottom": "25px"}),
    html.Div(id="sub-tabs-container", style={"marginBottom": "20px"}),
    html.Div(id="content", style={"minHeight": "300px", "paddingBottom": "40px"})
], fluid=True)

# Mise à jour module actif
@app.callback(
    dash.Output("active-module", "data"),
    [dash.Input({"type": "main-btn", "index": key}, "n_clicks") for key in modules]
)
def switch_module(*args):
    ctx_id = dash.callback_context.triggered_id
    if ctx_id:
        return ctx_id["index"]
    return  "chargement"

# Intégration des modules
register_chargement(app)
