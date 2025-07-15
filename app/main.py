import dash
from dash import dcc, html, Input, Output, State, callback
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
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY],
    suppress_callback_exceptions=True,
    prevent_initial_callbacks=True
)
server = app.server
app.title = "JunbiData"

# Construction du layout principal
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

        # Stores globaux
        dcc.Store(id="df-store", data=None),
        dcc.Store(id="filename-store", data=None),
        dcc.Store(id="active-module", data="chargement"),
        dcc.Store(id="module-cache", data={}),
        dcc.Store(id="module-status", data={
            "chargement": False,
            "visualisation": False,
            "modifications": False,
            "export": False,
            "historique": False
        }),

        html.Button("Init", id="dummy-init", style={"display": "none"}),

        html.Div(id="step-navigation", style={"marginBottom": "25px"}),
        html.Hr(style={"borderTop": "1px solid #888", "marginBottom": "25px"}),
        html.Div(id="sub-tabs-container", style={"marginBottom": "20px"}),
        html.Div(id="content", style={"minHeight": "300px", "paddingBottom": "40px"})
    ], fluid=True)

app.layout = build_layout

# Navigation principale
@callback(
    Output("step-navigation", "children"),
    Input("active-module", "data"),
    Input("dummy-init", "n_clicks"),
    State("module-status", "data")
)
def update_navigation(active_key, _, status):
    nav = []
    for i, (key, label) in enumerate(modules.items()):
        validated = status.get(key, False)
        if validated and key == active_key:
            color = "success"
        elif key == active_key:
            color = "primary"
        elif validated:
            color = "success"
        else:
            color = "light"
        nav.append(
            dbc.Button(f"{i+1}. {label}", id={"type": "main-btn", "index": key}, color=color,
                       style={"margin-top": "5px", "fontWeight": "bold"})
        )
        if i < len(modules) - 1:
            nav.append(html.Span("→", style={"fontSize": "20px", "margin": "0 8px"}))

    return html.Div(nav, style={"display": "flex", "justifyContent": "center", "alignItems" : "center", "flexWrap": "wrap"})

# Mise à jour module actif
@callback(
    Output("active-module", "data"),
    [Input({"type": "main-btn", "index": key}, "n_clicks") for key in modules],
    State("active-module", "data")
)
def switch_module(*args):
    current_module = args[-1]  # 👈 le dernier arg est bien le state "active-module"
    clicks = args[:-1]
    ctx_id = dash.callback_context.triggered_id

    if ctx_id:
        clicked_module = ctx_id["index"]
        if clicked_module != current_module:
            return clicked_module

    raise dash.exceptions.PreventUpdate

@callback(
    Output("content", "children"),
    Output("module-cache", "data"),
    Input("active-module", "data"),
    Input("df-store", "data"),
    Input("filename-store", "data"),
    Input("df-store", "modified_timestamp"),
    State("module-cache", "data"),
    State("module-status", "data")
)
def render_module_content(active_module, df_json, filename, modified_ts, module_cache, module_status):
    validated = module_status.get(active_module, False)

    # ✅ Bloc spécial pour le module chargement
    if active_module == "chargement":
        from app.modules.chargement import get_content as get_chargement

        # 🔁 Si contenu validé déjà en cache → on le garde
        if validated and "chargement" in module_cache:
            return module_cache["chargement"], module_cache

        # ✅ Sinon, on génère dynamiquement (vide ou rempli selon `validated`)
        content = get_chargement(validated, df_json if validated else None, filename if validated else None)

        # ✅ On ne met à jour le cache QUE si `validated` est True
        if validated:
            module_cache["chargement"] = content

        return content, module_cache

    # Autres modules (avec cache)
    if active_module in module_cache:
        return module_cache[active_module], module_cache

    if active_module == "visualisation":
        from app.modules.visualisation import get_content as get_visualisation
        content = get_visualisation()
    else:
        content = html.Div("Module inconnu.")

    module_cache[active_module] = content
    return content, module_cache

### Intégration des modules
# Module chargement du dataset
register_chargement(app)
