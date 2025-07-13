import dash
from dash import html, dcc, Input, Output, State, ctx
import dash_bootstrap_components as dbc
import base64
import io
import pandas as pd

# Modules principaux
modules = {
    "chargement": "Chargement",
    "visualisation": "Visualisation",
    "modifications": "Modifications",
    "export": "Export",
    "historique": "Historique"
}

# Sous-onglets pour le module de chargement
subtabs_chargement = [
    {"id": "upload", "label": "Chargement du jeu de données"},
]

# Sous-onglets pour Visualisation
subtabs_visualisation = [
    {"id": "completion", "label": "Taux de complétion"},
    {"id": "distribution", "label": "Distribution"},
    {"id": "uniques", "label": "Valeurs uniques"},
    {"id": "doublons", "label": "Doublons"},
]

# App Dash avec Bootstrap + exceptions
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY], suppress_callback_exceptions=True)
app.title = "JunbiData"
server = app.server

# === Layout global ===
app.layout = dbc.Container([
html.Div([
    html.H1("JunbiData", style={
        "textAlign": "center",
        "marginTop": "40px",
        "marginBottom": "30px",
        "fontWeight": "bold"
    }),
    html.Div([
        html.Span("🛠️", style={"fontSize": "20px", "marginRight": "10px"}),
        html.Span("Préparez votre jeu de données", style={"fontSize": "24px", "verticalAlign": "middle"})
    ], style={
        "textAlign": "left",
        "marginTop": "30px",
        "marginBottom": "15px"
    }),
    html.Hr(style={"borderTop": "1px solid #888", "marginBottom": "25px"})
    ]),
    dcc.Store(id="df-store", data=None),
    dcc.Store(id="active-module", data="chargement"),
    dcc.Store(id="step-status", data={key: False for key in modules}),

    html.Div(id="step-navigation"),
    html.Div(id="sub-tabs-container", style={"marginTop": "20px"}),
    html.Hr(style={"borderColor": "#ccc"}),
    html.Div(id="content", style={"minHeight": "300px", "paddingBottom": "40px"})
], fluid=True)

# === Bandeau de navigation principal personnalisé ===
@app.callback(
    Output("step-navigation", "children"),
    Input("step-status", "data"),
    State("active-module", "data")
)
def update_nav(status, active_key):
    nav = []
    for i, (key, label) in enumerate(modules.items()):
        validated = status.get(key, False)
        classes = "custom-btn"
        if validated:
            classes += " validated"
        if key == active_key:
            classes += " active"
        nav.append(html.Button(f"{i+1}. {label}", id={"type": "main-btn", "index": key}, className=classes, n_clicks=0))
        if i < len(modules) - 1:
            nav.append(html.Span("→", className="arrow"))
    return html.Div(nav, className="custom-nav")

# === Changement de module actif
@app.callback(
    Output("active-module", "data"),
    [Input({"type": "main-btn", "index": key}, "n_clicks") for key in modules]
)
def update_main_module(*args):
    ctx_triggered = ctx.triggered_id
    if ctx_triggered:
        return ctx_triggered["index"]
    return dash.no_update

# === Affichage des sous-onglets (niveau 2)
@app.callback(
    Output("sub-tabs-container", "children"),
    Input("active-module", "data")
)
def update_subtabs(module):
    if module == "visualisation":
        return dbc.Tabs(
            id="subtab",
            active_tab="completion",
            children=[dbc.Tab(label=sub["label"], tab_id=sub["id"]) for sub in subtabs_visualisation]
        )
    return html.Div()

# === Contenu dynamique selon onglet actif
@app.callback(
    Output("content", "children"),
    Input("active-module", "data"),
    Input("sub-tabs-container", "children"),
    State("step-status", "data")
)
def display_content(main_tab, subtabs_container, status):
    ctx_triggered = ctx.triggered_id
    sub_tab = "completion"  # valeur par défaut

    if main_tab == "visualisation":
        # on lit la vraie valeur depuis le DOM si présent
        triggered_inputs = dash.callback_context.inputs
        if "subtab.active_tab" in triggered_inputs:
            sub_tab = triggered_inputs["subtab.active_tab"]

    # ensuite tu continues normalement...
    status = status.copy()
    if main_tab == "chargement":
        status["chargement"] = True
        return html.Div([
            html.H5("📂 Chargement du jeu de données :", style={"marginBottom": "15px"}),
            dcc.Upload(
                id='upload-csv',
                children=html.Div(['📁 Glissez-déposez un fichier CSV ici ou cliquez.']),
                style={
                    'width': '100%',
                    'height': '60px',
                    'lineHeight': '60px',
                    'borderWidth': '1px',
                    'borderStyle': 'dashed',
                    'borderRadius': '5px',
                    'textAlign': 'center',
                    'marginBottom': '10px',
                    'backgroundColor': '#1a2a3a',
                    'color': 'white'
                },
                multiple=False
            ),
            html.Div(id='upload-result')
        ])
    elif main_tab == "visualisation":
        status["visualisation"] = True
        if sub_tab == "completion":
            return html.Div("📊 Taux de complétion")
        elif sub_tab == "distribution":
            return html.Div("📈 Distribution")
        elif sub_tab == "uniques":
            return html.Div("🔎 Valeurs uniques")
        elif sub_tab == "doublons":
            return html.Div("📑 Doublons")
        return html.Div("⚠️ Sous-onglet non reconnu.")

    elif main_tab == "modifications":
        status["modifications"] = True
        return html.Div("🧹 Modifications...")

    elif main_tab == "export":
        status["export"] = True
        return html.Div("💾 Export...")

    elif main_tab == "historique":
        status["historique"] = True
        return html.Div("📜 Historique...")

    return html.Div("❓ Rien à afficher.")

# === Callback 4 – Traitement de l’upload CSV
# @app.callback(
#     Output("upload-result", "children"),
#     Output("df-store", "data"),
#     # Output("step-status", "data"),
#     Input("upload-csv", "contents"),
#     State("upload-csv", "filename"),
#     # State("step-status", "data")
# )
# def handle_upload(content, filename, status):
#     if content is None:
#         return dash.no_update, dash.no_update, dash.no_update

#     try:
#         content_type, content_string = content.split(',')
#         decoded = base64.b64decode(content_string)
#         df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
#         status = status.copy()
#         status["chargement"] = True
#         return (
#             html.Div([
#                 html.P(f"✅ Fichier '{filename}' chargé avec succès."),
#                 html.P(f"Le dataset contient {df.shape[0]} lignes et {df.shape[1]} colonnes."),
#                 html.P("🧾 Aperçu :"),
#                 html.Pre(df.head().to_string(index=False))
#             ]),
#             df.to_json(date_format='iso', orient='split'),
#             status
#         )
#     except Exception as e:
#         return html.Div(f"❌ Erreur : {str(e)}"), dash.no_update, dash.no_update

# === Lancement de l'application
if __name__ == "__main__":
    app.run(debug=True)
