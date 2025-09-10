# modules/visualisation.py
# Manager global du module Visualisation: assemble le layout et enregistre les callbacks
import dash
from dash import html, dcc, Input, Output, State, no_update
import dash_bootstrap_components as dbc

import plotly.io as pio
import plotly.graph_objects as go

# Sous-modules
from .visualisation_submodules.completion import get_layout as completion_layout, register_callbacks as register_completion
from .visualisation_submodules.distribution import get_layout as distribution_layout, register_callbacks as register_distribution
from .visualisation_submodules.uniques import get_layout as uniques_layout, register_callbacks as register_uniques
from .visualisation_submodules.doublons import (get_layout as duplicates_layout, register_callbacks as register_doublons)
from .visualisation_submodules.outliers import get_layout as outliers_layout, register_callbacks as register_outliers

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
MODULE_KEY = "visualisation"

def _register_plotly_template():
    # Point de départ stable
    base = pio.templates["plotly_white"]
    custom = go.layout.Template(base)  # copie du template existant

    # Personnalisations légères (facultatives)
    custom.layout.font = dict(family="Inter, system-ui, -apple-system, Segoe UI, Roboto, Arial", size=12)
    custom.layout.margin = dict(l=50, r=20, t=60, b=30)

    # Nettoyage prudent: si des traces bar existent dans le template,
    # ne pas forcer un pattern au niveau du template (source d'instabilité).
    bars = getattr(custom.data, "bar", ())  # <- au lieu de .get(...)
    for t in bars:
        if hasattr(t, "marker") and hasattr(t.marker, "pattern"):
            t.marker.pattern = None

    pio.templates["junbi"] = custom
    pio.templates.default = "junbi"

# Enregistrer au chargement du module
_register_plotly_template()
SAFE_TEMPLATE_NAME = "junbi"

def get_content():
    # Stores de contrôle propres au module
    return html.Div([
        dcc.Store(id="display-mode-store-cols", data="graph_descending"),
        dcc.Store(id="display-mode-store-rows", data="graph_descending"),
        dcc.Store(id="selected-columns-store", data=[]),
        dcc.Store(id="row-page-store", data=0),
        html.H5("📊 Visualisons la qualité des données :", style={"marginBottom": "15px"}),
        dbc.Tabs(id="subtabs-visu", active_tab="completion", children=[
            completion_layout(STYLE_DROPDOWN, OPTIONS_DROPDOWN),
            distribution_layout(),
            uniques_layout(STYLE_DROPDOWN, OPTIONS_DROPDOWN),
            duplicates_layout(STYLE_DROPDOWN, OPTIONS_DROPDOWN),
            outliers_layout(STYLE_DROPDOWN, OPTIONS_DROPDOWN),
        ], style={"marginBottom": "20px"}),
        html.Div(id="visu-content-container", style={"marginTop": "20px"})
    ])

def register_callbacks_visualisation(app):
    # Enregistrement des callbacks de chaque sous-module
    register_completion(app)
    register_distribution(app)
    register_uniques(app)
    register_doublons(app)
    register_outliers(app)

    # Validation du module dès qu'il est activé dans la nav principale
    @app.callback(
        Output("module-status", "data", allow_duplicate=True),
        Input("active-module", "data"),
        State("module-status", "data"),
        prevent_initial_call=True,  # évite de tirer au 1er render
    )
    def _mark_visualisation_valid(active_module, status):
        if active_module != MODULE_KEY:
            raise dash.exceptions.PreventUpdate
        status = status or {}
        if status.get(MODULE_KEY) is True:
            raise dash.exceptions.PreventUpdate
        return {**status, MODULE_KEY: True}