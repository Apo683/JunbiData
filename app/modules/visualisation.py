# modules/visualisation.py
# Manager global du module Visualisation: assemble le layout et enregistre les callbacks
from dash import html, dcc
import dash_bootstrap_components as dbc

# Sous-modules
from .visualisation_submodules.completion import get_layout as completion_layout, register_callbacks as register_completion
from .visualisation_submodules.distribution import get_layout as distribution_layout, register_callbacks as register_distribution
from .visualisation_submodules.uniques import get_layout as uniques_layout, register_callbacks as register_uniques

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
        ], style={"marginBottom": "20px"}),
        html.Div(id="visu-content-container", style={"marginTop": "20px"})
    ])

def register_callbacks_visualisation(app):
    # Enregistrement des callbacks de chaque sous-module
    register_completion(app)
    register_distribution(app)
    register_uniques(app)
