# app/modules/nettoyage.py
import dash
import dash_bootstrap_components as dbc
from dash import html, dcc, Input, Output, State
from .common.io import load_df, format_warning
from .cleaning_submodules import missing, duplicates_clean, outliers_clean, formats

from app.modules.common.ui import STYLE_DROPDOWN, OPTIONS_DROPDOWN

def get_content():
    return dbc.Tab(tab_id="nettoyage", label="Nettoyage", children=[
        html.Div([
            html.H5("🧹 Nettoyons le jeu de données :", style={"marginBottom": "15px"}),
            dbc.Tabs(id="cleaning-subtabs", active_tab="clean-missing", children=[
                missing.get_tab(),
                duplicates_clean.get_tab(),
                outliers_clean.get_tab(),
                formats.get_tab(),
            ], className="mb-3"),
            # Conteneurs par sous-onglet
            html.Div(id="cleaning-content"),
        ])
    ])

def register_callbacks_cleaning(app):
    # Route le contenu selon le sous-onglet actif
    @app.callback(
        Output("cleaning-content", "children"),
        Input("cleaning-subtabs", "active_tab"),
    )
    def render_subtab(active):
        if active == "clean-missing":
            return missing.get_layout()
        if active == "clean-duplicates":
            return duplicates_clean.get_layout()
        if active == "clean-outliers":
            return outliers_clean.get_layout()
        if active == "clean-formats":
            return formats.get_layout()
        return format_warning("Sous-onglet inconnu.")

    # Callbacks spécifiques de chaque sous-module
    missing.register_callbacks(app)
    duplicates_clean.register_callbacks(app)
    outliers_clean.register_callbacks(app)
    formats.register_callbacks(app)
