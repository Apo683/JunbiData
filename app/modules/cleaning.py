# app/modules/nettoyage.py
import dash
import dash_bootstrap_components as dbc
from dash import html, dcc, Input, Output, State
from .common.io import load_df, format_warning
from .cleaning_submodules import missing_values, duplicates, outliers, formats

from app.modules.common.ui import STYLE_DROPDOWN


def get_content():
    # Sous-modules
    from .cleaning_submodules.formats import get_layout as formats_layout

    return dbc.Tab(tab_id="nettoyage", label="Nettoyage", children=[
        html.Div([
            html.H5("🧹 Nettoyons le jeu de données :", style={"marginBottom": "15px"}),
            dbc.Tabs(id="cleaning-subtabs", active_tab="clean-formats", children=[
                formats.get_tab(),
                missing_values.get_tab(),
                duplicates.get_tab(),
                outliers.get_tab(),
            ], style={"marginBottom": "20px"}, className="mb-3"),
            # Conteneurs par sous-onglet
            html.Div(id="cleaning-content", style={"marginTop": "20px"}),
        ])
    ])

def register_callbacks_cleaning(app):
    # Sous-modules
    from app.modules.cleaning_submodules.missing_values import get_layout as missing_layout, register_callbacks as register_missing
    from app.modules.cleaning_submodules.duplicates import get_layout as duplicates_clean_layout, register_callbacks as register_duplicates_clean
    from app.modules.cleaning_submodules.outliers import get_layout as outliers_clean_layout, register_callbacks as register_outliers_clean
    from app.modules.cleaning_submodules.formats import get_layout as formats_layout, register_callbacks as register_formats

    # Route le contenu selon le sous-onglet actif
    @app.callback(
        Output("cleaning-content", "children"),
        Input("cleaning-subtabs", "active_tab"),
    )
    def render_subtab(active):
        if active == "clean-formats":
            return formats_layout(STYLE_DROPDOWN)
        if active == "clean-missing":
            return missing_layout(STYLE_DROPDOWN)
        if active == "clean-duplicates":
            return duplicates_clean_layout()
        if active == "clean-outliers":
            return outliers_clean_layout()
        
        return format_warning("Sous-onglet inconnu.")

    # Callbacks spécifiques de chaque sous-module
    register_formats(app)
    register_missing(app)
    register_duplicates_clean(app)
    register_outliers_clean(app)
