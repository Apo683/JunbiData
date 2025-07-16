import dash
from dash import html, dcc, Input, Output, State, callback, ctx, dash_table
import dash_bootstrap_components as dbc
import pandas as pd
import io

# 🧩 Fonction appelée par main.py pour afficher le layout de l’onglet Visualisation
def get_content():
    return html.Div([
        dbc.Tabs(id="subtabs-visu", active_tab="completion", children=[
            dbc.Tab(label="Taux de complétion", tab_id="completion"),
            dbc.Tab(label="Distribution", tab_id="distribution"),
            dbc.Tab(label="Valeurs uniques", tab_id="uniques"),
            dbc.Tab(label="Doublons", tab_id="doublons"),
            dbc.Tab(label="Valeurs aberrantes", tab_id="outliers"),
        ]),
        html.Div(id="visu-tab-content", style={"marginTop": "20px"})
    ])

# 📦 À appeler dans main.py pour enregistrer les callbacks du module
def register_callbacks_visualisation(app):
# 🧠 Callback principal pour mettre à jour dynamiquement le contenu selon l’onglet actif
    @app.callback(
        Output("visu-tab-content", "children"),
        Output("module-status", "data", allow_duplicate=True),  # ✅ Mise à jour du statut de validation
        Input("subtabs-visu", "active_tab"),
        Input("df-store", "data"),
        State("module-status", "data"),
        prevent_initial_call=True
    )
    def update_visualisation_tab(active_tab, df_json, current_status):
        if df_json is None:
            return html.I("⚠️ Aucun dataset chargé."), dash.no_update

        # Reconstruct DataFrame
        df = pd.read_json(io.StringIO(df_json), orient="split")

        # Marque ce module comme “validé” dans le pipeline
        status = current_status.copy()
        # status["visualisation"] = True

        if active_tab == "completion":
            percent = df.isna().mean().sort_values(ascending=False) * 100
            return html.Div([
                html.H6("📉 Taux de complétion par colonne :"),
                dash_table.DataTable(
                    data=percent.reset_index().rename(columns={"index": "Colonne", 0: "% de NaN"}).to_dict("records"),
                    columns=[{"name": i, "id": i} for i in ["Colonne", "% de NaN"]],
                    style_table={"overflowX": "auto"},
                    style_cell={"textAlign": "left", "fontSize": "14px", "backgroundColor": "#f2f2f2", "color": "#111"},
                    style_header={"backgroundColor": "#e0e0e0", "fontWeight": "bold", "color": "#000"},
                    page_size=10
                )
            ]), status

        elif active_tab == "distribution":
            return html.Div("📊 Distribution à venir."), status

        elif active_tab == "uniques":
            return html.Div("🔢 Valeurs uniques à venir."), status

        elif active_tab == "doublons":
            return html.Div("🧾 Doublons à venir."), status

        return html.Div("❔ Sous-onglet inconnu."), status