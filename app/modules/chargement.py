from dash import html, dcc, Output, Input, State
import dash
import base64
import io
import pandas as pd
from dash import dash_table


def register_callbacks_chargement(app):

    # === Callback 1 – Rendu de la page "Chargement" ===
    @app.callback(
        Output("content", "children"),
        Input("active-module", "data")
    )
    def render_chargement(active_tab):
        if active_tab != "chargement":
            return dash.no_update

        return html.Div([
            html.H5("📂 Chargement du jeu de données :", style={"marginBottom": "15px"}),
            dcc.Upload(
                id='upload-csv',
                children=html.Div([
                    html.Div("📁 Glissez-déposez un fichier CSV ici ou cliquez."),
                    html.Div("🔍 Détection auto du séparateur avec Pandas", style={
                        "fontSize": "14px",
                        "color": "#ccc",
                        "marginTop": "8px"
                    })
                ]),
                    style={
                    'width': '100%',
                    'height': '70px',
                    'lineHeight': '20px',
                    'borderWidth': '1px',
                    'borderStyle': 'dashed',
                    'borderRadius': '5px',
                    'textAlign': 'center',
                    'marginBottom': '10px',
                    'backgroundColor': '#1a2a3a',
                    'color': 'white',
                    'paddingTop': '10px'
                },
                multiple=False
            ),
            html.Div(id='upload-result', style={"marginTop": "15px"}),
            html.Div(id="data-preview", style={"marginTop": "25px"})
        ])

    # === Callback 2 – Traitement de l’upload CSV ===
    @app.callback(
        Output("upload-result", "children"),
        Output("df-store", "data"),
        Output("upload-status", "data"),
        Input("upload-csv", "contents"),
        State("upload-csv", "filename")
    )
    def handle_upload(contents, filename):
        if contents is None:
            return dash.no_update, dash.no_update, False

        try:
            content_type, content_string = contents.split(',')
            decoded = base64.b64decode(content_string)
            try:
                df = pd.read_csv(io.StringIO(decoded.decode('utf-8')), sep=None, engine='python')
            except UnicodeDecodeError:
                df = pd.read_csv(io.StringIO(decoded.decode('latin-1')), sep=None, engine='python')
            return (
                html.Div([
                    html.P(f"✅ Fichier '{filename}' chargé avec succès !"),
                    html.P(f"🔢 Ce jeu de données contient {df.shape[0]} lignes – {df.shape[1]} colonnes")
                ]),
                df.to_json(date_format='iso', orient='split'),
                True
            )
        except Exception as e:
            return html.Div(f"❌ Erreur : {str(e)}"), dash.no_update, False

    # === Callback 3 – Affichage de l’aperçu du dataset ===
    @app.callback(
        Output("data-preview", "children"),
        Input("df-store", "data")
    )
    def show_dataset_preview(df_json):
        if df_json is None:
            return html.I("Aucun fichier chargé.")
        
        df = pd.read_json(io.StringIO(df_json), orient='split')
        
        return html.Div([
            html.H6("🔎 Aperçu du dataset :"),
            dash_table.DataTable(
                data=df.head(10).to_dict('records'),
                columns=[{"name": i, "id": i} for i in df.columns],
                style_table={"overflowX": "auto"},
                style_cell={
                    "textAlign": "left",
                    "fontSize": "14px",
                    "backgroundColor": "#f2f2f2",    # gris très clair
                    "color": "#111"                  # texte noir
                },
                style_header={
                    "backgroundColor": "#e0e0e0",    # gris plus foncé pour l’en-tête
                    "fontWeight": "bold",
                    "color": "#000"
                },
                page_size=10
            )
        ])
