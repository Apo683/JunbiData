import dash
from dash import html, dcc, Input, Output, State, dash_table
import dash_bootstrap_components as dbc
import base64
import io
import pandas as pd

def generate_upload_info(df, filename):
    return html.Div([
        html.P(f"✅ Fichier '{filename}' chargé avec succès !"),
        html.P(f"🔢 Ce jeu de données contient {df.shape[0]} lignes – {df.shape[1]} colonnes")
    ])

def show_dataset_preview(df_json):
    df = pd.read_json(io.StringIO(df_json), orient='split')
    columns = [{"name": [str(dtype), col], "id": col} for col, dtype in df.dtypes.items()]
    return html.Div([
        html.H6("🔎 Aperçu du dataset :"),
        dash_table.DataTable(
            data=df.head(10).to_dict('records'),
            columns=columns,
            merge_duplicate_headers=False,
            style_table={"overflowX": "auto"},
            style_cell={"textAlign": "left", "fontSize": "14px", "backgroundColor": "#f2f2f2", "color": "#111"},
            style_header={"backgroundColor": "#e0e0e0", "fontWeight": "bold", "color": "#000"},
            page_size=10
        )
    ])

# 🎯 Rendu dynamique de la page selon l’état d’upload
def get_content(validated, df_json, filename):
    elements = []

    # Style dynamique des composants
    show_upload = not validated
    show_reset = validated
    display_upload = {"display": "block"} if show_upload else {"display": "none"}
    display_reset = {"display": "inline-block", "marginBottom": "15px"} if show_reset else {"display": "none"}

    elements.append(html.H5("📂 Chargement du jeu de données :", style={"marginBottom": "15px"}))

    # Bouton reset (toujours présent pour les callbacks Dash)
    elements.append(dbc.Button("🔄 Réinitialiser", id="reset-upload", n_clicks=0, style=display_reset))

    # Composant upload (affiché uniquement si aucun fichier n'est encore chargé)
    elements.append(
        html.Div([
            dcc.Upload(
                id='upload-csv',
                children=html.Div([
                    html.Div("📁 Glissez-déposez un fichier CSV ici ou cliquez."),
                    html.Div("🔍 Détection auto du séparateur", style={"fontSize": "14px", "color": "#ccc", "marginTop": "8px"})
                ]),
                style={
                    'width': '100%',
                    'height': '70px',
                    'lineHeight': '20px',
                    'borderWidth': '1px',
                    'borderStyle': 'dashed',
                    'borderRadius': '5px',
                    'textAlign': 'center',
                    'marginTop': '15px',
                    'marginBottom': '10px',
                    'backgroundColor': '#1a2a3a',
                    'color': 'white',
                    'paddingTop': '10px',
                    **display_upload
                },
                multiple=False
            )
        ])
    )

    # Si dataset déjà chargé, afficher les infos
    if validated and df_json and filename:
        df = pd.read_json(io.StringIO(df_json), orient='split')
        elements.append(generate_upload_info(df, filename))
        elements.append(show_dataset_preview(df_json))

    return html.Div(elements)

# 📥 Callback – Upload / Reset
def register_callbacks_chargement(app):
    @app.callback(
        Output("df-store", "data"),
        Output("filename-store", "data"),
        Output("module-status", "data"),
        Output("module-cache", "data", allow_duplicate=True),
        Input("upload-csv", "contents"),
        Input("reset-upload", "n_clicks"),
        State("upload-csv", "filename"),
        State("module-status", "data"),
        State("module-cache", "data"),
        prevent_initial_call=True
    )
    def handle_upload_or_reset(contents, reset_clicks, upload_filename, current_status, current_cache):
        triggered = dash.callback_context.triggered_id

        status = current_status.copy()
        cache = current_cache.copy()

        # --- Réinitialisation ---
        if triggered == "reset-upload":
            if reset_clicks > 0:
                status["chargement"] = False
                cache.pop("chargement", None)
                return None, None, status, cache

        # --- Upload ---
        if triggered == "upload-csv" and contents:
            try:
                content_type, content_string = contents.split(',')
                decoded = base64.b64decode(content_string)

                # Lecture CSV avec encodage adaptatif
                encodings = [
                    "utf-8",        # standard par défaut
                    "utf-8-sig",    # BOM au début
                    "latin-1",      # très tolérant (ISO-8859-1)
                    "cp1252",       # Excel Windows (Windows-1252)
                    "ISO-8859-15",  # extension de latin-1
                    "utf-16",       # parfois utilisé dans les exports SQL Server
                    "shift_jis",    # japonais
                    "gbk"           # chinois simplifié
                ]
                for encoding in encodings:
                    try:
                        df = pd.read_csv(io.StringIO(decoded.decode(encoding)), sep=None, engine='python')
                        break
                    except UnicodeDecodeError:
                        continue
                else:
                    raise ValueError("Erreur de décodage avec UTF-8 et Latin-1.")

                status["chargement"] = True
                cache.pop("chargement", None)  # ⚠️ Remplacer le cache au prochain render
                return df.to_json(date_format='iso', orient='split'), upload_filename, status, cache

            except Exception as e:
                status["chargement"] = False
                cache.pop("chargement", None)
                return None, None, status, cache

        return dash.no_update, dash.no_update, dash.no_update, dash.no_update

