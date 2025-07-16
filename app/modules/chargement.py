import dash
from dash import html, dcc, Input, Output, State, dash_table, callback
import dash_bootstrap_components as dbc
import base64
import io
import pandas as pd
import random

UPLOAD_STYLE = {
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
    'display': 'block'
}

def generate_upload_info(df, filename):
    return html.Div([
        html.P(f"✅ Fichier '{filename}' chargé avec succès !"),
        html.P(f"🔢 Ce jeu de données contient {df.shape[0]} lignes et {df.shape[1]} colonnes")
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

# 🎯 Layout du module chargement (statique, sans paramètre)
def get_content(show_upload=True, df_json=None, filename=None):
    upload_box = dcc.Upload(
        id="upload-csv",
        children=html.Div([
            html.Div("📁 Glissez-déposez un fichier CSV ici ou cliquez."),
            html.Div("🔍 Détection auto du séparateur", style={"fontSize": "14px", "color": "#ccc", "marginTop": "8px"})
        ]),
        style=UPLOAD_STYLE if show_upload else {"display": "none"},
        multiple=False
    )

    # Contenu à afficher uniquement si upload terminé
    additional_info = html.Div([
        generate_upload_info(pd.read_json(io.StringIO(df_json), orient="split"), filename),
        show_dataset_preview(df_json)
    ]) if not show_upload and df_json and filename else html.Div()

    reset_button_style = {"display": "inline-block" if not show_upload else "none", "marginBottom": "15px"}

    return html.Div([
        html.H5("📂 Chargement du jeu de données :", style={"marginBottom": "15px"}),
        dbc.Button("🔄 Réinitialiser", id="reset-upload", n_clicks=0, style=reset_button_style),
        upload_box,
        additional_info,
        html.Div(id="upload-result", style={"marginTop": "20px"}),
        html.Div(id="upload-preview", style={"marginTop": "20px"})
    ])

# 📥 Callback – Upload / Reset
def register_callbacks_chargement(app):
    @app.callback(
        Output("df-store", "data"),
        Output("filename-store", "data"),
        Output("module-status", "data"),
        Output("upload-result", "children"),
        Output("upload-preview", "children"),
        Output("upload-csv", "style"),
        Output("show-upload", "data"),
        Output("upload-refresh", "data"),
        Input("upload-csv", "contents"),
        Input("reset-upload", "n_clicks"),
        State("upload-csv", "filename"),
        State("module-status", "data"),
        State("upload-refresh", "data"),
        prevent_initial_call=True
    )
    def handle_upload_or_reset(contents, reset_clicks, filename, current_status, upload_refresh):
        triggered = dash.callback_context.triggered_id
        status = current_status.copy()

        # --- Reset
        if triggered == "reset-upload" and reset_clicks and reset_clicks > 0:
            status.update({
                "chargement": False,
                "visualisation": False,
                "modifications": False,
                "export": False,
                "historique": False,
                "_refresh": random.random()
            })
            return None, None, status, None, None, UPLOAD_STYLE, True, upload_refresh + 1

        # --- Upload
        if triggered == "upload-csv" and contents:
            try:
                content_type, content_string = contents.split(',')
                decoded = base64.b64decode(content_string)

                for encoding in ["utf-8", "utf-8-sig", "latin-1", "cp1252", "ISO-8859-15", "utf-16", "shift_jis", "gbk"]:
                    try:
                        df = pd.read_csv(io.StringIO(decoded.decode(encoding)), sep=None, engine='python')
                        break
                    except UnicodeDecodeError:
                        continue
                else:
                    raise ValueError("Erreur d'encodage sur tous les formats testés.")

                status["chargement"] = True
                df_json = df.to_json(date_format='iso', orient='split')

                return (
                    df_json,
                    filename,
                    status,
                    generate_upload_info(df, filename),
                    show_dataset_preview(df_json),
                    {"display": "inline-block", "marginBottom": "15px"},
                    False,
                    dash.no_update
                )

            except Exception as e:
                status["chargement"] = False
                return None, None, status, html.Div(f"❌ Erreur : {str(e)}"), None, {"display": "block"}, dash.no_update, dash.no_update

        raise dash.exceptions.PreventUpdate