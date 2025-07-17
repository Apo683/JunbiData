import dash
from dash import html, dcc, Input, Output, State, dash_table
import dash_bootstrap_components as dbc
import base64
import io
import pandas as pd

# Style du bouton upload
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

# Génération des résultat de l'upload
def generate_upload_info(df, filename):
    return html.Div([
        html.P(f"✅ Fichier '{filename}' chargé avec succès !"),
        html.P(f"🔢 Ce jeu de données contient {df.shape[0]} lignes et {df.shape[1]} colonnes")
    ])

def show_dataset_preview(df_json):
    df = pd.read_json(io.StringIO(df_json), orient='split')
    columns = [{"name": [str(dtype), col], "id": col} for col, dtype in df.dtypes.items()]
    return html.Div([
        html.H6("🔎 Aperçu du dataset :", style={"marginBottom": "15px"}),
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

# 🎯 Rendu dynamique du module chargement selon l’état d’upload
def get_content(show_upload=True, df_json=None, filename=None):
    upload_box = dcc.Upload(
        id="upload-csv",
        children=html.Div([
            html.Div("📁 Glissez-déposez un fichier CSV ici ou cliquez."),
            html.Div("🔍 Détection auto du séparateur avec Pandas", style={"fontSize": "14px", "color": "#ccc", "marginTop": "8px"})
        ]),
        style=UPLOAD_STYLE if show_upload else {"display": "none"},
        multiple=False
    )

    additional_info = html.Div([
        generate_upload_info(pd.read_json(io.StringIO(df_json), orient="split"), filename),
        show_dataset_preview(df_json)
    ]) if df_json and filename else html.Div()

    reset_button_style = {"display": "inline-block" if df_json else "none", "marginBottom": "5px"}

    return html.Div([
        html.H5("📂 Chargement du jeu de données :", style={"marginBottom": "15px"}),
        dbc.Button("🔄 Réinitialiser", id="reset-upload", n_clicks=0, style=reset_button_style),
        html.Div(id="upload-error", style={"color": "red", "marginTop": "10px", "marginBottom": "10px"}),
        upload_box,
        additional_info
    ])

# 📥 Callback – Upload / Reset
def register_callbacks_chargement(app):
    @app.callback(
        [Output("df-store", "data"),
         Output("filename-store", "data"),
         Output("module-status", "data"),
         Output("show-upload", "data"),
         Output("upload-error", "children"),
         Output("module-cache", "data", allow_duplicate=True)],
        [Input("upload-csv", "contents"),
         Input("reset-upload", "n_clicks")],
        [State("upload-csv", "filename"),
         State("module-status", "data"),
         State("module-cache", "data")],
        prevent_initial_call='initial_duplicate'
    )
    def handle_upload_or_reset(contents, reset_clicks, filename, current_status, module_cache):
        triggered = dash.callback_context.triggered_id
        status = current_status.copy()
        cache = module_cache.copy()
        print(f"DEBUG - handle_upload_or_reset: triggered={triggered}, contents={contents is not None}, reset_clicks={reset_clicks}")

        # --- Réinitialisation ---
        if triggered == "reset-upload" and reset_clicks and reset_clicks > 0:
            status["chargement"] = False
            cache.pop("chargement", None)
            return None, None, status, True, None, cache
        
        # --- Upload du fichier ---
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
                return df_json, filename, status, False, None, cache

            except Exception as e:
                status["chargement"] = False
                return None, None, status, True, html.Div(f"❌ Erreur : {str(e)}"), cache

        raise dash.exceptions.PreventUpdate