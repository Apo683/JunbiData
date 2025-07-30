import dash
from dash import html, dcc, Input, Output, State, dash_table
import dash_bootstrap_components as dbc
import base64
import io
import os
import json
import pandas as pd
from pyspark.sql import SparkSession

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
    print(f"----- Chargement du module 'chargement' avec df_json={df.head(5)} -----")
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
def get_content(show_upload=True, df_json=None, filename=None, error=None):
    print(f"get_content appelé avec df_json={df_json[:5] if df_json else 'None'}, filename={filename}, error={error}")
    upload_box = dcc.Upload(
        id="upload-file",
        children=html.Div([
            html.Div("📁 Glissez-déposez un fichier CSV ou JSON ici ou cliquez."),
            html.Div("🔍 Formats acceptés : .csv, .json (Détection auto du séparateur pour .csv, .json au format (orient='split'))", 
                     style={"fontSize": "14px", "color": "#ccc", "marginTop": "8px"}),
        ]),
        style=UPLOAD_STYLE if show_upload else {"display": "none"},
        multiple=False
    )

    additional_info = html.Div([
        generate_upload_info(pd.read_json(io.StringIO(df_json), orient="split"), filename),
        show_dataset_preview(df_json)
    ]) if df_json and filename else html.Div()

    error_display = dbc.Alert(
            error.get('chargement', ''),
            color="danger",  # Couleur choisie : "danger" pour une erreur critique
            style={"marginTop": "10px", "marginBottom": "10px", "display": "block" if error and error.get('chargement') else "none"}
        ) if error and isinstance(error, dict) else html.Div()

    reset_button_style = {"display": "inline-block" if df_json else "none", "marginBottom": "5px"}

    return html.Div([
        html.H5("📂 Chargement du jeu de données :", style={"marginBottom": "15px"}),
        dbc.Button("🔄 Réinitialiser", id="reset-upload", n_clicks=0, style=reset_button_style),
        error_display,
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
         Output("error-store", "data"),  # Store pour les erreurs avec structure dictionnaire
         Output("module-cache", "data", allow_duplicate=True)],
        [Input("upload-file", "contents"),
         Input("reset-upload", "n_clicks")],
        [State("upload-file", "filename"),
         State("module-status", "data"),
         State("module-cache", "data"),
         State("error-store", "data")],  # State pour récupérer l'erreur précédente
        prevent_initial_call='initial_duplicate'
    )
    def handle_upload_or_reset(contents, reset_clicks, filename, current_status, module_cache, current_error):
        triggered = dash.callback_context.triggered_id
        status = current_status.copy()
        cache = module_cache.copy()
        error = current_error or {}  # Initialise comme dictionnaire vide si None

        if triggered == "reset-upload" and reset_clicks and reset_clicks > 0:
            if 'spark' in globals():
                spark.stop()
            status["chargement"] = False
            cache.pop("chargement", None)
            error = {}  # Réinitialise les erreurs
            return None, None, status, True, error, cache

        if triggered == "upload-file" and contents:
            try:
                content_type, content_string = contents.split(',')
                decoded = base64.b64decode(content_string)
                file_size = len(decoded) / (1024 * 1024)

                # Traitement des fichiers JSON
                if filename and filename.lower().endswith('.json'):
                    print("Fichier JSON détecté, traitement direct.")
                    try:
                        # Tester plusieurs encodages
                        for encoding in ['utf-8', 'utf-16', 'latin-1']:
                            try:
                                print(f"Décodage avec l'encodage : {encoding}")
                                df_json_str = decoded.decode(encoding)
                                print("Décodage réussi.")
                                break
                            except UnicodeDecodeError:
                                continue
                        else:
                            raise ValueError("Aucun encodage valide trouvé pour le JSON.")
                        print("Conversion du JSON en DataFrame.")
                        json_data = json.loads(df_json_str)
                        print(f"JSON chargé avec {len(json_data)} entrées.")
                        # Tester différents formats
                        df = None
                        for orient in ["split", "records", "index", "columns", "values"]:
                            try:
                                print(f"Essai de conversion avec l'orient : {orient}")
                                df = pd.read_json(io.StringIO(df_json_str), orient=orient)
                                print(f"Format JSON détecté : {orient}")
                                break
                            except (ValueError, Exception) as e:
                                print(f"Échec avec {orient} : {str(e)}")
                                continue
                        if df is None:
                            print("Aucun format standard détecté, conversion directe en DataFrame.")
                            df = pd.DataFrame(json_data)
                        # Aplanir si nécessaire
                        nested_cols = [col for col in df.columns if df[col].apply(lambda x: isinstance(x, (dict, list))).any()]
                        if nested_cols:
                            df = pd.json_normalize(json_data)
                            print(f"Colonnes imbriquées aplanies : {nested_cols}")
                        df_json_str = df.to_json(orient="split")
                    except (json.JSONDecodeError, ValueError, Exception) as e:
                        print(f"Erreur détectée dans le parsing JSON : {str(e)}")
                        error['chargement'] = str(e)
                        return None, None, status, True, error, cache  # Stocke l'erreur spécifique

                # Traitement des fichiers CSV
                elif filename and filename.lower().endswith('.csv'):
                    for encoding in ["utf-8", "utf-8-sig", "latin-1", "cp1252"]:
                        try:
                            sample = io.StringIO(decoded.decode(encoding))
                            first_five_lines = [next(sample) for _ in range(5)] if len(decoded) > 0 else [""]
                            sample.seek(0)
                            potential_separators = [',', ';', '\t']
                            line = first_five_lines[0].strip()
                            separator_counts = {sep: line.count(sep) for sep in potential_separators}
                            separator = max(separator_counts.items(), key=lambda x: x[1])[0] if max(separator_counts.values()) > 0 else ','
                            print(f"Échantillon des 5 premières lignes : {first_five_lines}")
                            print(f"----- Séparateur détecté ----- : '{separator}'")
                            break
                        except (UnicodeDecodeError, StopIteration):
                            continue
                    else:
                        raise ValueError("Erreur d'encodage pour détecter le séparateur.")

                    if file_size < 30:
                        print("Fichier de petite taille, traitement avec Pandas.")
                        for encoding in ["utf-8", "utf-8-sig", "latin-1"]:
                            try:
                                df = pd.read_csv(io.StringIO(decoded.decode(encoding)), sep=separator, engine='python')
                                break
                            except UnicodeDecodeError:
                                continue
                        else:
                            raise ValueError("Erreur d'encodage avec Pandas.")
                        df_json_str = df.to_json(orient="split")
                    else:
                        print("Fichier de grande taille, traitement avec Spark.")
                        spark = SparkSession.builder.appName("JunbiData").master("local[*]").getOrCreate()
                        temp_file = "temp_upload.csv"
                        with open(temp_file, 'wb') as f:
                            f.write(decoded)
                        try:
                            for encoding in ["utf-8", "iso-8859-1", "us-ascii", "utf-16", "utf-16be", "utf-16le", "utf-32"]:
                                try:
                                    df = spark.read.option("encoding", encoding).option("delimiter", separator).option("header", "true").option("inferSchema", "true").csv(temp_file)
                                    print(f"Spark a lu le fichier avec l'encodage {encoding} et le séparateur '{separator}'")
                                    df.write.parquet("data/large_dataset.parquet", mode="overwrite")
                                    print("Fichier sauvegardé en Parquet.")
                                    pdf = df.toPandas()
                                    df_json_str = pdf.to_json(orient="split")
                                    break
                                except Exception as e:
                                    print(f"Erreur avec encodage {encoding}: {str(e)}")
                                    continue
                            else:
                                raise ValueError("Aucun encodage valide trouvé avec Spark.")
                        finally:
                            if os.path.exists(temp_file):
                                os.remove(temp_file)
                            spark.stop()
                else:
                    error['chargement'] = "❌ Type de fichier non pris en charge. Utilisez .csv ou .json."
                    status["chargement"] = False
                    return None, None, status, True, error, cache

                status["chargement"] = True
                return df_json_str, filename, status, False, None, cache
            except Exception as e:
                if 'spark' in globals():
                    spark.stop()
                status["chargement"] = False
                print(f"Exception globale capturée : {str(e)}")
                error['chargement'] = str(e)
                return None, None, status, True, error, cache

        raise dash.exceptions.PreventUpdate