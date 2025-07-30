import dash
from dash import html, dcc, Input, Output, State, dash_table
import dash_bootstrap_components as dbc
import base64
import io
import os
import shutil
import json
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, ArrayType

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
    if os.path.exists("data/large_dataset.parquet"):
        spark = SparkSession.builder.appName("JunbiData").master("local[*]").getOrCreate()
        try:
            parquet_df = spark.read.parquet("data/large_dataset.parquet")
            row_count = parquet_df.count()
            col_count = len(parquet_df.columns)
            spark.stop()
        except Exception as e:
            print(f"Erreur lors de la lecture du Parquet : {str(e)}")
            row_count = df.shape[0]
            col_count = df.shape[1]
    else:
        row_count = df.shape[0]
        col_count = df.shape[1]
    
    return html.Div([
        html.P(f"✅ Fichier '{filename}' chargé avec succès !"),
        html.P(f"🔢 Ce jeu de données contient {row_count} lignes et {col_count} colonnes")
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
    ], style={"marginTop": "10px"}) if df_json and filename else html.Div()

    error_display = dbc.Alert(
        error.get('chargement', ''),
        color="danger",
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
         Output("error-store", "data"),
         Output("module-cache", "data", allow_duplicate=True)],
        [Input("upload-file", "contents"),
         Input("reset-upload", "n_clicks")],
        [State("upload-file", "filename"),
         State("module-status", "data"),
         State("module-cache", "data"),
         State("error-store", "data")],
        prevent_initial_call=True  # Changé pour éviter l'exécution initiale
    )
    def handle_upload_or_reset(contents, reset_clicks, filename, current_status, module_cache, current_error):
        triggered = dash.callback_context.triggered_id
        status = current_status.copy()
        cache = module_cache.copy()
        error = current_error or {}

        if triggered == "reset-upload" and reset_clicks and reset_clicks > 0:
            if 'spark' in globals():
                spark.stop()
            status["chargement"] = False
            cache.pop("chargement", None)
            error = {}
            if os.path.exists("data/large_dataset.parquet"):
                shutil.rmtree("data/large_dataset.parquet")  # Suppression du Parquet
            return [None, None, status, True, error, cache]  # Retour explicite comme liste

        if triggered == "upload-file" and contents:
            try:
                content_type, content_string = contents.split(',')
                decoded = base64.b64decode(content_string)
                file_size = len(decoded) / (1024 * 1024)  # Taille en Mo

                # Écriture temporaire du fichier pour Spark
                temp_file = "temp_upload"
                with open(temp_file, 'wb') as f:
                    f.write(decoded)
                    
                # Traitement avec Pandas pour petits fichiers
                if file_size < 30:
                    # Traitement des fichiers JSON
                    if filename.lower().endswith('.json'):
                        print("Fichier JSON détecté (petit), traitement avec Pandas.")
                        try:
                            for encoding in ['utf-8', 'utf-16', 'latin-1']:
                                try:
                                    df_json_str = decoded.decode(encoding)
                                    break
                                except UnicodeDecodeError:
                                    continue
                            else:
                                raise ValueError("Aucun encodage valide trouvé pour le JSON.")
                            json_data = json.loads(df_json_str)
                            df = None
                            for orient in ["split", "records", "index", "columns", "values"]:
                                try:
                                    df = pd.read_json(io.StringIO(df_json_str), orient=orient)
                                    break
                                except (ValueError, Exception):
                                    continue
                            if df is None:
                                df = pd.DataFrame(json_data)
                            nested_cols = [col for col in df.columns if df[col].apply(lambda x: isinstance(x, (dict, list))).any()]
                            if nested_cols:
                                df = pd.json_normalize(json_data)
                            df_json_str = df.to_json(orient="split")
                        except (json.JSONDecodeError, ValueError, Exception) as e:
                            print(f"Erreur détectée dans le parsing JSON : {str(e)}")
                            error['chargement'] = str(e)
                            return [None, None, status, True, error, cache]
                    # Traitement des fichiers CSV
                    elif filename.lower().endswith('.csv'):
                        print("Fichier CSV détecté (petit), traitement avec Pandas.")
                        for encoding in ["utf-8", "utf-8-sig", "latin-1", "cp1252"]:
                            try:
                                sample = io.StringIO(decoded.decode(encoding))
                                first_five_lines = [next(sample) for _ in range(5)] if len(decoded) > 0 else [""]
                                sample.seek(0)
                                potential_separators = [',', ';', '\t']
                                line = first_five_lines[0].strip()
                                separator_counts = {sep: line.count(sep) for sep in potential_separators}
                                separator = max(separator_counts.items(), key=lambda x: x[1])[0] if max(separator_counts.values()) > 0 else ','
                                break
                            except (UnicodeDecodeError, StopIteration):
                                continue
                        else:
                            raise ValueError("Erreur d'encodage pour détecter le séparateur.")
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
                        raise ValueError("Type de fichier non pris en charge. Utilisez .csv ou .json.")
                # Traitement avec Spark pour gros fichiers
                else:
                    print("Fichier volumineux détecté, traitement avec Spark.")
                    spark = SparkSession.builder.appName("JunbiData").master("local[*]").getOrCreate()
                    try:
                        # Traitement des fichiers JSON avec Spark
                        if filename.lower().endswith('.json'):
                            print("Traitement d'un gros fichier JSON avec Spark.")
                            # Étape 1 : Tester le format JSON
                            for encoding in ['utf-8', 'utf-16', 'latin-1']:
                                try:
                                    with open(temp_file, 'r', encoding=encoding) as f:
                                        json_data = json.load(f)
                                    break
                                except (UnicodeDecodeError, json.JSONDecodeError):
                                    continue
                            else:
                                raise ValueError("Aucun encodage valide trouvé pour le JSON.")
                            # Étape 2 : Convertir si nécessaire
                            if isinstance(json_data, dict) and 'columns' in json_data and 'data' in json_data:
                                # Format 'split', convertir en liste de dictionnaires
                                json_data = [dict(zip(json_data['columns'], row)) for row in json_data['data']]
                                with open(temp_file, 'w', encoding='utf-8') as f:
                                    json.dump(json_data, f)
                            df = spark.read.option("multiline", "true").option("inferSchema", "true").json(temp_file)
                            # Aplanissement des colonnes imbriquées
                            for column in df.columns:
                                if isinstance(df.schema[column].dataType, StructType):
                                    for field_name, field_type in df.schema[column].dataType.fields:
                                        df = df.withColumn(f"{column}.{field_name}", df[column][field_name])
                                    df = df.drop(column)
                                elif isinstance(df.schema[column].dataType, ArrayType):
                                    df = df.withColumn(column, df[column].cast("string"))
                            pdf = df.limit(10).toPandas()
                            df_json_str = pdf.to_json(orient="split")
                            df.write.parquet("data/large_dataset.parquet", mode="overwrite")

                        # Traitement des fichiers CSV avec Spark
                        elif filename.lower().endswith('.csv'):
                            print("Traitement d'un gros fichier CSV avec Spark.")
                            for encoding in ["utf-8", "iso-8859-1", "us-ascii", "utf-16", "utf-16be", "utf-16le", "utf-32"]:
                                try:
                                    df = spark.read.option("encoding", encoding).option("delimiter", ",").option("header", "true").option("inferSchema", "true").csv(temp_file)
                                    print(f"Spark a lu le fichier avec l'encodage {encoding}")
                                    df.write.parquet("data/large_dataset.parquet", mode="overwrite")
                                    pdf = df.limit(10).toPandas()
                                    df_json_str = pdf.to_json(orient="split")
                                    break
                                except Exception as e:
                                    print(f"Erreur avec encodage {encoding}: {str(e)}")
                                    continue
                            else:
                                raise ValueError("Aucun encodage valide trouvé avec Spark.")
                        else:
                            raise ValueError("Type de fichier non pris en charge. Utilisez .csv ou .json.")
                    finally:
                        spark.stop()
                        if os.path.exists(temp_file):
                            os.remove(temp_file)

                status["chargement"] = True
                return [df_json_str, filename, status, False, None, cache]  # Retour explicite comme liste
            except Exception as e:
                if 'spark' in globals():
                    spark.stop()
                status["chargement"] = False
                print(f"Exception globale capturée : {str(e)}")
                error['chargement'] = str(e)
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                return [None, None, status, True, error, cache]

        raise dash.exceptions.PreventUpdate