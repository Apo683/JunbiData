import dash
from dash import html, dcc, Input, Output, State, dash_table
import dash_bootstrap_components as dbc
import dash_uploader as du
import io
import os
import shutil
import json
import uuid
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, ArrayType

# Style du container upload
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

# Génération des résultats de l'upload
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
    print(f"----- Aperçu du dataset avec {df.shape[0]} lignes et {df.shape[1]} colonnes -----")
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

# 🎯 Rendu dynamique du module chargement selon l'état d'upload
def get_content(show_upload=True, df_json=None, filename=None, error=None):
    print(f"get_content appelé avec show_upload={show_upload}, df_json présent={df_json is not None}, filename={filename}")
    
    # Validation du paramètre error
    if error is not None and not isinstance(error, dict):
        error = {"chargement": str(error)} if error else {}
    
    # Composant upload (visible uniquement si pas de dataset chargé)
    upload_box = html.Div([
        du.Upload(
            id="upload-file",
            text="📁 Glissez-déposez un fichier CSV ou JSON ici ou cliquez",
            cancel_button=True,
            pause_button=True,
            max_file_size=500,
            filetypes=["csv", "json"],
            default_style={
                'width': '100%',
                'height': '90px',
                'border': '1px dashed',
                'borderRadius': '5px',
                'textAlign': 'center',
                'marginTop': '15px',
                'marginBottom': '10px',
                'backgroundColor': '#1a2a3a',
                'color': 'white',
                'display': 'block',
                'position': 'relative',
            }
        ),
        html.Div(
            "🔍 Formats acceptés : .csv, .json (Détection auto du séparateur pour .csv, .json au format (orient='split'))",
            style={
                "fontSize": "14px",
                "color": "#ccc",
                "position": "absolute",
                "top": "40%",
                "width": "100%",
                "textAlign": "center",
                "zIndex": 0
            }
        )
    ], style={"display": "block" if show_upload else "none", "position": "relative"})

    # Informations et aperçu du dataset (visible uniquement si dataset chargé)
    additional_info = html.Div([
        generate_upload_info(pd.read_json(io.StringIO(df_json), orient="split"), filename),
        show_dataset_preview(df_json)
    ], style={"marginTop": "10px"}) if df_json and filename else html.Div()

    # Affichage des erreurs
    error_display = dbc.Alert(
        error.get('chargement', '') if isinstance(error, dict) and error else '',
        color="danger",
        style={
            "marginTop": "10px", 
            "marginBottom": "10px", 
            "display": "block" if (isinstance(error, dict) and error and error.get('chargement')) else "none"
        }
    )

    # Bouton reset (visible uniquement si dataset chargé)
    reset_button_style = {"display": "inline-block" if df_json else "none", "marginBottom": "15px"}

    return html.Div([
        html.H5("📂 Chargement du jeu de données :", style={"marginBottom": "15px"}),
        dbc.Button("🔄 Réinitialiser", id="reset-upload", n_clicks=0, style=reset_button_style),
        error_display,
        upload_box,
        additional_info
    ])

# 📥 Callbacks séparés pour Upload et Reset
def register_callbacks_chargement(app):
    
    # 1️⃣ CALLBACK UPLOAD - Traite uniquement les uploads avec dash-uploader
    @du.callback(
        output=[Output("df-store", "data"),
                Output("filename-store", "data"),
                Output("module-status", "data"),
                Output("show-upload", "data"),
                Output("error-store", "data"),
                Output("module-cache", "data", allow_duplicate=True)],
        id="upload-file"  # Restaurer cet ID pour lier au du.Upload
    )
    def handle_upload(status: du.UploadStatus):
        print(f"=== CALLBACK UPLOAD - Status: {type(status)} ===")
        print(f"Status dict: {status.__dict__ if hasattr(status, '__dict__') else 'Pas de __dict__'}")
        print(f"Uploaded files: {status.uploaded_files if hasattr(status, 'uploaded_files') else 'Pas de uploaded_files'}")
        
        # Récupération sécurisée des states actuels
        try:
            current_status = dash.callback_context.states.get("module-status.data", {})
            if not isinstance(current_status, dict):
                current_status = {"chargement": False}
        except:
            current_status = {"chargement": False}
            
        try:
            current_error = dash.callback_context.states.get("error-store.data", {})
            if not isinstance(current_error, dict):
                current_error = {}
        except:
            current_error = {}
            
        try:
            cache = dash.callback_context.states.get("module-cache.data", {}) or {}
            if not isinstance(cache, dict):
                cache = {}
        except:
            cache = {}

        status_dict = current_status.copy()
        error_dict = current_error.copy()

        # Vérification que status est bien un objet UploadStatus
        if not hasattr(status, 'is_completed'):
            print(f"ERREUR: status n'est pas un objet UploadStatus: {type(status)}")
            raise dash.exceptions.PreventUpdate

        if status.is_completed:
            # Utilisation du chemin de sauvegarde réel
            upload_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tmp")
            filename = status.uploaded_files[0] if status.uploaded_files else None
            print(f"Fichier via uploaded_files: {filename}")

            if filename:
                # Extraire le nom de fichier brut et construire le chemin correct
                base_filename = os.path.basename(filename)
                corrected_filename = os.path.join(upload_folder, base_filename)
                print(f"Chemin corrigé: {corrected_filename}")

                if not os.path.exists(corrected_filename):
                    error_dict['chargement'] = f"Impossible de localiser le fichier uploadé: {corrected_filename}"
                    return [None, None, status_dict, True, error_dict, cache]
                filename = corrected_filename

            # Le reste du code de traitement avec impressions détaillées
            try:
                file_size = os.path.getsize(filename) / (1024 * 1024)
                print(f"Taille du fichier: {file_size:.2f} Mo")

                if file_size < 30:
                    # Traitement des petits fichiers avec Pandas
                    if filename.lower().endswith('.json'):
                        print("Fichier JSON détecté (petit), traitement avec Pandas.")
                        try:
                            with open(filename, 'r') as f:
                                df_json_str = f.read()
                            for encoding in ['utf-8', 'utf-16', 'latin-1']:
                                try:
                                    df_json_str = df_json_str.encode().decode(encoding)
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
                                    print(f"JSON lu avec orient={orient}")
                                    break
                                except (ValueError, Exception) as e:
                                    print(f"Erreur avec orient={orient}: {str(e)}")
                                    continue
                            if df is None:
                                df = pd.DataFrame(json_data)
                                print("Conversion en DataFrame réussie")
                            nested_cols = [col for col in df.columns if df[col].apply(lambda x: isinstance(x, (dict, list))).any()]
                            if nested_cols:
                                df = pd.json_normalize(json_data)
                                print("Normalisation JSON appliquée")
                            df_json_str = df.to_json(orient="split")
                            print(f"df_json_str généré: {df_json_str[:3]}...")  # Affiche les 3 premiers caractères
                        except (json.JSONDecodeError, ValueError, Exception) as e:
                            error_dict['chargement'] = f"Erreur JSON: {str(e)}"
                            return [None, None, status_dict, True, error_dict, cache]
                            
                    elif filename.lower().endswith('.csv'):
                        print("Fichier CSV détecté (petit), traitement avec Pandas.")
                        for encoding in ["utf-8", "utf-8-sig", "latin-1", "cp1252"]:
                            try:
                                with open(filename, 'r', encoding=encoding) as f:
                                    sample = io.StringIO(f.read())
                                first_five_lines = [next(sample) for _ in range(5)] if os.path.getsize(filename) > 0 else [""]
                                sample.seek(0)
                                potential_separators = [',', ';', '\t']
                                line = first_five_lines[0].strip()
                                separator_counts = {sep: line.count(sep) for sep in potential_separators}
                                separator = max(separator_counts.items(), key=lambda x: x[1])[0] if max(separator_counts.values()) > 0 else ','
                                print(f"Séparateur détecté: {separator}")
                                break
                            except (UnicodeDecodeError, StopIteration):
                                continue
                        else:
                            raise ValueError("Erreur d'encodage pour détecter le séparateur.")
                        
                        for encoding in ["utf-8", "utf-8-sig", "latin-1"]:
                            try:
                                with open(filename, 'r', encoding=encoding) as f:
                                    df = pd.read_csv(io.StringIO(f.read()), sep=separator, engine='python')
                                print(f"CSV lu avec encodage={encoding}")
                                break
                            except UnicodeDecodeError:
                                continue
                        else:
                            raise ValueError("Erreur d'encodage avec Pandas.")
                        df_json_str = df.to_json(orient="split")
                        print(f"df_json_str généré: {df_json_str[:3]}...")
                    else:
                        raise ValueError("Type de fichier non pris en charge. Utilisez .csv ou .json.")
                else:
                    # Traitement des gros fichiers avec Spark
                    print("Fichier volumineux détecté, traitement avec Spark.")
                    spark = SparkSession.builder.appName("JunbiData").master("local[*]").getOrCreate()
                    try:
                        if filename.lower().endswith('.json'):
                            print("Traitement d'un gros fichier JSON avec Spark.")
                            for encoding in ['utf-8', 'utf-16', 'latin-1']:
                                try:
                                    with open(filename, 'r', encoding=encoding) as f:
                                        json_data = json.load(f)
                                    print(f"JSON lu avec encodage={encoding}")
                                    break
                                except (UnicodeDecodeError, json.JSONDecodeError):
                                    continue
                            else:
                                raise ValueError("Aucun encodage valide trouvé pour le JSON.")
                            
                            if isinstance(json_data, dict) and 'columns' in json_data and 'data' in json_data:
                                json_data = [dict(zip(json_data['columns'], row)) for row in json_data['data']]
                                with open(filename, 'w', encoding='utf-8') as f:
                                    json.dump(json_data, f)
                            
                            df = spark.read.option("multiline", "true").option("inferSchema", "true").json(filename)
                            for column in df.columns:
                                if isinstance(df.schema[column].dataType, StructType):
                                    for field in df.schema[column].dataType.fields:
                                        df = df.withColumn(f"{column}.{field.name}", df[column][field.name])
                                    df = df.drop(column)
                                elif isinstance(df.schema[column].dataType, ArrayType):
                                    df = df.withColumn(column, df[column].cast("string"))
                            pdf = df.limit(10).toPandas()
                            df_json_str = pdf.to_json(orient="split")
                            df.write.parquet("data/large_dataset.parquet", mode="overwrite")
                            print(f"df_json_str généré: {df_json_str[:3]}...")
                            
                        elif filename.lower().endswith('.csv'):
                            print("Traitement d'un gros fichier CSV avec Spark.")
                            for encoding in ["utf-8", "iso-8859-1", "us-ascii", "utf-16", "utf-16be", "utf-16le", "utf-32"]:
                                try:
                                    df = spark.read.option("encoding", encoding).option("delimiter", ",").option("header", "true").option("inferSchema", "true").csv(filename)
                                    print(f"Spark a lu le fichier avec l'encodage {encoding}")
                                    df.write.parquet("data/large_dataset.parquet", mode="overwrite")
                                    pdf = df.limit(10).toPandas()
                                    df_json_str = pdf.to_json(orient="split")
                                    print(f"df_json_str généré: {df_json_str[:3]}...")
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
                        if os.path.exists(filename):
                            os.remove(filename)

                # Succès : mise à jour des stores
                error_dict.pop('chargement', None)  # Supprime les erreurs précédentes
                status_dict["chargement"] = True
                print("Upload réussi - Dataset chargé")
                return [df_json_str, os.path.basename(filename), status_dict, False, error_dict, cache]
                
            except Exception as e:
                print(f"Erreur lors du traitement : {str(e)}")
                status_dict["chargement"] = False
                error_dict['chargement'] = str(e)
                if os.path.exists(filename):
                    os.remove(filename)
                return [None, None, status_dict, True, error_dict, cache]

        raise dash.exceptions.PreventUpdate

    # 2️⃣ CALLBACK RESET - Traite uniquement les resets
    @app.callback(
        [Output("df-store", "data", allow_duplicate=True),
         Output("filename-store", "data", allow_duplicate=True),
         Output("module-status", "data", allow_duplicate=True),
         Output("show-upload", "data", allow_duplicate=True),
         Output("error-store", "data", allow_duplicate=True),
         Output("module-cache", "data", allow_duplicate=True)],
        Input("reset-upload", "n_clicks"),
        [State("module-status", "data"),
         State("error-store", "data"),
         State("module-cache", "data")],
        prevent_initial_call=True
    )
    def handle_reset(reset_clicks, current_status, current_error, cache):
        print(f"=== CALLBACK RESET - Clicks: {reset_clicks} ===")
        
        if reset_clicks and reset_clicks > 0:
            # Nettoyage des fichiers
            if os.path.exists("data/large_dataset.parquet"):
                shutil.rmtree("data/large_dataset.parquet")
                print("Fichier Parquet supprimé")
            
            upload_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tmp")
            if os.path.exists(upload_folder):
                shutil.rmtree(upload_folder)
                os.makedirs(upload_folder, exist_ok=True)
                print("Dossier upload nettoyé")
            
            # Reset des données
            status = {"chargement": False}
            error = {}
            cache_reset = {}
            
            print("Reset effectué - Retour à l'état initial")
            return [None, None, status, True, error, cache_reset]
        
        raise dash.exceptions.PreventUpdate