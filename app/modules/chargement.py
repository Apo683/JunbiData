# app/modules/chargement.py
import dash
from dash import html, dcc, Input, Output, State, dash_table
import dash_bootstrap_components as dbc
import dash_uploader as du
import io
import os
import shutil
import json
import uuid
import numpy as np
import pandas as pd
import traceback

from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, ArrayType
from app.modules import spark_utils
from datetime import date, datetime

from app.modules.common.io import prepare_preview_dataframe, prepare_adaptive_preview, truncate_preview_value

# Normalisation des marqueurs manquants
### Pandas
DEFAULT_MISSING_MARKERS = {
    "",
    " ",
    "na",
    "n/a",
    "nan",
    "null",
    "none",
}

def normalize_missing_markers_pandas(df: pd.DataFrame, markers=None, strip_spaces=True):
    """
    Convertit uniquement certains marqueurs textuels en valeurs manquantes.

    Retourne :
        df_normalized, normalization_info
    """

    if markers is None:
        markers = DEFAULT_MISSING_MARKERS

    markers = {
        str(marker).strip().lower()
        for marker in markers
    }

    result = df.copy()
    info = {
        "operation": "normalize_missing_markers",
        "columns": {},
        "total_replaced": 0,
    }

    for col in result.columns:
        if not (
            pd.api.types.is_object_dtype(result[col])
            or pd.api.types.is_string_dtype(result[col])
        ):
            continue
        values = result[col].astype("string")

        if strip_spaces:
            values = values.str.strip()

        normalized = values.str.lower()
        mask = normalized.isin(markers)
        replaced_count = int(mask.sum())

        if replaced_count > 0:
            result.loc[mask, col] = pd.NA
            info["columns"][col] = {
                "replaced": replaced_count
            }

        info["total_replaced"] += replaced_count

    return result, info

### Spark
from pyspark.sql import functions as F
from pyspark.sql.types import StringType


def normalize_missing_markers_spark(df, markers=None, strip_spaces=True):
    if markers is None:
        markers = DEFAULT_MISSING_MARKERS

    markers = {
        str(marker).strip().lower()
        for marker in markers
    }

    result = df
    info = {
        "operation": "normalize_missing_markers",
        "columns": {},
        "total_replaced": 0,
    }

    for field in df.schema.fields:
        if not isinstance(field.dataType, StringType):
            continue

        col_name = field.name
        value = F.col(col_name)

        normalized = F.lower(F.trim(value)) if strip_spaces else F.lower(value)
        condition = normalized.isin(list(markers))

        replaced_count = (
            df.select(
                F.sum(F.when(condition, 1).otherwise(0))
                .alias("count")
            )
            .collect()[0]["count"]
        )

        replaced_count = int(replaced_count or 0)
        if replaced_count > 0:
            info["columns"][col_name] = {
                "replaced": replaced_count
            }
        info["total_replaced"] += replaced_count

        result = result.withColumn(
            col_name,
            F.when(condition, F.lit(None).cast(StringType()))
             .otherwise(F.col(col_name))
        )

    return result, info

# Génération des résultats de l'upload
def generate_upload_info(parquet_path, filename, metadata=None):
    print(f"generate_upload_info: Vérification de {parquet_path}")
    metadata = metadata or {}

    row_count = metadata.get("row_count", 0)
    column_count = metadata.get("column_count", 0)

    return html.Div([
        html.P(f"✅ Fichier '{filename}' chargé avec succès !"),
        html.P(f"🔢 Ce jeu de données contient {row_count} lignes et {column_count} colonnes")
    ])

def show_dataset_preview(df_json, n_rows=10):
    if not df_json:
        return html.Div("⚠️ Aucun aperçu disponible.")

    try:
        df = pd.read_json(
            io.StringIO(df_json),
            orient="split"
        )

        display_df = df.head(n_rows)
        records = display_df.to_dict("records")

        columns = [
            {
                "name": [str(dtype), str(column)],
                "id": str(column),
                "type": "text"
            }
            for column, dtype in display_df.dtypes.items()
        ]
        
        print("Nombre de lignes JSON :", len(df))
        print("Nombre de lignes records :", len(records))
        print("Nombre de lignes affichées :", len(display_df))
        return html.Div([
            html.H6("🔎 Aperçu du dataset :"),
            dash_table.DataTable(
                data=records,
                columns=columns,
                merge_duplicate_headers=False,
                style_cell={"textAlign": "left", "fontSize": "14px", "backgroundColor": "#f2f2f2", "color": "#111"},
                style_header={"backgroundColor": "#e0e0e0", "fontWeight": "bold", "color": "#000"},
                page_action="none",
                page_size=len(records) or 1,
                style_table={"overflowX": "auto", "overflowY": "visible"},
                virtualization=False
            )
        ])

    except Exception as exc:
        print(f"Erreur aperçu : {exc}")
        return dbc.Alert(
            "Aperçu indisponible.",
            color="warning"
        )

# 🎯 Rendu dynamique du module chargement selon l'état d'upload
def get_content(show_upload=True, parquet_path=None, filename=None, df_json=None, error=None, cache=None):
    print(f"get_content appelé avec show_upload={show_upload}, parquet_path={parquet_path}, filename={filename}, df_json présent={df_json is not None}")
    
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
            max_file_size=1000,
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
            "🔍 Formats acceptés : .csv, .json (Détection auto du séparateur pour .csv)",
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
    if not isinstance(cache, dict):
        cache = {}
    metadata = cache.get("chargement", {})
    print("CACHE REÇU PAR get_content :", cache)
    print("METADATA :", metadata)
    metadata = cache.get("chargement", {}) if cache else {}
    additional_info = html.Div([
        generate_upload_info(parquet_path, filename, metadata),
        show_dataset_preview(df_json) if df_json else html.Div("⚠️ Aucun aperçu disponible.")
    ], style={"marginTop": "10px", "display": "block" if parquet_path else "none"})

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
    reset_button_style = {"display": "inline-block" if parquet_path else "none", "marginBottom": "5px"}

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
        output=[
            Output("original-parquet-path-store", "data"),
            Output("filename-store", "data"),
            Output("module-status", "data"),
            Output("show-upload", "data"),
            Output("error-store", "data"),
            Output("df-json-store", "data", allow_duplicate=True),
            Output("module-cache", "data", allow_duplicate=True)
        ],
        id="upload-file"
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

            if not filename:
                error_dict['chargement'] = "Aucun fichier uploadé détecté."
                return [None, None, None, status_dict, True, error_dict, None, cache]

            # Extraire le nom de fichier brut et construire le chemin correct
            base_filename = os.path.basename(filename)
            corrected_filename = os.path.join(upload_folder, base_filename)
            print(f"Chemin corrigé: {corrected_filename}")

            if not os.path.exists(corrected_filename):
                error_dict['chargement'] = f"Impossible de localiser le fichier uploadé: {corrected_filename}"
                return [None, None, None, status_dict, True, error_dict, None, cache]

            # Définir le chemin Parquet selon la taille
            file_size = os.path.getsize(corrected_filename) / (1024 * 1024)
            is_large_dataset = file_size >= 30
            parquet_path = os.path.join("data", "large_dataset_parquet" if is_large_dataset else "small_dataset_parquet")
            os.makedirs(os.path.dirname(parquet_path), exist_ok=True)

            # Le reste du code de traitement avec impressions détaillées
            try:
                print(f"----- Taille du fichier: {file_size:.2f} Mo -----")
                ##### Traitement des petits fichiers avec Pandas #####
                if not is_large_dataset:
                    # Si un petit dataset est chargé après un gros, arrêter Spark
                    if not is_large_dataset and spark_utils.is_spark_active():
                        spark_utils.stop_spark_session()
                    ##### Traitement des fichiers JSON avec Pandas #####
                    if corrected_filename.lower().endswith('.json'):
                        print("----- Fichier JSON détecté (petit), traitement avec Pandas. -----")
                        df = None
                        try:
                            with open(corrected_filename, 'r') as f:
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
                            for orient in ["split", "records", "index", "columns", "values"]:
                                try:
                                    df = pd.read_json(io.StringIO(df_json_str), orient=orient)
                                    print(f"----- JSON lu avec orient={orient} -----")
                                    break
                                except (ValueError, Exception) as e:
                                    print(f"----- Erreur avec orient={orient}: {str(e)} -----")
                                    continue
                            if df is None:
                                df = pd.DataFrame(json_data)
                                print("----- Conversion en DataFrame réussie -----")
                            nested_cols = [col for col in df.columns if df[col].apply(lambda x: isinstance(x, (dict, list))).any()]
                            if nested_cols:
                                df = pd.json_normalize(json_data)
                                print("----- Normalisation JSON appliquée -----")
                            ### Normalisation des marqueurs manquants
                            try:
                                df, normalization_info = normalize_missing_markers_pandas(df)
                            except Exception as e:
                                print(f"----- Erreur normalisation marqueurs manquants : {e} -----")
                                traceback.print_exc()
                                error_dict["chargement"] = (
                                    f"Erreur lors de la normalisation automatique : {str(e)}"
                                )
                                raise                          
                            ### Sauvegarde en Parquet
                            df.to_parquet(parquet_path)
                            ### Génération de l'aperçu JSON
                            df_json_str = prepare_adaptive_preview(df)
                            print(f"df_json_str (extrait) généré: {df_json_str[:3]}...")
                        except (json.JSONDecodeError, ValueError, Exception) as e:
                            error_dict['chargement'] = f"Erreur JSON: {str(e)}"
                            raise
                    ##### Traitement des fichiers CSV avec Pandas #####
                    elif corrected_filename.lower().endswith('.csv'):
                        print("----- Fichier CSV détecté (petit), traitement avec Pandas. -----")
                        try:
                            for encoding in ["utf-8", "utf-8-sig", "latin-1", "cp1252"]:
                                with open(corrected_filename, 'r', encoding=encoding) as f:
                                    sample = io.StringIO(f.read())
                                first_five_lines = [next(sample) for _ in range(5)] if os.path.getsize(corrected_filename) > 0 else [""]
                                sample.seek(0)
                                potential_separators = [',', ';', '\t']
                                line = first_five_lines[0].strip()
                                separator_counts = {sep: line.count(sep) for sep in potential_separators}
                                separator = max(separator_counts.items(), key=lambda x: x[1])[0] if max(separator_counts.values()) > 0 else ','
                                print(f"----- Séparateur détecté: {separator} -----")
                                break
                            else:
                                raise ValueError("Erreur d'encodage pour détecter le séparateur.")

                            for encoding in ["utf-8", "utf-8-sig", "latin-1"]:
                                try:
                                    with open(corrected_filename, 'r', encoding=encoding) as f:
                                        df = pd.read_csv(
                                            corrected_filename,
                                            sep=separator,
                                            engine="python",
                                            quotechar='"',
                                            doublequote=True,
                                            encoding=encoding,
                                            on_bad_lines="error"
                                        )
                                    print(f"----- CSV lu avec encodage={encoding} -----")
                                    break
                                except UnicodeDecodeError:
                                    continue
                            else:
                                raise ValueError("Erreur d'encodage avec Pandas.")
                            ### Normalisation des marqueurs manquants
                            try:
                                df, normalization_info = normalize_missing_markers_pandas(df)
                            except Exception as e:
                                print(f"----- Erreur normalisation marqueurs manquants : {e} -----")
                                traceback.print_exc()
                                error_dict["chargement"] = (
                                    f"Erreur lors de la normalisation automatique : {str(e)}"
                                )
                                raise                             
                            ### Sauvegarde en Parquet
                            df.to_parquet(parquet_path)
                            ### Génération de l'aperçu JSON
                            df_json_str = prepare_adaptive_preview(df)
                            print(f"----- df_json_str (extrait) généré: {df_json_str[:3]}... -----")
                        except (ValueError, Exception) as e:
                            error_dict['chargement'] = f"Erreur CSV: {str(e)}"
                            raise
                    else:
                        raise ValueError("Type de fichier non pris en charge. Utilisez .csv ou .json.")
                else:
                    ##### Traitement des gros fichiers avec Spark #####
                    print("----- Fichier volumineux détecté, traitement avec Spark. -----")
                    spark = spark_utils.get_spark_session()  # Initialiser ou récupérer la SparkSession globale
                    try:
                        ##### Traitement des fichiers JSON avec Spark #####
                        if corrected_filename.lower().endswith(".json"):
                            print("----- Traitement d'un gros fichier JSON avec Spark. -----")
                            json_data = None
                            last_error = None
                            for encoding in ["utf-8", "utf-16", "latin-1", "cp1252"]:
                                try:
                                    with open(corrected_filename, "r", encoding=encoding) as f:
                                        json_data = json.load(f)
                                    print(f"----- JSON correctement décodé avec {encoding} -----")
                                    break
                                except UnicodeDecodeError as e:
                                    last_error = e
                                    print(f"----- Encodage incompatible : {encoding} -----")
                                except json.JSONDecodeError as e:
                                    raise ValueError(
                                        f"Le fichier est décodable avec l'encodage {encoding}, "
                                        f"mais son contenu JSON est invalide : {e}"
                                    )
                            if json_data is None:
                                raise ValueError(
                                    f"Impossible de décoder le fichier JSON."
                                    f"Dernière erreur : {last_error}"
                                )
                            ### Lecture du JSON avec Spark
                            df = (
                                spark.read
                                .option("multiline", "true")
                                .option("inferSchema", "true")
                                .json(corrected_filename)
                            )
                            ### Normalisation des marqueurs manquants
                            try:
                                df, normalization_info = normalize_missing_markers_spark(df)

                            except Exception as e:
                                print(f"----- Erreur normalisation Spark : {e} -----")
                                traceback.print_exc()

                                error_dict["chargement"] = (
                                    f"Erreur lors de la normalisation automatique : {str(e)}"
                                )
                                raise
                            ### Sauvegarde en Parquet
                            df.write.mode("overwrite").parquet(parquet_path)
                            ### Génération de l'aperçu JSON
                            pdf = df.limit(10).toPandas()
                            df_json_str = prepare_adaptive_preview(pdf)
                            print(f"----- Taille module-cache : {len(json.dumps(cache, default=str))/ 1024 / 1024:.2f} Mo -----")
                            print(f"----- Taille aperçu : {len(df_json_str or '')/ 1024 / 1024:.2f} Mo -----")
                            print(
                                f"----- Taille aperçu JSON : "
                                f"{len(df_json_str) / 1024 / 1024:.2f} Mo -----"
                            )

                        ##### Traitement des fichiers CSV avec Spark #####
                        elif corrected_filename.lower().endswith(".csv"):
                            print("----- Traitement d'un gros fichier CSV avec Spark. -----")
                            encodings = [
                                "UTF-8",
                                "UTF-8-SIG",
                                "ISO-8859-1",
                                "Windows-1252",
                                "UTF-16",
                                "UTF-16LE",
                                "UTF-16BE",
                            ]
                            df = None
                            last_error = None
                            ### Tentative de lecture du CSV avec différents encodages
                            for encoding in encodings:
                                try:
                                    print(f"Test de l'encodage {encoding}")
                                    candidate_df = (
                                        spark.read
                                        .format("csv")
                                        .option("header", "true")
                                        .option("sep", ",")
                                        .option("encoding", encoding)
                                        .option("quote", '"')
                                        .option("escape", '"')
                                        .option("multiLine", "true")
                                        .option("mode", "PERMISSIVE")
                                        .option("inferSchema", "true")
                                        .option("columnNameOfCorruptRecord", "_corrupt_record")
                                        .csv(corrected_filename)
                                    )
                                    if "_corrupt_record" in candidate_df.columns:
                                        corrupt_count = candidate_df.filter(
                                            F.col("_corrupt_record").isNotNull()
                                        ).count()

                                        if corrupt_count > 0:
                                            raise ValueError(
                                                f"{corrupt_count} ligne(s) CSV sont mal formées."
                                            )

                                        candidate_df = candidate_df.drop("_corrupt_record")
                                    # Force une première lecture réelle
                                    candidate_df.limit(1).collect()
                                    df = candidate_df
                                    print(f"CSV correctement lu avec l'encodage {encoding}")
                                    break
                                except Exception as e:
                                    last_error = e
                                    print(f"Échec avec {encoding}: {type(e).__name__}: {e}")
                            if df is None:
                                raise ValueError(
                                    f"Impossible de lire le CSV avec les encodages testés. "
                                    f"Dernière erreur : {last_error}"
                                )
                            ### Normalisation des marqueurs manquants
                            try:
                                df, normalization_info = normalize_missing_markers_spark(df)

                            except Exception as e:
                                print(f"----- Erreur normalisation Spark : {e} -----")
                                traceback.print_exc()

                                error_dict["chargement"] = (
                                    f"Erreur lors de la normalisation automatique : {str(e)}"
                                )
                                raise
                            ### Sauvegarde en Parquet
                            df.write.mode("overwrite").parquet(parquet_path)
                            ### Génération de l'aperçu JSON
                            pdf = df.limit(10).toPandas()
                            df_json_str = prepare_adaptive_preview(pdf)
                            df_json_str = pdf.to_json(orient="split")
                            print("----- Taille module-cache :", len(json.dumps(cache, default=str)), "-----")
                            print("----- Taille aperçu :", len(df_json_str or ""), "-----")
                            print(f"----- df_json_str (extrait) généré : {df_json_str[:100]} -----")
                    except Exception as e:
                        traceback.print_exc()
                        error_dict["chargement"] = f"Erreur Spark: {type(e).__name__}: {e}"
                        raise

                # Succès : mise à jour des stores
                error_dict.pop('chargement', None)
                status_dict["chargement"] = True
                print("----- Upload réussi - Dataset chargé -----")
                original_parquet_path = parquet_path  # Stocker le chemin original

                # Mise à jour du cache du module
                if is_large_dataset:
                    backend = "spark"
                    row_count = df.count()
                    column_count = len(df.columns)
                else:
                    backend = "pandas"
                    row_count = len(df)
                    column_count = len(df.columns)

                cache.setdefault("chargement", {}).update({
                    "filename": os.path.basename(corrected_filename),
                    "parquet_path": parquet_path,
                    "backend": backend,
                    "row_count": row_count,
                    "column_count": column_count,
                    "file_size_mb": round(file_size, 2),
                    "processed_at": datetime.now().isoformat(timespec="seconds"),
                    "normalization_info": normalization_info,
                })

                return [original_parquet_path, os.path.basename(corrected_filename), status_dict, False, error_dict, df_json_str, cache]
                
            except Exception as e:
                print(f"----- Erreur lors du traitement : {str(e)} -----")
                status_dict["chargement"] = False
                error_dict['chargement'] = str(e)
                if os.path.exists(corrected_filename):
                    os.remove(corrected_filename)
                    print(f"----- Fichier temporaire {corrected_filename} supprimé en cas d'erreur -----")

                return [None, None, status_dict, True, error_dict, None, cache]

            finally:
                if os.path.exists(corrected_filename):
                    os.remove(corrected_filename)
                    print(f"----- Fichier temporaire {corrected_filename} supprimé (finally) -----")

        raise dash.exceptions.PreventUpdate

    # 2️⃣ CALLBACK RESET - Traite uniquement les resets
    @app.callback(
        [Output("original-parquet-path-store", "data", allow_duplicate=True),
         Output("filename-store", "data", allow_duplicate=True),
         Output("pipeline-store", "data", allow_duplicate=True),
         Output("module-status", "data", allow_duplicate=True),
         Output("show-upload", "data", allow_duplicate=True),
         Output("error-store", "data", allow_duplicate=True),
         Output("df-json-store", "data", allow_duplicate=True),
         Output("module-cache", "data", allow_duplicate=True)],
        Input("reset-upload", "n_clicks"),
        [State("module-status", "data"),
         State("error-store", "data"),
         State("module-cache", "data"),
         State("original-parquet-path-store", "data")],
        prevent_initial_call=True
    )
    def handle_reset(reset_clicks, current_status, current_error, cache, parquet_path):
        print(f"=== CALLBACK RESET - Clicks: {reset_clicks} ===")
        
        if reset_clicks and reset_clicks > 0:
            # Arrêter la SparkSession si active
            # spark_utils.stop_spark_session()
            
            # Nettoyage des fichiers Parquet
            for file in ["large_dataset_parquet", "small_dataset_parquet"]:
                full_path = os.path.join("data", file)
                if os.path.exists(full_path):
                    if os.path.isdir(full_path):
                        shutil.rmtree(full_path)
                        print(f"----- Dossier Parquet {full_path} supprimé -----")
                    else:
                        os.remove(full_path)
                        print(f"----- Fichier Parquet {full_path} supprimé -----")
            
            upload_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tmp")
            if os.path.exists(upload_folder):
                shutil.rmtree(upload_folder)
                os.makedirs(upload_folder, exist_ok=True)
                print(f"----- Dossier upload nettoyé (sécurité, mais vide ici) -----")
            
            # Reset des données
            status = {"chargement": False}
            error = {}
            cache_reset = {}
            
            print("----- Reset effectué - Retour à l'état initial -----")
            return [None, None, None, status, True, error, None, cache_reset]
        
        raise dash.exceptions.PreventUpdate