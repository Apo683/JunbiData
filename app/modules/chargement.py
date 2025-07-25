import dash
from dash import html, dcc, Input, Output, State, dash_table
import dash_bootstrap_components as dbc
import base64
import io
import os
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
            html.Div("🔍 Détection auto du séparateur", style={"fontSize": "14px", "color": "#ccc", "marginTop": "8px"})
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

        if triggered == "reset-upload" and reset_clicks and reset_clicks > 0:
            if 'spark' in globals():
                spark.stop()  # Ferme la session Spark si elle existe
            status["chargement"] = False
            cache.pop("chargement", None)
            return None, None, status, True, None, cache

        if triggered == "upload-csv" and contents:
            try:
                content_type, content_string = contents.split(',')
                decoded = base64.b64decode(content_string)
                print(f"----- Taille du fichier brut len(decoded) : {len(decoded)} octets -----")
                file_size = len(decoded) / (1024 * 1024)  # En Mo
                print(f"----- Taille du fichier file_size : {file_size:.2f} Mo -----")

                # Étape 1 : Charger les 5 premières lignes avec Pandas pour détecter le séparateur
                for encoding in ["utf-8", "utf-8-sig", "latin-1", "cp1252"]:
                    try:
                        sample = io.StringIO(decoded.decode(encoding))
                        first_five_lines = [next(sample) for _ in range(5)] if len(decoded) > 0 else [""]
                        sample.seek(0)
                        # Analyser manuellement le séparateur
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
                    # Étape 2a : Utiliser Pandas pour tout le fichier
                    for encoding in ["utf-8", "utf-8-sig", "latin-1"]:
                        try:
                            df = pd.read_csv(io.StringIO(decoded.decode(encoding)), sep=separator, engine='python')
                            break
                        except UnicodeDecodeError:
                            continue
                    else:
                        raise ValueError("Erreur d'encodage avec Pandas.")
                    df_json = df.to_json(date_format='iso', orient='split')
                else:
                    print("Fichier de grande taille, traitement avec Spark.")
                    # Étape 2b : Utiliser Spark avec le séparateur détecté
                    # Créer une session Spark à la volée
                    spark = SparkSession.builder.appName("JunbiData").master("local[*]").getOrCreate()
                    # Sauvegarder temporairement dans un fichier local
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
                                # df_sampled = df.limit(1000)  # Limiter à 1000 lignes pour éviter les problèmes de mémoire
                                # pdf = df_sampled.toPandas()
                                pdf = df.toPandas()
                                df_json = pdf.to_json(date_format='iso', orient='split')
                                print(f"----- Extrait JSON du DataFrame Spark : {df_json[:2]}... -----")
                                print(f"Echantillon généré : {len(df_json)} lignes")
                                break
                            except Exception as e:
                                print(f"Erreur avec encodage {encoding}: {str(e)}")
                                continue
                        else:
                            raise ValueError("Aucun encodage valide trouvé avec Spark.")
                    finally:
                        if os.path.exists(temp_file):
                            os.remove(temp_file)  # Nettoyage du fichier temporaire

                status["chargement"] = True
                return df_json, filename, status, False, None, cache
            except Exception as e:
                if 'spark' in globals():
                    spark.stop()
                status["chargement"] = False
                return None, None, status, True, html.Div(f"❌ Erreur : {str(e)}"), cache

        raise dash.exceptions.PreventUpdate