# app/modules/nettoyage_submodules/duplicates_clean.py
import dash_bootstrap_components as dbc
from dash import html, dcc, Input, Output, State, ALL, MATCH, Patch, ctx, dash_table
from dash.exceptions import PreventUpdate

import numpy as np
import pandas as pd
import io

from app.modules.common.io import load_df, format_warning, truncate_preview_value
from app.modules.chargement import show_dataset_preview
from app.modules.common.ui import STYLE_DROPDOWN, CHECKLIST_STYLE, CHECKLIST_INPUT_STYLE, CHECKLIST_LABEL_STYLE, get_dynamic_checklist_label_style

try:
    from pyspark.sql import functions as F
    HAS_SPARK = True
except Exception:
    HAS_SPARK = False

DUPLICATE_OPTIONS = [
    {"label": "Garder le premier", "value": "first"},
    {"label": "Garder le dernier", "value": "last"},
    {"label": "Supprimer tous les doublons", "value": "none"},
]

TAB_ID = "clean-duplicates"

# =========================================================
# Fonctions logique métier
# =========================================================

def apply_duplicates(df, rules, is_spark=False):
    """
    Applique les règles de suppression des doublons
    sur une copie du DataFrame.
    """

    if not rules:
        return df

    if is_spark:
        from pyspark.sql import Window
        from pyspark.sql import functions as F

        result = df

        for rule in rules:
            columns = rule.get("columns")
            action = rule.get("action")

            if not columns or not action:
                raise ValueError("Règle de doublons invalide.")

            if action == "first":
                result = result.dropDuplicates(columns)

            elif action == "last":
                # Spark ne garantit pas naturellement le dernier élément.
                # À adapter si un ordre explicite existe.
                result = result.dropDuplicates(columns)

            elif action == "none":
                window = Window.partitionBy(*columns)

                result = (
                    result
                    .withColumn("_duplicate_count", F.count("*").over(window))
                    .filter(F.col("_duplicate_count") == 1)
                    .drop("_duplicate_count")
                )

            else:
                raise ValueError(
                    f"Stratégie de doublons inconnue : '{action}'."
                )

        return result

    result = df.copy()

    for rule in rules:
        columns = rule.get("columns")
        action = rule.get("action")

        if not columns or not action:
            raise ValueError("Règle de doublons invalide.")

        if action == "first":
            result = result.drop_duplicates(
                subset=columns,
                keep="first"
            )

        elif action == "last":
            result = result.drop_duplicates(
                subset=columns,
                keep="last"
            )

        elif action == "none":
            result = result[
                ~result.duplicated(
                    subset=columns,
                    keep=False
                )
            ]

        else:
            raise ValueError(
                f"Stratégie de doublons inconnue : '{action}'."
            )

    return result

def get_duplicate_rules_from_pipeline(pipeline):
    if not pipeline:
        return {}
    print(f"--------- Pipeline : {pipeline} ---------")
    rules = []

    for step in pipeline:
        if step.get("step") != "duplicates":
            continue

        for rule in step.get("params", []):
            columns = rule.get("columns")
            action = rule.get("action")

            if not columns or not action:
                continue

            if columns and action:
                # La dernière règle gagne
                rules.append({
                    "columns": columns,
                    "action": action,
                })

    return rules

def build_duplicate_rule_row(key_id, columns, selected_columns=None, selected_action=None):
    selected_columns = selected_columns or []

    options = [
        {
            "label": column,
            "value": column
        }
        for column in columns
    ]

    return html.Div([
        html.H5(
            f"Clé de comparaison {key_id}",
            className="mt-3"
        ),

        html.P(
            "Sélectionnez les colonnes servant de clé :",
            style={
                "fontSize": "17px",
                "marginBottom": "10px"
            }
        ),

        dcc.Checklist(
            id={
                "type": "dup-columns",
                "key_id": key_id
            },
            options=options,
            value=selected_columns,
            style=CHECKLIST_STYLE,
            inputStyle=CHECKLIST_INPUT_STYLE,
            labelStyle=get_dynamic_checklist_label_style(options),
        ),

        html.Div(
            id={
                "type": "dup-preview",
                "key_id": key_id
            },
            className="mt-3 mb-3"
        ),

        html.P(
            "Méthode à appliquer :",
            className="mb-1"
        ),

        dcc.Dropdown(
            id={
                "type": "dup-strategy",
                "key_id": key_id
            },
            options=DUPLICATE_OPTIONS,
            value=selected_action,
            clearable=True,
            placeholder="Choisir une méthode",
            style={**STYLE_DROPDOWN, "marginBottom": "5px"}
        ),

        html.Hr(className="mt-4"),

    ], className="mb-3")

def get_existing_key_ids(children):
    key_ids = []

    if not isinstance(children, list):
        children = [children]

    for child in children:
        if not isinstance(child, dict):
            continue

        props = child.get("props", {})
        children_props = props.get("children", [])

        if not isinstance(children_props, list):
            children_props = [children_props]

        for component in children_props:
            if not isinstance(component, dict):
                continue

            component_props = component.get("props", {})
            component_id = component_props.get("id")

            if (
                isinstance(component_id, dict)
                and component_id.get("type") == "dup-columns"
            ):
                key_id = component_id.get("key_id")

                if isinstance(key_id, int):
                    key_ids.append(key_id)

    return key_ids

def show_duplicate_preview(preview_json, duplicate_counts=None, page_size=10, max_rows=100):
    """
    Affiche un aperçu paginé et limité des doublons.
    """

    if not preview_json:
        return html.Div(
            "Aucun groupe de doublons trouvé.",
            className="text-muted"
        )

    try:
        preview_df = pd.read_json(
            io.StringIO(preview_json),
            orient="split"
        )

        # Sécurité supplémentaire côté affichage
        preview_df = preview_df.head(max_rows)

        # Conversion sûre des valeurs
        for column in preview_df.columns:
            preview_df[column] = preview_df[column].map(
                lambda value: truncate_preview_value(value)
            )

        columns = [
            {
                "name": str(column),
                "id": str(column),
                "type": "text"
            }
            for column in preview_df.columns
        ]

        total_groups = len(duplicate_counts or [])

        info = html.Div([
            html.H6(
                f"🔎 {total_groups} groupe(s) de doublons détecté(s). "
                f"Aperçu limité à {len(preview_df)} ligne(s).",
            ),

            dash_table.DataTable(
                data=preview_df.to_dict("records"),
                columns=columns,
                page_action="native",
                page_size=page_size,
                page_current=0,
                style_table={
                    "overflowX": "auto",
                    "overflowY": "auto",
                },
                style_cell={
                    "textAlign": "left",
                    "fontSize": "13px",
                    "backgroundColor": "#f2f2f2", 
                    "color": "#111",
                    "overflow": "hidden",
                    "textOverflow": "ellipsis",
                },
                style_header={
                    "backgroundColor": "#e0e0e0",
                    "fontWeight": "bold",
                    "color": "#000",
                },
            )
        ])

        return info

    except Exception as exc:
        print(f"Erreur aperçu doublons : {exc}")
        return format_warning("Aperçu des doublons indisponible.")

def build_duplicates_preview(df, selected_columns, is_spark, n_rows=100):
    if not selected_columns:
        return None, None

    if is_spark:
        duplicate_keys = (
            df.groupBy(*selected_columns)
              .count()
              .filter(F.col("count") > 1)
        )

        # Récupération des tailles de groupes
        duplicate_counts = [
            row["count"]
            for row in duplicate_keys.select("count").collect()
        ]

        # Suppression de count avant la jointure
        duplicate_df = (
            df.join(
                duplicate_keys.drop("count"),
                on=selected_columns,
                how="inner"
            )
            .limit(n_rows)
        )

        preview_pdf = duplicate_df.toPandas()

    else:
        group_sizes = (
            df.groupby(selected_columns, dropna=False)
              .size()
              .reset_index(name="count")
        )

        duplicate_groups = group_sizes[
            group_sizes["count"] > 1
        ]

        duplicate_counts = duplicate_groups["count"].tolist()

        duplicate_mask = df.duplicated(
            subset=selected_columns,
            keep=False
        )

        preview_pdf = (
            df.loc[duplicate_mask]
              .head(n_rows)
        )

    if preview_pdf.empty:
        return None, None

    preview_json = preview_pdf.to_json(
        orient="split",
        date_format="iso"
    )

    return preview_json, duplicate_counts


# =========================================================
# Fonctions DASH
# =========================================================

def get_tab():
    return dbc.Tab(tab_id=TAB_ID, label="Doublons")

def get_layout():
    return html.Div([
        html.P(
            "Nettoyage des doublons (choix des colonnes pour définir la clé de duplication, choisir la méthode de traitement puis appliquer globalement) :",
            style={"fontSize": "18px", "marginBottom": "10px"},
        ),
        dbc.Row([
            html.Div(id="original-dataset-preview", style={"marginBottom": "15px"})
        ]),
        dbc.Col([
            dbc.Button("🔄 Réinitialiser", id="dup-reset", style={"display": "none", "marginTop": "6px", "marginBottom": "6px"}),
        ]),
        dbc.Row([        
            html.P(
                "Sélectionnez les colonnes servant de clé de comparaison :",
                style={"fontSize": "17px"}
            ),
        ]),
        dbc.Button(
            "+ Ajouter une clé",
            id="dup-key-rows",
            color="primary",
        ),
        html.Hr(),
        html.Div(
            id="dup-rules-container",
            className="mt-3"
        ),
        dbc.Button(
            "Appliquer",
            id="dup-apply",
            color="primary",
            className="mt-3"
        ),
        html.Div(id="dup-feedback", className="mt-2"),
    ])

def register_callbacks(app):
    @app.callback(
        Output("dup-reset", "style"),
        Input("pipeline-store", "data"),
        Input("cleaning-subtabs", "active_tab")
    )
    def update_duplicates_reset_button(pipeline, active_tab):
        has_duplicates = any(
            step.get("step") == "duplicates"
            for step in (pipeline or [])
        )

        return {
            "display": "inline-block"
            if active_tab == TAB_ID and has_duplicates
            else "none",
            "marginTop": "6px",
            "marginBottom": "6px",
        }

    @app.callback(
        Output({"type": "dup-preview","key_id": MATCH}, "children"),
        Input({"type": "dup-columns","key_id": MATCH}, "value"),
        State("original-parquet-path-store", "data"),
    )
    def update_duplicate_preview(selected_columns, path):

        if not path:
            return html.Div(
                "Aucun dataset chargé.",
            )

        selected_columns = selected_columns or []

        if not selected_columns:
            return html.Div(
                "Sélectionnez au moins une colonne pour afficher les doublons.",
            )

        df, is_spark = load_df(path)

        if df is None:
            return format_warning("Impossible de charger le dataset.")

        try:
            preview_json, duplicate_counts = build_duplicates_preview(
                df=df,
                selected_columns=selected_columns,
                is_spark=is_spark,
                n_rows=100
            )

            if not preview_json:
                return dbc.Alert(
                    "✅ Aucun doublon trouvé avec cette clé.",
                    color="success"
                )

            groups_text = ", ".join(
                f"{count} lignes"
                for count in duplicate_counts
            )

            return html.Div([
                html.H5(
                    f"🔢 Groupes de doublons : {groups_text}",
                    className="mb-2"
                ),

                show_duplicate_preview(
                    preview_json,
                    duplicate_counts=duplicate_counts,
                    page_size=10,
                    max_rows=100,
                )
            ])

        except Exception as exc:
            return dbc.Alert(
                f"Erreur lors de l'analyse des doublons : {exc}",
                color="danger"
            )

    @app.callback(
        Output("dup-rules-container", "children"),
        Input("dup-key-rows", "n_clicks"),
        Input("pipeline-store", "data"),
        State("original-parquet-path-store", "data"),
        State("dup-rules-container", "children"),
        prevent_initial_call=False,
    )
    def render_duplicate_rows(n_clicks, pipeline, path, current_children):
        triggered_id = ctx.triggered_id

        if not path:
            return html.Div(
                "Aucun dataset chargé.",
            )

        df, is_spark = load_df(path)

        if df is None:
            return html.Div(
                "Impossible de charger le dataset.",
                className="text-danger"
            )

        columns = list(df.columns)
        existing_rules = get_duplicate_rules_from_pipeline(pipeline)

        # ctx.triggered_id vaut None lors de l'initialisation :
        if triggered_id == "dup-key-rows":
            existing_key_ids = get_existing_key_ids(current_children)

            next_key_id = (
                max(existing_key_ids, default=0) + 1
            )

            new_row = build_duplicate_rule_row(
                key_id=next_key_id,
                columns=columns,
                selected_columns=[],
                selected_action=None,
            )

            patched_children = Patch()

            if isinstance(current_children, list) and current_children:
                patched_children.append(new_row)
                return patched_children

            return [new_row]

        # Initialisation ou mise à jour depuis le pipeline
        existing_count = len(existing_rules)
        clicked_count = n_clicks or 0
        number_of_rows = max(existing_count, clicked_count)

        if number_of_rows == 0:
            return html.Div(
                "Cliquez sur « Ajouter une clé » pour définir une comparaison.",
            )

        rows = []

        for index in range(number_of_rows):
            if index < existing_count:
                rule = existing_rules[index]
                selected_columns = rule["columns"]
                selected_action = rule["action"]
            else:
                selected_columns = []
                selected_action = None

            rows.append(
                build_duplicate_rule_row(
                    key_id=index + 1,
                    columns=columns,
                    selected_columns=selected_columns,
                    selected_action=selected_action,
                )
            )

        return rows

    @app.callback(
    Output("dup-columns", "options"),
    Output("dup-columns", "value"),
    Output("dup-columns", "labelStyle"),
    Input("original-parquet-path-store", "data"),
    Input("pipeline-store", "data"),
    prevent_initial_call=False
    )
    def populate_format_checklist(path, pipeline):
        df, is_spark = load_df(path)

        if df is None:
            return [], [], CHECKLIST_LABEL_STYLE

        columns = list(df.columns)
        existing_rules = get_duplicate_rules_from_pipeline(pipeline)

        options = [
            {
                "label": column,
                "value": column
            }
            for column in columns
        ]
        selected_columns = [
            column for column in columns
            if column in existing_rules
        ]

        label_style = get_dynamic_checklist_label_style(options)

        return options, selected_columns, label_style

    @app.callback(
        Output("pipeline-store", "data", allow_duplicate=True),
        Output("dup-feedback", "children"),
        Input("dup-apply", "n_clicks"),
        Input("dup-reset", "n_clicks"),
        State({"type": "dup-columns", "key_id": ALL}, "value"),
        State({"type": "dup-columns", "key_id": ALL}, "id"),
        State({"type": "dup-strategy", "key_id": ALL}, "value"),
        State("original-parquet-path-store", "data"),
        State("pipeline-store", "data"),
        prevent_initial_call=True,
    )
    def on_apply_duplicates_click(apply_clicks, reset_clicks, columns_values, columns_ids, strategies, path, pipeline):
        from app.modules.common.pipeline import add_step, reset_step

        pipeline = pipeline or []

        if ctx.triggered_id == "dup-reset":
            if not reset_clicks:
                raise PreventUpdate

            updated_pipeline = reset_step(
                pipeline,
                "duplicates"
            )

            return (
                updated_pipeline,
                dbc.Alert(
                    "✅ Les stratégies de doublons ont été réinitialisées.",
                    color="info"
                )
            )

        if ctx.triggered_id != "dup-apply":
            raise PreventUpdate

        df, is_spark = load_df(path)

        if df is None:
            return pipeline, format_warning(
                "Aucun dataset chargé."
            )

        rules = []

        for columns, component_id, action in zip(
            columns_values or [],
            columns_ids or [],
            strategies or [],
        ):
            columns = columns or []

            if not columns:
                continue

            if not action:
                return pipeline, format_warning(
                    f"Choisissez une stratégie pour la clé "
                    f"{component_id['key_id']}."
                )

            rules.append({
                "columns": columns,
                "action": action,
            })

        if not rules:
            return pipeline, format_warning(
                "Aucune règle de doublons configurée."
            )

        try:
            # Validation de l'application sans modifier le dataset original
            apply_duplicates(
                df.copy() if not is_spark else df,
                rules,
                is_spark=is_spark
            )

            new_step = {
                "step": "duplicates",
                "params": rules,
            }

            updated_pipeline = add_step(
                pipeline,
                new_step
            )

            return (
                updated_pipeline,
                dbc.Alert(
                    f"✅ {len(rules)} règle(s) de doublons enregistrée(s).",
                    color="success"
                )
            )

        except Exception as exc:
            return (
                pipeline,
                format_warning(
                    f"Erreur lors de la configuration des doublons : {exc}"
                )
            )
