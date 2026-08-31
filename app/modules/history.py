# app/modules/history.py

from dash import html, Input, Output, callback
import dash_bootstrap_components as dbc


STEP_LABELS = {
    "formats": "Conversion des colonnes",
    "missing": "Gestion des valeurs manquantes",
    "duplicates": "Suppression des doublons",
    "outliers": "Gestion des valeurs extrêmes",
}

ACTION_LABELS = {
    # Formats
    "to_numeric": "Convertir en nombre",
    "to_string": "Convertir en texte",
    "to_datetime": "Convertir en date",
    "lowercase": "Convertir en minuscules",
    "uppercase": "Convertir en majuscules",
    "strip": "Supprimer les espaces inutiles",

    # Valeurs manquantes
    "drop_rows": "Supprimer les lignes",
    "mean": "Remplacer par la moyenne",
    "median": "Remplacer par la médiane",
    "mode": "Remplacer par le mode",
    "constant": "Remplacer par une constante",
    "ffill": "Propagation vers l'avant",
    "bfill": "Propagation vers l'arrière",

    # Doublons
    "first": "Garder la première occurrence",
    "last": "Garder la dernière occurrence",
    "none": "Supprimer tous les doublons",
}


def format_value(value):
    """Transforme une valeur technique en texte lisible."""
    if value is None:
        return "Non renseigné"

    if isinstance(value, bool):
        return "Oui" if value else "Non"

    return str(value)

def build_parameter_lines(step_name, params):
    parameter_lines = []

    if not isinstance(params, list):
        return [html.Li(format_value(params))]

    for param in params:
        action = param.get("action")

        action_label = ACTION_LABELS.get(
            action,
            f"Action « {format_value(action)} »"
        )

        # Cas duplicates : plusieurs colonnes dans une clé
        if step_name == "duplicates":
            columns = param.get("columns") or []

            if not isinstance(columns, list):
                columns = [columns]

            columns_label = " + ".join(
                format_value(column)
                for column in columns
            )

            parameter_lines.append(
                html.Li([
                    html.Strong("Clé de comparaison : "),
                    format_value(columns_label),
                    html.Br(),

                    html.Strong("Stratégie : "),
                    action_label,
                ])
            )

        # Cas formats et missing_values : une seule colonne
        else:
            column = param.get("column")

            parameter_lines.append(
                html.Li([
                    html.Strong("Colonne : "),
                    format_value(column),
                    html.Br(),

                    html.Strong("Transformation : "),
                    action_label,

                    # Affichage facultatif de la constante
                    html.Br() if "constant" in param else None,

                    (
                        html.Span([
                            html.Strong("Valeur de remplacement : "),
                            format_value(param.get("value"))
                        ])
                        if "constant" in param
                        else None
                    ),
                ])
            )

    return parameter_lines

def get_content(pipeline=None, cache=None):
    pipeline = pipeline or []
    cache = cache or {}

    loading_info = cache.get("chargement", {})

    filename = loading_info.get("filename")
    backend = loading_info.get("backend")
    row_count = loading_info.get("row_count")
    column_count = loading_info.get("column_count")
    processed_at = loading_info.get("processed_at")

    dataset_info = dbc.Card(
        [
            dbc.CardHeader("📄 Jeu de données"),
            dbc.CardBody(
                [
                    html.P([
                        html.Strong("Nom du fichier : "),
                        format_value(filename),
                    ]),
                    html.P([
                        html.Strong("Mode de traitement : "),
                        format_value(backend),
                    ]),
                    html.P([
                        html.Strong("Nombre de lignes : "),
                        format_value(row_count),
                    ]),
                    html.P([
                        html.Strong("Nombre de colonnes : "),
                        format_value(column_count),
                    ]),
                    html.P([
                        html.Strong("Chargé le : "),
                        format_value(processed_at),
                    ]),
                ]
            ),
        ],
        className="mb-4",
    )

    if not pipeline:
        pipeline_info = dbc.Alert(
            "Aucune transformation n'a encore été configurée.",
            color="info",
        )
    else:
        steps = []

        for index, step in enumerate(pipeline, start=1):
            step_name = step.get("step", "unknown")
            params = step.get("params", [])

            readable_step = STEP_LABELS.get(
                step_name,
                f"Étape « {step_name} »"
            )

            parameter_lines = build_parameter_lines(
                step_name=step_name,
                params=params
            )

            steps.append(
                dbc.Card(
                    dbc.CardBody([
                        html.H5(
                            f"{index}. {readable_step}",
                            className="card-title",
                        ),

                        html.P(
                            "Transformation prévue à l'export.",
                            className="text-muted",
                        ),

                        html.Ul(parameter_lines),
                    ]),
                    className="mb-3",
                )
            )

        pipeline_info = html.Div(steps)

    return html.Div(
        [
            html.H3("📜 Historique du traitement"),
            html.P(
                "Cette page présente les transformations configurées "
                "pour votre jeu de données."
            ),
            dataset_info,
            html.H4("🔧 Étapes prévues"),
            pipeline_info,
        ]
    )


def register_callbacks_history(app):
    @app.callback(
        Output("module-historique", "children"),
        [
            Input("pipeline-store", "data"),
            Input("module-cache", "data"),
        ],
    )
    def update_history(pipeline, cache):
        return get_content(pipeline=pipeline, cache=cache)