# app/modules/cleaning_submodules/missing_values.py
import dash_bootstrap_components as dbc
from dash import html, dcc, Input, Output, State, ALL, MATCH, ctx, no_update
from dash.exceptions import PreventUpdate
import pandas as pd
from typing import Any, Dict, List, Tuple, Optional

from app.modules.common.io import load_df, load_df_only, format_warning, get_column_dtype
from app.modules.chargement import show_dataset_preview
from app.modules.common.ui import STYLE_DROPDOWN, CHECKLIST_STYLE, CHECKLIST_INPUT_STYLE, CHECKLIST_LABEL_STYLE, get_dynamic_checklist_label_style

try:
    from pyspark.sql import functions as F
    HAS_SPARK = True
except Exception:
    HAS_SPARK = False

MISSING_OPTIONS = [
    {"label": "Supprimer les lignes", "value": "drop_rows"},
    {"label": "Moyenne", "value": "mean"},
    {"label": "Médiane", "value": "median"},
    {"label": "Mode", "value": "mode"},
    {"label": "Valeur précédente", "value": "ffill"},
    {"label": "Valeur suivante", "value": "bfill"},
    {"label": "Valeur personnalisée", "value": "constant"},
]

TAB_ID = "clean-missing"

# =========================================================
# Fonctions logique métier
# =========================================================

def apply_missings_pandas(df, rules, is_spark=None):
    """
    Applique les stratégies de traitement des valeurs manquantes
    sur une copie d'un DataFrame pandas.
    """

    pdf = df.copy()

    def check_column(column):
        if column not in pdf.columns:
            raise ValueError(
                f"La colonne '{column}' n'existe pas dans le dataset."
            )

    def get_constant_value(series, value):
        if value is None:
            raise ValueError(
                "Une valeur est requise pour la stratégie 'constant'."
            )

        if pd.api.types.is_numeric_dtype(series):
            try:
                return pd.to_numeric(value)
            except Exception:
                raise ValueError(
                    f"La valeur constante '{value}' "
                    "n'est pas numérique."
                )

        return value

    for rule in rules:
        column = rule.get("column")
        strategy = rule.get("strategy") or rule.get("action")

        # Selon le nom utilisé dans ton composant Dash
        constant = (
            rule.get("constant")
            if "constant" in rule
            else rule.get("value")
        )

        if not column or not strategy:
            raise ValueError(f"Règle de valeurs manquantes invalide : {rule}")

        check_column(column)

        series = pdf[column]

        if strategy == "drop_rows":
            pdf = pdf.loc[pdf[column].notna()].copy()

        elif strategy == "mean":
            if not pd.api.types.is_numeric_dtype(series):
                raise ValueError(
                    f"La moyenne ne peut être appliquée qu'à une colonne "
                    f"numérique : '{column}'."
                )

            fill_value = series.mean()

            if pd.isna(fill_value):
                raise ValueError(
                    f"Impossible de calculer la moyenne de '{column}' "
                    "car la colonne ne contient aucune valeur exploitable."
                )

            pdf[column] = series.fillna(fill_value)

        elif strategy == "median":
            if not pd.api.types.is_numeric_dtype(series):
                raise ValueError(
                    f"La médiane ne peut être appliquée qu'à une colonne "
                    f"numérique : '{column}'."
                )

            fill_value = series.median()

            if pd.isna(fill_value):
                raise ValueError(
                    f"Impossible de calculer la médiane de '{column}'."
                )

            pdf[column] = series.fillna(fill_value)

        elif strategy == "mode":
            mode_values = series.dropna().mode()

            if mode_values.empty:
                raise ValueError(
                    f"Impossible de calculer le mode de '{column}'."
                )

            pdf[column] = series.fillna(mode_values.iloc[0])

        elif strategy == "zero":
            if not pd.api.types.is_numeric_dtype(series):
                raise ValueError(
                    f"La valeur zéro ne peut être appliquée qu'à une "
                    f"colonne numérique : '{column}'."
                )

            pdf[column] = series.fillna(0)

        elif strategy == "empty":
            if pd.api.types.is_numeric_dtype(series):
                fill_value = 0
            else:
                fill_value = ""

            pdf[column] = series.fillna(fill_value)

        elif strategy == "ffill":
            pdf[column] = series.ffill()

        elif strategy == "bfill":
            pdf[column] = series.bfill()

        elif strategy == "constant":
            fill_value = get_constant_value(series, constant)
            pdf[column] = series.fillna(fill_value)

        else:
            raise ValueError(
                f"Stratégie de valeurs manquantes inconnue : '{strategy}'."
            )

    return pdf

def apply_missings_spark(df, rules, is_spark=None):
    """
    Applique les stratégies de valeurs manquantes sur un DataFrame Spark.

    Retourne un DataFrame Spark transformé.
    """

    if not HAS_SPARK:
        raise RuntimeError("Spark n'est pas disponible.")

    from pyspark.sql import functions as F
    from pyspark.sql.types import (
        NumericType,
        StringType,
        BooleanType,
        DateType,
        TimestampType,
    )

    sdf = df

    schema_types = {
        field.name: field.dataType
        for field in sdf.schema.fields
    }

    def check_column(column):
        if column not in schema_types:
            raise ValueError(
                f"La colonne '{column}' n'existe pas dans le dataset."
            )

    def missing_condition(column):
        condition = F.col(column).isNull()
        dtype = schema_types[column]

        if isinstance(dtype, NumericType):
            condition = condition | F.isnan(F.col(column))

        return condition

    def cast_constant(column, value):
        if value is None:
            raise ValueError(
                "Une valeur est requise pour la stratégie 'constant'."
            )

        dtype = schema_types[column]

        try:
            if isinstance(dtype, NumericType):
                return F.lit(float(str(value).replace(",", "."))).cast(dtype)

            if isinstance(dtype, BooleanType):
                normalized = str(value).strip().lower()

                if normalized in {"true", "1", "oui", "yes"}:
                    return F.lit(True)

                if normalized in {"false", "0", "non", "no"}:
                    return F.lit(False)

                raise ValueError(
                    f"La valeur constante '{value}' n'est pas booléenne."
                )

            if isinstance(dtype, DateType):
                return F.to_date(F.lit(value))

            if isinstance(dtype, TimestampType):
                return F.to_timestamp(F.lit(value))

            if isinstance(dtype, StringType):
                return F.lit(str(value))

            return F.lit(value).cast(dtype)

        except Exception as exc:
            raise ValueError(
                f"La valeur constante '{value}' est incompatible "
                f"avec la colonne '{column}'."
            ) from exc

    for rule in rules or []:
        column = rule.get("column")
        strategy = rule.get("strategy") or rule.get("action")
        constant = (
            rule.get("constant")
            if "constant" in rule
            else rule.get("value")
        )

        if not column or not strategy:
            raise ValueError(
                f"Règle de valeurs manquantes invalide : {rule}"
            )

        check_column(column)

        condition = missing_condition(column)
        column_expr = F.col(column)
        dtype = schema_types[column]

        if strategy == "drop_rows":
            sdf = sdf.filter(~condition)

        elif strategy == "mean":
            if not isinstance(dtype, NumericType):
                raise ValueError(
                    f"La moyenne ne peut être appliquée qu'à une colonne "
                    f"numérique : '{column}'."
                )

            fill_value = sdf.select(
                F.mean(
                    F.when(~missing_condition(column), column_expr)
                ).alias("value")
            ).first()["value"]

            if fill_value is None:
                raise ValueError(
                    f"Impossible de calculer la moyenne de '{column}'."
                )

            sdf = sdf.withColumn(
                column,
                F.when(condition, F.lit(fill_value))
                .otherwise(column_expr)
            )

        elif strategy == "median":
            if not isinstance(dtype, NumericType):
                raise ValueError(
                    f"La médiane ne peut être appliquée qu'à une colonne "
                    f"numérique : '{column}'."
                )

            median_values = sdf.approxQuantile(
                column,
                [0.5],
                0.01,
            )

            if not median_values:
                raise ValueError(
                    f"Impossible de calculer la médiane de '{column}'."
                )

            sdf = sdf.withColumn(
                column,
                F.when(condition, F.lit(median_values[0]))
                .otherwise(column_expr)
            )

        elif strategy == "mode":
            mode_row = (
                sdf
                .filter(~condition)
                .groupBy(column)
                .count()
                .orderBy(F.desc("count"))
                .first()
            )

            if mode_row is None:
                raise ValueError(
                    f"Impossible de calculer le mode de '{column}'."
                )

            mode_value = mode_row[column]

            sdf = sdf.withColumn(
                column,
                F.when(condition, F.lit(mode_value))
                .otherwise(column_expr)
            )

        elif strategy == "constant":
            constant_expr = cast_constant(column, constant)

            sdf = sdf.withColumn(
                column,
                F.when(condition, constant_expr)
                .otherwise(column_expr)
            )

        elif strategy == "ffill":
            raise ValueError(
                "La stratégie 'ffill' nécessite une colonne d'ordre "
                "explicitement définie pour Spark."
            )

        elif strategy == "bfill":
            raise ValueError(
                "La stratégie 'bfill' nécessite une colonne d'ordre "
                "explicitement définie pour Spark."
            )

        else:
            raise ValueError(
                f"Stratégie de valeurs manquantes inconnue : '{strategy}'."
            )

    return sdf


def build_missing_distribution_figure(df, column, title):
    import plotly.express as px
    if column not in df.columns:
        raise ValueError(
            f"La colonne '{column}' n'existe pas dans le dataset."
        )

    series = df[column]

    if pd.api.types.is_numeric_dtype(series):
        fig = px.histogram(
            df,
            x=column,
            nbins=30,
            title=title,
        )
    else:
        values = (
            series.astype("string")
            .fillna("<NA>")
            .value_counts()
            .reset_index()
        )

        values.columns = [column, "count"]

        fig = px.bar(
            values,
            x=column,
            y="count",
            title=title,
        )

    fig.update_layout(
        margin=dict(l= treinta if False else 40, r=20, t=50, b=40),
        height=400,
    )

    return fig


def analyze_missing_values(df, is_spark=False):
    if is_spark:
        return _analyze_missing_values_spark(df)

    return _analyze_missing_values_pandas(df)

def _analyze_missing_values_pandas(df):
    row_count = len(df)

    analysis = {
        "row_count": row_count,
        "columns": {},
    }

    for column in df.columns:
        series = df[column]

        missing_count = int(series.isna().sum())
        non_missing_count = int(series.notna().sum())

        analysis["columns"][column] = {
            "dtype": str(series.dtype),
            "missing_count": missing_count,
            "missing_ratio": round(
                missing_count / row_count * 100,
                2
            ) if row_count else 0,
            "non_missing_count": non_missing_count,
            "unique_count": int(series.nunique(dropna=True)),
        }

    return analysis

def _analyze_missing_values_spark(df):
    from pyspark.sql import functions as F
    from pyspark.sql.types import FloatType, DoubleType    

    row_count = df.count()

    analysis = {
        "row_count": row_count,
        "columns": {},
    }

    for field in df.schema.fields:
        column = field.name
        data_type = field.dataType
        column_expr = F.col(column)

        missing_condition = column_expr.isNull()

        if isinstance(data_type, (FloatType, DoubleType)):
            missing_condition = (
                column_expr.isNull()
                | F.isnan(column_expr)
            )

        missing_count = (
            df.filter(missing_condition).count()
        )

        non_missing_count = row_count - missing_count

        unique_count = (
            df.select(column)
            .where(column_expr.isNotNull())
            .distinct()
            .count()
        )

        analysis["columns"][column] = {
            "dtype": str(data_type),
            "missing_count": missing_count,
            "missing_ratio": round(
                missing_count / row_count * 100,
                2
            ) if row_count else 0,
            "non_missing_count": non_missing_count,
            "unique_count": unique_count,
        }

    return analysis

def get_missing_options(dtype, is_spark=False):
    """
    Retourne les stratégies compatibles avec le type actuel.
    """

    if is_spark:
        from pyspark.sql.types import (
            StringType,
            BooleanType,
            NumericType,
            DateType,
            TimestampType,
        )

        if isinstance(dtype, NumericType):
            allowed = {
                "drop_rows",
                "mean",
                "median",
                "mode",
                "constant",
                "ffill",
                "bfill",
            }

        elif isinstance(dtype, (StringType, BooleanType)):
            allowed = {
                "drop_rows",
                "mode",
                "constant",
                "ffill",
                "bfill",
            }

        elif isinstance(dtype, (DateType, TimestampType)):
            allowed = {
                "drop_rows",
                "mode",
                "constant",
                "ffill",
                "bfill",
            }

        else:
            allowed = {
                "drop_rows",
                "constant",
            }

    else:
        import pandas as pd

        if pd.api.types.is_numeric_dtype(dtype):
            allowed = {
                "drop_rows",
                "mean",
                "median",
                "mode",
                "constant",
                "ffill",
                "bfill",
            }

        elif (
            pd.api.types.is_string_dtype(dtype)
            or pd.api.types.is_object_dtype(dtype)
            or pd.api.types.is_bool_dtype(dtype)
        ):
            allowed = {
                "drop_rows",
                "mode",
                "constant",
                "ffill",
                "bfill",
            }

        elif pd.api.types.is_datetime64_any_dtype(dtype):
            allowed = {
                "drop_rows",
                "mode",
                "constant",
                "ffill",
                "bfill",
            }

        else:
            allowed = {
                "drop_rows",
                "constant",
            }

    return [
        option
        for option in MISSING_OPTIONS
        if option["value"] in allowed
    ]

# Récupère les stratégies déjà enregistrées dans le pipeline.
def get_missing_rules_from_pipeline(pipeline):
    if not pipeline:
        return {}

    rules_by_column = {}

    for step in pipeline:
        if step.get("step") != "missing_values":
            continue

        for rule in step.get("params", []):
            column = rule.get("column")
            action = rule.get("action")

            if not column or not action:
                continue

            if column and action:
                # La dernière règle gagne
                rules_by_column[column] = action

    return rules_by_column

def vals_by_col(ids, vals):
    out = {}
    ids = ids or []
    vals = vals or []
    for i, item_id in enumerate(ids):
        if isinstance(item_id, dict) and "column" in item_id:
            v = vals[i] if i < len(vals) else None
            out[item_id["column"]] = v
    return out
    
def preview_fig_pandas(df, column):
    from plotly import express as px
    if column not in df.columns:
        return px.histogram(pd.Series([], name=column), title=f"{column} (absente)")
    s = df[column]
    na = int(s.isna().sum()) if hasattr(s, "isna") else int(pd.isna(s).sum())
    if pd.api.types.is_numeric_dtype(s):
        nbins = 50 if s.size < 5000 else 60
        fig = px.histogram(s, nbins=nbins, title=f"Distribution de {column} (NA: {na})", height=300)
        return fig
    else:
        vc = s.astype(str).value_counts(dropna=False).nlargest(50)
        return px.bar(x=vc.index, y=vc.values,
                        title=f"Top modalités de {column} (NA: {na})",
                        labels={"x": column, "y": "count"}, height=300)




def build_missing_rule_row(column, dtype, selected_action=None, is_spark=False, missing_analysis=None):
    options = get_missing_options(dtype, is_spark=is_spark)

    if missing_analysis is None:
        missing_analysis = {}
    else:
        column_analysis = missing_analysis.get("columns", {}).get(column, {})
        missing_count = column_analysis.get("missing_count", 0)
        missing_ratio = column_analysis.get("missing_ratio", 0)
        non_missing_count = column_analysis.get("non_missing_count", 0)

    if missing_count > 0:
        options = [
            {
                "label": "Supprimer les lignes",
                "value": "drop_rows",
            },
            *[
                option
                for option in options
                if option["value"] != "drop_rows"
            ],
        ]

    badge_style = {
        "fontSize": "15px",
        # "lineHeight": "1.2",
        # "padding": "6px 8px",
    }
    
    column_info = html.Div(
        [
            html.Strong(
                column,
                className="me-3"
            ),

            html.Div(
                [
                    dbc.Badge(
                        f"Type : {dtype}",
                        color="secondary",
                        className="me-1",
                        style=badge_style
                    ),
                    dbc.Badge(
                        f"Nbre de NA : {missing_count}",
                        color="danger" if missing_count else "success",
                        className="me-1",
                        style=badge_style
                    ),
                    dbc.Badge(
                        f"Nbre de non NA : {non_missing_count}",
                        color="warning" if missing_count else "success",
                        className="me-1",
                        style=badge_style
                    ),
                    dbc.Badge(
                        f"Ratio : {missing_ratio} %",
                        color="info" if missing_count else "success",
                        className="me-1",
                        style=badge_style
                    ),
                ],
                className="d-flex"
            ),
        ],
        className="d-flex align-items-left flex-wrap"
    )

    return dbc.Row([
        dbc.Col(column_info, xs=12, md=4, className="pe-1 d-flex align-items-center"),

        dbc.Col(
            dcc.Dropdown(
                id={
                    "type": "miss-strategy",
                    "column": column,
                },
                options=options,
                style={**STYLE_DROPDOWN, "marginBottom": "0px"},
                value=selected_action,
                placeholder="Choisir une méthode",
                clearable=True,
            ),
            xs=12, md="auto", className="px-1 d-flex align-items-center",
        ),

        dbc.Col(
            dbc.Input(
                id={
                    "type": "miss-constant",
                    "column": column,
                },
                type="text",
                placeholder="Valeur constante",
                # disabled=not bool(selected_action),
                # style={"display": "none"},
            ),
            xs=12, md="auto", className="px-1 d-flex align-items-center",
        ),

        dbc.Col(
                dbc.Button(
                    "Visualiser",
                    id={"type": "miss-visualize", "column": column},
                    color="primary",
                    outline=True,
                    disabled=True,
                ),
                width="auto",
                xs=12, md="auto", className="px-1 d-flex align-items-center",
        ),
    ],
    className="mb-2 g-0 align-items-center"
    )

# =========================================================
# Fonctions DASH
# =========================================================

def get_tab():
    return dbc.Tab(tab_id=TAB_ID, label="Formats")

def get_layout(STYLE_DROPDOWN):    
    return html.Div([
        html.P(
            "Remplacement des valeurs manquantes (choix de la méthode, visualiser l'impact puis appliquer globalement) :",
            style={"fontSize": "18px", "marginBottom": "10px"},
        ),
        dcc.Store(id="missing-analysis-store", data={}),
        dbc.Col([
            dbc.Button("🔄 Réinitialiser", id="miss-reset", style={"display": "none", "marginTop": "6px", "marginBottom": "6px"}),
        ]),
        dbc.Row([
            dbc.Col([
                html.P(
                    "Sélectionnez les colonnes à traiter :",
                    style={"fontSize": "17px", "marginBottom": "10px", "marginTop": "10px"}
                ),
                html.Div(
                    id="missing-no-values-message",
                    style={"marginTop": "10px"}
                ),
                dcc.Checklist(
                    id="miss-columns",
                    options=[],
                    value=[],
                    style=CHECKLIST_STYLE,
                    inputStyle=CHECKLIST_INPUT_STYLE,
                    labelStyle=CHECKLIST_LABEL_STYLE,
                ),
            ], md=10)
        ]),
        html.Hr(),

        html.Div(
            id="miss-rules-container",
            className="mt-3"
        ),

        dbc.Button(
            "Appliquer",
            id="miss-apply",
            color="primary",
            className="mt-3"
        ),

        html.Div(
            id="miss-feedback",
            className="mt-2"
        ),

        # Modal global pour l'aperçu Avant/Après
        dbc.Modal(
            [
                dbc.ModalHeader(dbc.ModalTitle(id="miss-visual-title")),
                dbc.ModalBody(id="miss-visual-body"),
                dbc.ModalFooter(
                    dbc.Button("Fermer", id="miss-visual-close", color="secondary")
                ),
            ],
            id="miss-visual-modal",
            is_open=False,
            size="xl",
            scrollable=False,
            backdrop="static",
        ),
    ])


def register_callbacks(app):
    @app.callback(
        Output({"type": "miss-constant", "column": MATCH}, "style"),
        Input({"type": "miss-strategy", "column": MATCH}, "value"),
    )
    def toggle_constant_input(strategy):
        return {
            "display": "block" if strategy == "constant" else "none"
        }

    @app.callback(
        Output("miss-reset", "style"),
        Input("pipeline-store", "data")
    )
    def update_missing_reset_button(pipeline):
        has_missing = any(
            step.get("step") == "missing_values"
            for step in (pipeline or [])
        )

        return {
            "display": "inline-block" if has_missing else "none",
        }

    @app.callback(
        Output("missing-analysis-store", "data"),
        Output("miss-columns", "options"),
        Output("miss-columns", "value"),
        Output("miss-columns", "labelStyle"),
        Output("missing-no-values-message", "children"),
        Input("original-parquet-path-store", "data"),
        Input("pipeline-store", "data"),
        prevent_initial_call=False
    )
    def populate_missing_checklist(path, pipeline):
        df, is_spark = load_df(path)

        if df is None:
            return {}, [], [], CHECKLIST_LABEL_STYLE, "Aucun dataset chargé."

        # Analyse des valeurs manquantes
        missing_analysis = analyze_missing_values(df, is_spark=is_spark)

        # Règles déjà enregistrées dans le pipeline
        existing_rules = get_missing_rules_from_pipeline(pipeline)

        options = [
            {
                "label": (
                    f"{column} "
                    f"(NA : "
                    f"{missing_analysis['columns'][column]['missing_count']})"
                ),
                "value": column
            }
            for column in df.columns
            if missing_analysis["columns"][column]["missing_count"] >= 1
        ]

        if not options:
            message = html.Div(
                "✅ Aucune valeur manquante détectée. Vous pouvez passer à la suite.",
                className="success-message",
            )
        else:
            message = ""

        # Seules les colonnes ayant déjà une règle sont sélectionnées
        available_columns = [option["value"] for option in options]

        selected_columns = [
            column
            for column in available_columns
            if column in existing_rules
        ]

        label_style = get_dynamic_checklist_label_style(options)

        return missing_analysis, options, selected_columns, label_style, message

    @app.callback(
        Output("miss-rules-container", "children"),
        Input("miss-columns", "value"),
        Input("miss-columns", "options"),
        Input("pipeline-store", "data"),
        State("original-parquet-path-store", "data"),
        State("missing-analysis-store", "data"),
    )
    def render_missing_rules(selected_columns, options, pipeline, path, missing_analysis):
        if not path:
            return html.Div(
                "Impossible de charger le dataset.",
                className="text-danger"
            )

        selected_columns = selected_columns or []

        df, is_spark = load_df(path)

        if df is None:
            return html.Div(
                "Impossible de charger le dataset.",
                className="text-danger"
            )

        if not options:
            return html.Div("")

        existing_rules = get_missing_rules_from_pipeline(pipeline)
        rows = []

        for column in selected_columns:
            try:
                dtype = get_column_dtype(
                    df,
                    column,
                    is_spark=is_spark
                )

                selected_action = existing_rules.get(column)

                rows.append(
                    build_missing_rule_row(
                        column=column,
                        dtype=dtype,
                        selected_action=selected_action,
                        is_spark=is_spark,
                        missing_analysis=missing_analysis
                    )
                )

            except Exception as exc:
                rows.append(dbc.Alert(f"Erreur pour '{column}' : {exc}", color="danger"))

        return rows

    @app.callback(
        Output({"type": "miss-visualize", "column": MATCH}, "disabled",),
        Input({"type": "miss-strategy", "column": MATCH},"value",),
    )
    def toggle_visualize_button(strategy):
        return not bool(strategy)

    @app.callback(
        Output("miss-visual-modal", "is_open"),
        Output("miss-visual-body", "children"),
        Output("miss-visual-title", "children"),

        Input({"type": "miss-visualize", "column": ALL}, "n_clicks"),
        Input("miss-visual-close", "n_clicks"),

        State({"type": "miss-visualize", "column": ALL}, "id"),
        State({"type": "miss-strategy", "column": ALL}, "id"),
        State({"type": "miss-strategy", "column": ALL}, "value"),
        State({"type": "miss-constant", "column": ALL}, "id"),
        State({"type": "miss-constant", "column": ALL}, "value"),
        State("original-parquet-path-store", "data"),

        prevent_initial_call=True,
    )
    def open_missing_preview(visualize_clicks, close_click, visualize_ids, strategy_ids, strategy_values, constant_ids, constant_values, original_path,):
        triggered_id = ctx.triggered_id

        # Fermeture du modal
        if triggered_id == "miss-visual-close":
            return False, no_update, no_update

        # Le déclencheur doit être un bouton de visualisation
        if not isinstance(triggered_id, dict):
            raise PreventUpdate

        if triggered_id.get("type") != "miss-visualize":
            raise PreventUpdate

        # Vérifie que le bouton a réellement été cliqué
        clicked_index = next(
            (
                index
                for index, button_id in enumerate(visualize_ids or [])
                if button_id == triggered_id
            ),
            None,
        )

        if clicked_index is None:
            raise PreventUpdate

        if not visualize_clicks or not visualize_clicks[clicked_index]:
            raise PreventUpdate

        column = triggered_id["column"]

        # Récupération de la stratégie sélectionnée
        strategy_map = vals_by_col(strategy_ids, strategy_values)
        strategy = strategy_map.get(column)

        if not strategy:
            return (True, dbc.Alert("Veuillez sélectionner une stratégie avant de visualiser.", color="warning",), f"Prévisualisation — {column}",)

        # Récupération de la constante éventuelle
        constant_map = vals_by_col(constant_ids, constant_values)
        constant = constant_map.get(column)

        if strategy == "constant" and constant in (None, ""):
            return (True, dbc.Alert("Veuillez renseigner une valeur constante.", color="warning",), f"Prévisualisation — {column}",)

        if not original_path:
            return (True, dbc.Alert("Aucun dataset original n'est disponible.", color="warning",), "Prévisualisation",)

        try:
            df, is_spark = load_df(original_path)

            if df is None:
                raise ValueError("Impossible de charger le dataset original.")

            rule = [{
                "column": column,
                "strategy": strategy,
                "constant": constant,
            }]

            # La visualisation actuelle repose sur Pandas
            if is_spark:
                df_before = df.select(column).toPandas()
                df_after = apply_missings_spark(df, rule, is_spark=True,).select(column).toPandas()
            else:
                df_before = df[[column]].copy()
                df_after = apply_missings_pandas(df, rule, is_spark=False,)[[column]]

            before_count = int(df_before[column].isna().sum())
            after_count = int(df_after[column].isna().sum())

            fig_before = preview_fig_pandas(df_before, column,)
            fig_after = preview_fig_pandas(df_after, column,)

            body = html.Div([
                dbc.Alert(f"Stratégie simulée : {strategy} — "f"valeurs manquantes : "f"{before_count} → {after_count}",color="info",),
                dbc.Row([
                    dbc.Col(html.Div([html.H6("Avant (original)"), dcc.Graph(figure=fig_before, style={"height": "38vh"})]), md=6),
                    dbc.Col(html.Div([html.H6("Après (simulation)"), dcc.Graph(figure=fig_after, style={"height": "38vh"})]), md=6),
                ], className="g-3")
            ])

            return (True, body, f"Impact du traitement — {column}",)

        except Exception as exc:
            return (True, dbc.Alert(f"Impossible de simuler la stratégie : {exc}", color="danger",), f"Erreur — {column}",)

    @app.callback(
        Output("pipeline-store", "data", allow_duplicate=True),
        Output("miss-feedback", "children"),
        Input("miss-apply", "n_clicks"),
        Input("miss-reset", "n_clicks"),
        State("original-parquet-path-store", "data"),
        State({"type": "miss-strategy", "column": ALL}, "value"),
        State({"type": "miss-strategy", "column": ALL}, "id"),
        State({"type": "miss-constant", "column": ALL}, "value"),
        State({"type": "miss-constant", "column": ALL}, "id"),
        State("pipeline-store", "data"),
        prevent_initial_call=True,
    )
    def on_apply_missing_click(apply_clicks, reset_clicks, path, strategies, strategy_ids, constant_values, constant_ids, pipeline,):
        from app.modules.common.pipeline import add_step, reset_step

        pipeline = pipeline or []

        if not ctx.triggered_id:
            raise PreventUpdate

        ### Réinitialisation des stratégies
        if ctx.triggered_id == "miss-reset":
            if not reset_clicks or reset_clicks < 1:
                raise PreventUpdate

            updated_pipeline = reset_step(pipeline, "missing_values",)

            return (updated_pipeline, dbc.Alert("✅ Les stratégies de valeurs manquantes ont été réinitialisées.", color="info",),)

        ### Application des stratégies
        if ctx.triggered_id != "miss-apply":
            raise PreventUpdate

        if not path:
            return pipeline, format_warning("Aucun dataset chargé.")

        # Association constante -> colonne
        constants_by_column = {}

        for constant_id, constant_value in zip(constant_ids or [], constant_values or [],):
            if (isinstance(constant_id, dict) and "column" in constant_id):
                constants_by_column[constant_id["column"]] = constant_value

        rules = []

        for strategy, strategy_id in zip(
            strategies or [],
            strategy_ids or [],
        ):
            if not strategy:
                continue

            column = strategy_id.get("column")

            rule = {
                "column": column,
                "action": strategy,
            }

            if strategy == "constant":
                constant = constants_by_column.get(column)

                if constant in (None, ""):
                    return pipeline, format_warning(
                        f"Veuillez renseigner une valeur constante pour « {column} »."
                    )

                rule["value"] = constant

            rules.append(rule)

        if not rules:
            return pipeline, format_warning("Aucune stratégie de valeurs manquantes sélectionnée.")

        try:
            df, is_spark = load_df(path)

            if df is None:
                return pipeline, format_warning("Aucun dataset chargé.")

            if not rules:
                return df
                
            # Détection automatique du moteur
            if is_spark is None:
                is_spark = (hasattr(df, "withColumn") and hasattr(df, "schema"))

            # Validation des règles sans modifier le dataset source
            if is_spark:
                apply_missings_spark(df, rules, is_spark=True,)
            else:
                apply_missings_pandas(df, rules, is_spark=False,)

            new_step = {
                "step": "missing_values",
                "params": rules,
            }

            updated_pipeline = add_step(pipeline or [], new_step,)

            return (updated_pipeline, dbc.Alert(f"✅ {len(rules)} stratégie(s) de valeurs manquantes enregistrée(s).", color="success",),)

        except Exception as exc:
            return (pipeline, format_warning("Erreur lors de la configuration des valeurs " f"manquantes : {exc}"),)
