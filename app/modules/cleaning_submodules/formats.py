# app/modules/nettoyage_submodules/formats.py
import dash_bootstrap_components as dbc
from dash import html, dcc, Input, Output, State, ALL, ctx
from dash.exceptions import PreventUpdate
import pandas as pd

from app.modules.common.io import load_df, format_warning
from app.modules.chargement import show_dataset_preview
from app.modules.common.ui import STYLE_DROPDOWN, CHECKLIST_STYLE, CHECKLIST_INPUT_STYLE, CHECKLIST_LABEL_STYLE

try:
    from pyspark.sql import functions as F
    HAS_SPARK = True
except Exception:
    HAS_SPARK = False

FORMAT_OPTIONS = [
    {"label": "Cast en numérique", "value": "to_numeric"},
    {"label": "Cast en date", "value": "to_date"},
    {"label": "Cast en chaîne", "value": "to_string"},
    {"label": "Cast en entier", "value": "to_integer"},
    {"label": "Cast en booléen", "value": "to_boolean"},
]
BOOLEAN_TRUE_VALUES = {"true", "vrai", "yes", "oui", "1"}
BOOLEAN_FALSE_VALUES = {"false", "faux", "no", "non", "0"}
BOOLEAN_VALUES = BOOLEAN_TRUE_VALUES | BOOLEAN_FALSE_VALUES

TAB_ID = "clean-formats"

# =========================================================
# Fonctions logique métier
# =========================================================

def get_invalid_examples(series, invalid_mask, limit=5):
    return (
        series[invalid_mask]
        .astype("string")
        .drop_duplicates()
        .head(limit)
        .tolist()
    )


def raise_if_invalid(column, action, source, converted):
    invalid_mask = source.notna() & converted.isna()

    if invalid_mask.any():
        examples = get_invalid_examples(source, invalid_mask)

        raise ValueError(
            f"Conversion '{action}' impossible sur '{column}'. "
            f"{invalid_mask.sum()} valeur(s) invalide(s). "
            f"Exemples : {examples}"
        )

def normalize_boolean_series(series):
    return (
        series.dropna()
        .astype("string")
        .str.strip()
        .str.lower()
    )


def is_boolean_compatible(series):
    values = set(normalize_boolean_series(series).unique())

    return (
        bool(values)
        and len(values) == 2
        and values.issubset(BOOLEAN_VALUES)
        and bool(values & BOOLEAN_TRUE_VALUES)
        and bool(values & BOOLEAN_FALSE_VALUES)
    )

def apply_formats_spark(df, rules, is_spark=None):
    if is_spark:
        from pyspark.sql import functions as F
        from pyspark.sql.types import (
            StringType,
            NumericType,
            DateType,
            TimestampType
        )

        print("=== COLONNES SPARK ===")
        print(df.columns)

        for rule in rules:
            print("=== REGLE SPARK ===")
            print(rule)
            print(df.select(rule["column"]).limit(5).collect())

        sdf = df
        spark_types = dict(sdf.dtypes)
        schema_types = {
            field.name: field.dataType
            for field in sdf.schema.fields
        }

        def check_column(column):
            if column not in sdf.columns:
                raise ValueError(
                    f"La colonne '{column}' n'existe pas dans le dataset."
                )

        def spark_invalid_values(sdf, column, converted_expr):
            source_expr = F.col(column)

            invalid_condition = (
                source_expr.isNotNull()
                & converted_expr.isNull()
            )

            invalid_count = sdf.filter(invalid_condition).count()

            if invalid_count > 0:
                examples = [
                    row[column]
                    for row in (
                        sdf
                        .filter(invalid_condition)
                        .select(column)
                        .limit(5)
                        .collect()
                    )
                ]

                raise ValueError(
                    f"Conversion impossible sur '{column}'. "
                    f"{invalid_count} valeur(s) invalide(s). "
                    f"Exemples : {examples}"
                )

        for rule in rules:
            column = rule.get("column")
            action = rule.get("action")

            if not column or not action:
                raise ValueError(f"Règle invalide : {rule}")

            check_column(column)
            data_type = schema_types[column]

            if action == "to_numeric":
                if isinstance(data_type, (DateType, TimestampType)):
                    raise ValueError(
                        f"Conversion numérique impossible sur '{column}' "
                        f"car la colonne contient des dates."
                    )

                converted = F.regexp_replace(
                    F.col(column).cast("string"),
                    ",",
                    "."
                ).cast("double")

                spark_invalid_values(
                    sdf,
                    column,
                    converted
                )

                sdf = sdf.withColumn(column, converted)

            elif action == "to_integer":
                if isinstance(data_type, (DateType, TimestampType)):
                    raise ValueError(
                        f"Conversion entière impossible sur '{column}' "
                        f"car la colonne contient des dates."
                    )

                source = F.col(column)

                converted_numeric = F.regexp_replace(
                    source.cast("string"),
                    ",",
                    "."
                ).cast("double")

                invalid_condition = (
                    source.isNotNull()
                    & (
                        converted_numeric.isNull()
                        | (converted_numeric % 1 != 0)
                    )
                )

                invalid_count = sdf.filter(invalid_condition).count()

                if invalid_count > 0:
                    examples = [
                        row[column]
                        for row in (
                            sdf
                            .filter(invalid_condition)
                            .select(column)
                            .limit(5)
                            .collect()
                        )
                    ]

                    raise ValueError(
                        f"Conversion entière impossible sur '{column}'. "
                        f"Valeurs invalides ou décimales : {examples}"
                    )

                sdf = sdf.withColumn(
                    column,
                    converted_numeric.cast("int")
                )

            elif action == "to_string":
                sdf = sdf.withColumn(
                    column,
                    F.col(column).cast("string")
                )

            elif action == "to_date":
                if isinstance(data_type, NumericType):
                    raise ValueError(
                        f"Conversion en date impossible sur '{column}' "
                        f"car son type est numérique."
                    )

                converted = F.to_date(
                    F.col(column).cast("string"),
                    "yyyy-MM-dd"
                )

                spark_invalid_values(
                    sdf,
                    column,
                    converted
                )

                sdf = sdf.withColumn(column, converted)

            elif action == "to_boolean":
                source = F.lower(
                    F.trim(
                        F.col(column).cast("string")
                    )
                )

                true_values = list(BOOLEAN_TRUE_VALUES)
                false_values = list(BOOLEAN_FALSE_VALUES)

                normalized_values = (
                    sdf
                    .select(source.alias("value"))
                    .where(F.col(column).isNotNull())
                    .distinct()
                    .limit(3)
                    .collect()
                )

                values = {
                    row["value"]
                    for row in normalized_values
                }

                is_compatible = (
                    len(values) == 2
                    and values.issubset(BOOLEAN_VALUES)
                    and bool(values & BOOLEAN_TRUE_VALUES)
                    and bool(values & BOOLEAN_FALSE_VALUES)
                )

                if not is_compatible:
                    raise ValueError(
                        f"Conversion en booléen impossible sur '{column}'. "
                        "La colonne doit contenir exactement deux valeurs "
                        "booléennes reconnues, hors valeurs nulles."
                    )

                converted = (
                    F.when(source.isin(true_values), F.lit(True))
                    .when(source.isin(false_values), F.lit(False))
                    .otherwise(F.lit(None))
                    .cast("boolean")
                )

                sdf = sdf.withColumn(column, converted)

            else:
                raise ValueError(
                    f"Action de format inconnue : '{action}'."
                )

        return sdf

def apply_formats_pandas(df, rules, is_spark=None):
    pdf = df.copy()

    def check_column(column):
        if column not in pdf.columns:
            raise ValueError(
                f"La colonne '{column}' n'existe pas dans le dataset."
            )

    for rule in rules:
        column = rule.get("column")
        action = rule.get("action")

        if not column or not action:
            raise ValueError(f"Règle invalide : {rule}")

        check_column(column)

        dtype = pdf[column].dtype
        dtype_name = str(dtype)

        if action == "to_numeric":
            if (pd.api.types.is_datetime64_any_dtype(dtype)
            or pd.api.types.is_timedelta64_dtype(dtype)):
                raise ValueError(
                    f"Conversion numérique impossible sur '{column}' "
                    f"car son type est '{dtype_name}'."
                )

            source = pdf[column]

            normalized = (
                source.astype("string")
                .str.replace(",", ".", regex=False)
            )

            converted = pd.to_numeric(
                normalized,
                errors="coerce"
            )

            raise_if_invalid(
                column,
                action,
                source,
                converted
            )

            pdf[column] = converted

        elif action == "to_integer":
            source = pdf[column]

            normalized = (
                source.astype("string")
                .str.replace(",", ".", regex=False)
            )

            numeric_values = pd.to_numeric(
                normalized,
                errors="coerce"
            )

            invalid_mask = source.notna() & numeric_values.isna()
            fractional_mask = (
                numeric_values.notna()
                & numeric_values.mod(1).ne(0)
            )

            if invalid_mask.any() or fractional_mask.any():
                invalid_values = source[
                    invalid_mask | fractional_mask
                ].astype("string").drop_duplicates().head(5).tolist()

                raise ValueError(
                    f"Conversion entière impossible sur '{column}'. "
                    f"La colonne contient des valeurs invalides ou décimales : "
                    f"{invalid_values}"
                )

            pdf[column] = numeric_values.astype("Int64")

        elif action == "to_string":
            pdf[column] = pdf[column].astype("string")

        elif action == "to_date":
            if pd.api.types.is_numeric_dtype(dtype):
                raise ValueError(
                    f"Conversion en date impossible sur '{column}' "
                    f"car son type est '{dtype_name}'."
                )

            source = pdf[column]

            converted = pd.to_datetime(
                source,
                errors="coerce"
            )

            raise_if_invalid(
                column,
                action,
                source,
                converted
            )

            pdf[column] = converted

        elif action == "to_boolean":
            source = pdf[column]

            if not is_boolean_compatible(source):
                raise ValueError(
                    f"Conversion en booléen impossible sur '{column}'. "
                    "La colonne doit contenir exactement deux valeurs "
                    "booléennes reconnues, hors valeurs nulles."
                )

            normalized = normalize_boolean_series(source)

            mapping = {
                **{
                    value: True
                    for value in BOOLEAN_TRUE_VALUES
                },
                **{
                    value: False
                    for value in BOOLEAN_FALSE_VALUES
                }
            }

            converted = normalized.map(mapping).astype("boolean")

            pdf[column] = converted

        else:
            raise ValueError(
                f"Action de format inconnue : '{action}'."
            )

    return pdf

def get_column_dtype(df, column, is_spark=False):
    if is_spark:
        field = next(
            (
                field
                for field in df.schema.fields
                if field.name == column
            ),
            None
        )

        if field is None:
            raise ValueError(
                f"La colonne '{column}' n'existe pas dans le dataset."
            )

        return field.dataType

    if column not in df.columns:
        raise ValueError(
            f"La colonne '{column}' n'existe pas dans le dataset."
        )

    return df[column].dtype

# Retourne les stratégies compatibles avec le type actuel.
def get_format_options(dtype, is_spark=False):

    if is_spark:
        from pyspark.sql.types import (
            StringType,
            NumericType,
            DateType,
            TimestampType
        )

        if isinstance(dtype, StringType):
            allowed = {
                "to_numeric",
                "to_integer",
                "to_date",
                "to_string",
                "to_boolean",
            }

        elif isinstance(dtype, NumericType):
            allowed = {
                "to_numeric",
                "to_integer",
                "to_string",
                "to_boolean",
            }

        elif isinstance(dtype, (DateType, TimestampType)):
            allowed = {
                "to_date",
                "to_string",
            }

        else:
            allowed = {"to_string"}

    else:

        if pd.api.types.is_string_dtype(dtype) or \
           pd.api.types.is_object_dtype(dtype):
            allowed = {
                "to_numeric",
                "to_integer",
                "to_date",
                "to_string",
                "to_boolean",
            }

        elif pd.api.types.is_numeric_dtype(dtype):
            allowed = {
                "to_numeric",
                "to_integer",
                "to_string",
                "to_boolean",
            }

        elif pd.api.types.is_datetime64_any_dtype(dtype):
            allowed = {
                "to_date",
                "to_string",
            }

        else:
            allowed = {"to_string"}

    return [
        option
        for option in FORMAT_OPTIONS
        if option["value"] in allowed
    ]

def get_format_rules_from_pipeline(pipeline):
    if not pipeline:
        return {}
    print(f"--------- Pipeline : {pipeline} ---------")
    rules_by_column = {}

    for step in pipeline:
        if step.get("step") != "formats":
            continue

        for rule in step.get("params", []):
            column = rule.get("column")
            action = rule.get("action")

            if column and action:
                # La dernière règle gagne
                rules_by_column[column] = action

    return rules_by_column

def build_format_rule_row(column, current_dtype, selected_action=None, is_spark=False,):
    options = get_format_options(
        current_dtype,
        is_spark=is_spark
    )

    return dbc.Row([
        dbc.Col(
            html.Strong(column),
            xs=12,
            md=2
        ),
        dbc.Col(
            html.Span(
                f"Type actuel : {current_dtype}"
            ),
            xs=12,
            md=2
        ),
        dbc.Col(
            dcc.Dropdown(
                id={
                    "type": "fmt-strategy",
                    "column": column,
                },
                options=options,
                value=selected_action,
                clearable=True,
                placeholder="Choisir une stratégie",
                style=STYLE_DROPDOWN,
            ),
            xs=12,
            md=8
        ),
    ], className="mb-2")

# =========================================================
# Fonctions DASH
# =========================================================

def get_tab():
    return dbc.Tab(tab_id=TAB_ID, label="Formats")

def get_layout(STYLE_DROPDOWN):    
    return html.Div([
        html.P(
            "Conversion du type des colonnes (choisir la conversion puis appliquer) :",
            style={"fontSize": "18px", "marginBottom": "10px"},
        ),
        dbc.Row([
            html.Div(id="original-dataset-preview", style={"marginBottom": "15px"})
        ]),
        dbc.Col([
            dbc.Button("🔄 Réinitialiser", id="fmt-reset", style={"display": "none", "marginTop": "6px", "marginBottom": "6px"}),
        ]),
        dbc.Row([
            dbc.Col([
                html.P(
                    "Sélectionnez les colonnes à traiter :",
                    style={"fontSize": "17px", "marginBottom": "10px", "marginTop": "10px"}
                ),

                dcc.Checklist(
                    id="fmt-columns",
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
            id="fmt-rules-container",
            className="mt-3"
        ),

        dbc.Button(
            "Appliquer",
            id="fmt-apply",
            color="primary",
            className="mt-3"
        ),

        html.Div(
            id="fmt-feedback",
            className="mt-2"
        )
    ])

def register_callbacks(app):  
    @app.callback(
        Output("original-dataset-preview", "children"),
        Input("df-json-store", "data")
    )
    def update_original_dataset_preview(df_json):
        if not df_json:
            return html.Div("⚠️ Aucun aperçu disponible.")

        return show_dataset_preview(df_json, n_rows=3)
    
    @app.callback(
        Output("fmt-reset", "style"),
        Input("pipeline-store", "data"),
        Input("cleaning-subtabs", "active_tab")
    )
    def update_formats_reset_button(pipeline, active_tab):
        has_formats = any(
            step.get("step") == "formats"
            for step in (pipeline or [])
        )

        return {
            "display": "inline-block"
            if active_tab == TAB_ID and has_formats
            else "none",
            "marginTop": "6px",
            "marginBottom": "6px",
        }
    
    @app.callback(
    Output("fmt-columns", "options"),
    Output("fmt-columns", "value"),
    Input("original-parquet-path-store", "data"),
    Input("pipeline-store", "data"),
    prevent_initial_call=False
    )
    def populate_format_checklist(path, pipeline):
        df, is_spark = load_df(path)

        if df is None:
            return [], []

        columns = list(df.columns)
        existing_rules = get_format_rules_from_pipeline(pipeline)

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

        return options, selected_columns

    @app.callback(
        Output("fmt-rules-container", "children"),
        Input("fmt-columns", "value"),
        Input("pipeline-store", "data"),
        State("original-parquet-path-store", "data"),
    )
    def render_format_rules(selected_columns, pipeline, path):
        if not path:
            return html.Div(
                "Aucun dataset chargé.",
                className="text-muted"
            )

        selected_columns = selected_columns or []

        df, is_spark = load_df(path)

        if df is None:
            return html.Div(
                "Impossible de charger le dataset.",
                className="text-danger"
            )

        existing_rules = get_format_rules_from_pipeline(pipeline)
        rows = []

        for column in selected_columns:
            try:
                # Le type est récupéré pour chaque colonne
                current_dtype = get_column_dtype(
                    df,
                    column,
                    is_spark=is_spark
                )

                # Stratégie déjà enregistrée dans le pipeline, si elle existe
                selected_action = existing_rules.get(column)

                rows.append(
                    build_format_rule_row(
                        column=column,
                        current_dtype=current_dtype,
                        selected_action=selected_action,
                        is_spark=is_spark,
                    )
                )

            except Exception as exc:
                rows.append(
                    dbc.Alert(
                        f"Erreur pour '{column}' : {exc}",
                        color="danger"
                    )
                )

        if not rows:
            return html.Div(
                "Sélectionnez une ou plusieurs colonnes.",
            )

        return rows
    
    @app.callback(
        Output("pipeline-store", "data"),
        Output("fmt-feedback", "children"),
        Input("fmt-apply", "n_clicks"),
        Input("fmt-reset", "n_clicks"),
        State("original-parquet-path-store", "data"),
        State({
            "type": "fmt-strategy",
            "column": ALL
        }, "value"),
        State({
            "type": "fmt-strategy",
            "column": ALL
        }, "id"),
        State("pipeline-store", "data"),
        prevent_initial_call=True
    )
    def on_apply_formats_click(apply_clicks, reset_clicks, path, strategies, strategy_ids, pipeline):
        from app.modules.common.pipeline import add_step, reset_step
        print("STRATEGIES :", strategies)
        print("STRATEGY IDS :", strategy_ids)
        
        pipeline = pipeline or []

        if not ctx.triggered_id:
            raise PreventUpdate

        if ctx.triggered_id == "fmt-reset":
            if not reset_clicks or reset_clicks < 1:
                raise PreventUpdate

            updated_pipeline = reset_step(
                pipeline,
                "formats"
            )

            return updated_pipeline, dbc.Alert(
                "✅ Les stratégies de format ont été réinitialisées.",
                color="info"
            )

        if ctx.triggered_id != "fmt-apply":
            raise PreventUpdate
        
        if not path:
            return pipeline, format_warning("Aucun dataset chargé.")

        rules = [
            {
                "column": component_id["column"],
                "action": strategy
            }
            for strategy, component_id in zip(strategies, strategy_ids)
            if strategy
        ]

        if not rules:
            return pipeline, format_warning(
                "Aucune stratégie de format sélectionnée."
            )
        print("=== RULES ENVOYEES ===")
        print(rules)
        
        try:
            df, is_spark = load_df(path)

            if df is None:
                return pipeline, format_warning("Aucun dataset chargé.")

            if not rules:
                return df

            # Détection automatique du moteur
            if is_spark is None:
                is_spark = hasattr(df, "withColumn") and hasattr(df, "schema")

            # Étape de validation : on vérifie que les transformations sont exécutables
            if is_spark is False:
                apply_formats_pandas(df, rules, is_spark)

            elif is_spark is True:
                apply_formats_spark(df, rules, is_spark)

            new_step = {
                "step": "formats",
                "params": rules
            }

            updated_pipeline = add_step(
                pipeline or [],
                new_step
            )

            return updated_pipeline, dbc.Alert(
                f"✅ {len(rules)} stratégie(s) de format enregistrée(s).",
                color="success"
            )

        except Exception as e:
            return pipeline, format_warning(
                f"Erreur lors de la configuration des formats : {str(e)}"
            )
        