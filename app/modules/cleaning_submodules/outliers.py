# app/modules/nettoyage_submodules/outliers.py
import dash_bootstrap_components as dbc
from dash import html, dcc, Input, Output, State, MATCH, ALL, ctx, no_update
from dash.exceptions import PreventUpdate
import numpy as np
import pandas as pd
import plotly.express as px

from app.modules.common.io import load_df, format_warning, get_column_dtype, sample_for_plot
from app.modules.chargement import show_dataset_preview
from app.modules.common.ui import STYLE_DROPDOWN, CHECKLIST_STYLE, CHECKLIST_INPUT_STYLE, CHECKLIST_LABEL_STYLE, get_dynamic_checklist_label_style

try:
    from pyspark.sql import functions as F
    HAS_SPARK = True
except Exception:
    HAS_SPARK = False

OUTLIER_METHODS = [
    {"label": "IQR", "value": "iqr"},
    {"label": "Z-score", "value": "z"},
    {"label": "MAD", "value": "mad"},
]

OUTLIER_ACTIONS = [
    {"label": "Supprimer les lignes", "value": "drop"},
    {"label": "Ajouter un indicateur", "value": "flag"},
    {"label": "Remplacer par la médiane", "value": "median"},
    {"label": "Remplacer par la moyenne", "value": "mean"},
    {"label": "Remplacer par une constante", "value": "constant"},
    # {"label": "Interpoler", "value": "interpolate"},
    {"label": "Borner les valeurs", "value": "cap"},
    {"label": "Utiliser des bornes personnalisées", "value": "custom_cap"},
]

TAB_ID = "clean-outliers"

# =========================================================
# Fonctions logique métier
# =========================================================
def _numeric_cols(df, is_spark):
    if is_spark:
        from pyspark.sql.types import (
            ByteType,
            ShortType,
            IntegerType,
            LongType,
            FloatType,
            DoubleType,
            DecimalType,
        )

        numeric_types = (
            ByteType,
            ShortType,
            IntegerType,
            LongType,
            FloatType,
            DoubleType,
            DecimalType,
        )

        return [
            field.name
            for field in df.schema.fields
            if isinstance(field.dataType, numeric_types)
        ]

    return [
        column
        for column in df.columns
        if pd.api.types.is_numeric_dtype(df[column])
        and not pd.api.types.is_bool_dtype(df[column])
    ]

def _get_outlier_bounds_pandas(series, method, iqr_k=1.5, z_threshold=3.0, mad_threshold=3.5):
    series = pd.to_numeric(series, errors="coerce")

    if method == "iqr":
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1

        if pd.isna(iqr) or iqr <= 0:
            return None, None

        return q1 - iqr_k * iqr, q3 + iqr_k * iqr

    if method == "z":
        mean = series.mean()
        std = series.std(ddof=0)

        if pd.isna(std) or std <= 0:
            return None, None

        return (
            mean - z_threshold * std,
            mean + z_threshold * std,
        )

    if method == "mad":
        median = series.median()
        mad = np.median(np.abs(series.dropna() - median))
        scale = 1.4826 * mad

        if pd.isna(scale) or scale <= 0:
            return None, None

        return (
            median - mad_threshold * scale,
            median + mad_threshold * scale,
        )

    raise ValueError(f"Méthode de détection inconnue : '{method}'.")

def apply_outliers_pandas(df, rules):
    pdf = df.copy()

    for rule in rules:
        column = rule.get("column")
        method = rule.get("method") or rule.get("outliers_method")
        action = rule.get("action")

        if not column or not method or not action:
            raise ValueError(
                f"Règle outlier invalide : {rule}"
            )

        if column not in pdf.columns:
            raise ValueError(
                f"La colonne '{column}' n'existe pas dans le dataset."
            )

        if not pd.api.types.is_numeric_dtype(pdf[column]):
            raise ValueError(
                f"La colonne '{column}' doit être numérique."
            )

        iqr_k = float(rule.get("iqr_k") or 1.5)
        z_threshold = float(rule.get("z_threshold") or 3.0)
        mad_threshold = float(rule.get("mad_threshold") or 3.5)

        series = pd.to_numeric(pdf[column], errors="coerce")

        low, high = _get_outlier_bounds_pandas(
            series=series,
            method=method,
            iqr_k=iqr_k,
            z_threshold=z_threshold,
            mad_threshold=mad_threshold,
        )

        # Aucun outlier détectable
        if low is None or high is None:
            if action == "flag":
                pdf[f"is_outlier_{column}"] = 0
            continue

        outlier_mask = (
            series.notna()
            & ((series < low) | (series > high))
        )

        if action == "drop":
            pdf = pdf.loc[~outlier_mask].copy()

        elif action == "flag":
            pdf[f"is_outlier_{column}"] = outlier_mask.astype(int)

        elif action == "median":
            replacement = series.median()
            pdf.loc[outlier_mask, column] = replacement

        elif action == "mean":
            replacement = series.mean()
            pdf.loc[outlier_mask, column] = replacement

        elif action == "constant":
            if rule.get("constant") is None:
                raise ValueError(
                    f"Une constante est requise pour '{column}'."
                )

            constant = pd.to_numeric(
                rule["constant"],
                errors="coerce",
            )

            if pd.isna(constant):
                raise ValueError(
                    f"La constante de '{column}' doit être numérique."
                )

            pdf.loc[outlier_mask, column] = constant

        elif action == "cap":
            pdf.loc[series < low, column] = low
            pdf.loc[series > high, column] = high

        elif action == "custom_cap":
            custom_low = rule.get("lower_bound")
            custom_high = rule.get("upper_bound")

            if custom_low is None or custom_high is None:
                raise ValueError(
                    f"Les deux bornes sont requises pour '{column}'."
                )

            custom_low = float(custom_low)
            custom_high = float(custom_high)

            if custom_low >= custom_high:
                raise ValueError(
                    f"La borne minimale doit être inférieure "
                    f"à la borne maximale pour '{column}'."
                )

            # Les bornes personnalisées remplacent les bornes calculées
            pdf.loc[series < custom_low, column] = custom_low
            pdf.loc[series > custom_high, column] = custom_high

        else:
            raise ValueError(
                f"Action outlier inconnue : '{action}'."
            )

    return pdf

def _get_outlier_bounds_spark(sdf, column, method, iqr_k=1.5, z_threshold=3.0, mad_threshold=3.5):
    if method == "iqr":
        quantiles = sdf.approxQuantile(
            column,
            [0.25, 0.75],
            1e-3,
        )

        if len(quantiles) != 2:
            return None, None

        q1, q3 = quantiles
        iqr = q3 - q1

        if iqr <= 0:
            return None, None

        return (
            q1 - iqr_k * iqr,
            q3 + iqr_k * iqr,
        )

    if method == "z":
        stats = sdf.select(
            F.mean(F.col(column)).alias("mean"),
            F.stddev_pop(F.col(column)).alias("std"),
        ).first()

        mean = stats["mean"]
        std = stats["std"]

        if mean is None or std is None or std <= 0:
            return None, None

        return (
            mean - z_threshold * std,
            mean + z_threshold * std,
        )

    if method == "mad":
        median = sdf.approxQuantile(
            column,
            [0.5],
            1e-3,
        )[0]

        if median is None:
            return None, None

        mad = sdf.select(
            F.percentile_approx(
                F.abs(F.col(column) - F.lit(median)),
                0.5,
                10000,
            ).alias("mad")
        ).first()["mad"]

        if mad is None:
            return None, None

        scale = 1.4826 * mad

        if scale <= 0:
            return None, None

        return (
            median - mad_threshold * scale,
            median + mad_threshold * scale,
        )

    raise ValueError(f"Méthode de détection inconnue : '{method}'.")

def apply_outliers_spark(df, rules):
    sdf = df

    for rule in rules:
        column = rule.get("column")
        method = rule.get("method") or rule.get("outliers_method")
        action = rule.get("action")

        if not column or not method or not action:
            raise ValueError(
                f"Règle outlier invalide : {rule}"
            )

        if column not in sdf.columns:
            raise ValueError(
                f"La colonne '{column}' n'existe pas dans le dataset."
            )

        iqr_k = float(rule.get("iqr_k") or 1.5)
        z_threshold = float(rule.get("z_threshold") or 3.0)
        mad_threshold = float(rule.get("mad_threshold") or 3.5)

        low, high = _get_outlier_bounds_spark(
            sdf=sdf,
            column=column,
            method=method,
            iqr_k=iqr_k,
            z_threshold=z_threshold,
            mad_threshold=mad_threshold,
        )

        if low is None or high is None:
            if action == "flag":
                sdf = sdf.withColumn(
                    f"is_outlier_{column}",
                    F.lit(0),
                )
            continue

        value = F.col(column)

        outlier_condition = (
            value.isNotNull()
            & ((value < F.lit(low)) | (value > F.lit(high)))
        )

        if action == "drop":
            sdf = sdf.filter(~outlier_condition)

        elif action == "flag":
            sdf = sdf.withColumn(
                f"is_outlier_{column}",
                F.when(outlier_condition, 1).otherwise(0),
            )

        elif action == "median":
            median = sdf.approxQuantile(
                column,
                [0.5],
                1e-3,
            )[0]

            sdf = sdf.withColumn(
                column,
                F.when(outlier_condition, F.lit(median))
                .otherwise(value),
            )

        elif action == "mean":
            mean = sdf.select(
                F.mean(value).alias("mean")
            ).first()["mean"]

            sdf = sdf.withColumn(
                column,
                F.when(outlier_condition, F.lit(mean))
                .otherwise(value),
            )

        elif action == "constant":
            constant = rule.get("constant")

            if constant is None:
                raise ValueError(
                    f"Une constante est requise pour '{column}'."
                )

            try:
                constant = float(constant)
            except (TypeError, ValueError):
                raise ValueError(
                    f"La constante de '{column}' doit être numérique."
                )

            sdf = sdf.withColumn(
                column,
                F.when(outlier_condition, F.lit(constant))
                .otherwise(value),
            )

        elif action == "cap":
            sdf = sdf.withColumn(
                column,
                F.when(value < F.lit(low), F.lit(low))
                .when(value > F.lit(high), F.lit(high))
                .otherwise(value),
            )

        elif action == "custom_cap":
            custom_low = rule.get("lower_bound")
            custom_high = rule.get("upper_bound")

            if custom_low is None or custom_high is None:
                raise ValueError(
                    f"Les deux bornes sont requises pour '{column}'."
                )

            custom_low = float(custom_low)
            custom_high = float(custom_high)

            if custom_low >= custom_high:
                raise ValueError(
                    f"La borne minimale doit être inférieure "
                    f"à la borne maximale pour '{column}'."
                )

            sdf = sdf.withColumn(
                column,
                F.when(value < F.lit(custom_low), F.lit(custom_low))
                .when(value > F.lit(custom_high), F.lit(custom_high))
                .otherwise(value),
            )

        else:
            raise ValueError(
                f"Action outlier inconnue : '{action}'."
            )

    return sdf

def apply_outliers(df, rules, is_spark=False):
    if is_spark:
        return apply_outliers_spark(df, rules)

    return apply_outliers_pandas(df, rules)

def get_outlier_rules_from_pipeline(pipeline):
    if not pipeline:
        return {}
    print(f"--------- Pipeline Outliers : {pipeline} ---------")
    rules_by_column = {}

    for step in pipeline:
        if step.get("step") != "outliers":
            continue

        for rule in step.get("params", []):
            column = rule.get("column")
            action = rule.get("action")

            if not column or not action:
                continue

            if column and action:
                # La dernière règle gagne
                rules_by_column[column] = rule

    return rules_by_column

def build_outlier_graphs(df, column, method, iqr_k=1.5, z_th=3.0, mad_th=3.5, sample_n=5000, log_scale=False, title_prefix=""):
    """
    Construit le boxplot et l'histogramme d'une colonne.
    Retourne (boxplot, histogramme).
    """
    is_spark = hasattr(df, "schema")

    pdf = sample_for_plot(
        df,
        column,
        is_spark,
        sample_n=sample_n,
    )

    s = pd.to_numeric(pdf[column], errors="coerce").dropna()

    if s.empty:
        return None, None

    # Calcul des seuils sur l'échantillon affiché
    if method == "iqr":
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        low, high = (
            (q1 - iqr_k * iqr, q3 + iqr_k * iqr)
            if iqr > 0 else (None, None)
        )
        suffix = f"IQR, k={iqr_k}"

    elif method == "z":
        mean = s.mean()
        std = s.std(ddof=0)
        low, high = (
            (mean - z_th * std, mean + z_th * std)
            if std > 0 else (None, None)
        )
        suffix = f"Z-score, seuil={z_th}"

    else:
        median = s.median()
        mad = np.median(np.abs(s - median))
        scale = 1.4826 * mad
        low, high = (
            (median - mad_th * scale, median + mad_th * scale)
            if scale > 0 else (None, None)
        )
        suffix = f"MAD, seuil={mad_th}"

    fig_box = px.box(
        pdf,
        y=column,
        points="outliers",
        title=f"{title_prefix}{column} — Boxplot ({suffix})",
        height=300,
    )

    fig_hist = px.histogram(
        pdf,
        x=column,
        nbins=50,
        title=f"{title_prefix}{column} — Histogramme ({suffix})",
        height=300,
    )

    if low is not None and high is not None:
        fig_hist.add_vline(x=low, line_color="tomato")
        fig_hist.add_vline(x=high, line_color="tomato")

    if log_scale:
        fig_box.update_yaxes(type="log")
        fig_hist.update_xaxes(type="log")

    return fig_box, fig_hist

def build_before_after_outlier_layout(before_box, before_hist, after_box, after_hist):
    return html.Div(
        [
            html.H5("Avant traitement"),
            dbc.Row(
                [
                    dbc.Col(
                        dcc.Graph(figure=before_box),
                        xs=12,
                        lg=6,
                    ),
                    dbc.Col(
                        dcc.Graph(figure=before_hist),
                        xs=12,
                        lg=6,
                    ),
                ],
                className="g-2",
            ),

            html.H5("Après traitement", className="mt-3"),
            dbc.Row(
                [
                    dbc.Col(
                        dcc.Graph(figure=after_box),
                        xs=12,
                        lg=6,
                    ),
                    dbc.Col(
                        dcc.Graph(figure=after_hist),
                        xs=12,
                        lg=6,
                    ),
                ],
                className="g-2",
            ),
        ]
    )

def build_outlier_rule_row(column, current_dtype, selected_rule=None):
    selected_rule  = selected_rule or {}
    badge_style={
        "fontSize": "15px",
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
                        f"Type actuel : {current_dtype}",
                        color="secondary",
                        className="me-1",
                        style=badge_style,
                    ),
                ],
                className="d-flex"
            ),
        ],
        className="d-flex align-items-left flex-wrap"
    )

    return dbc.Row([
        dbc.Col(column_info, xs="auto", md=3, className="d-flex align-items-center"),

        dbc.Col(
            [
                html.Label("Méthode de détection :"),
                dcc.Dropdown(
                    id={
                        "type": "out-method",
                        "column": column,
                    },
                    options=OUTLIER_METHODS,
                    value=selected_rule.get("method", "iqr"),
                    clearable=False,
                    placeholder="Choisir une méthode",
                    style={**STYLE_DROPDOWN, "width": "150px", "marginBottom": "0px"},
                ),
            ],
            xs="auto",
        ),
        dbc.Col(
            [
                html.Label("Coefficient IQR :"),
                dbc.Input(
                    id={"type": "out-iqr-k", "column": column},
                    type="number",
                    value=selected_rule.get("iqr_k", 1.5),
                    min=0,
                    step=0.1,
                ),
            ],
            xs="auto",
            id={"type": "out-iqr-container", "column": column},
        ),
        dbc.Col(
            [
                html.Label("Seuil Z-score :"),
                dbc.Input(
                    id={"type": "out-z", "column": column},
                    type="number",
                    value=selected_rule.get("z_threshold", 3.0),
                    min=0,
                    step=0.1,
                ),
            ],
            xs="auto",
            id={"type": "out-z-container", "column": column},
        ),
        dbc.Col(
            [
                html.Label("Seuil MAD :"),
                dbc.Input(
                    id={"type": "out-mad", "column": column},
                    type="number",
                    value=selected_rule.get("mad_threshold", 3.5),
                    min=0,
                    step=0.1,
                ),
            ],
            xs="auto",
            id={"type": "out-mad-container", "column": column},
        ),
        dbc.Col([
            html.Label("Méthode de traitement :"),

            dcc.Dropdown(
                id={
                    "type": "out-action",
                    "column": column,
                },
                options=OUTLIER_ACTIONS,
                value=selected_rule.get("action", "drop"),
                clearable=False,
                style={**STYLE_DROPDOWN, "width": "280px", "marginBottom": "0px"}
            ),
        ], xs="auto"),
        dbc.Col(
            [
                dbc.Input(
                    id={"type": "out-constant-value", "column": column},
                    type="number",
                    placeholder="Valeur constante",
                ),
            ],
            xs="auto",
        ),
        dbc.Col(
            [
                dbc.Input(
                    id={"type": "out-lower-bound", "column": column},
                    type="number",
                    placeholder="Borne inférieure",
                ),
            ],
            xs="auto",
        ),
        dbc.Col(
            [
                dbc.Input(
                    id={"type": "out-upper-bound", "column": column},
                    type="number",
                    placeholder="Borne supérieure",
                ),
            ],
            xs="auto",
        ),       
        dbc.Col(
                dbc.Button(
                    "Visualiser",
                    id={"type": "out-visualize", "column": column},
                    color="primary",
                    outline=True,
                    disabled=True,
                ),
                width="auto",
                xs="auto", md="auto", className="px-1 d-flex align-items-center",
        ),
    ], className="mb-2 g-2 align-items-end")


# =========================================================
# Fonctions DASH
# =========================================================

def get_tab():
    return dbc.Tab(tab_id=TAB_ID, label="Valeurs aberrantes")

def get_layout():
    return html.Div([
        html.P(
            "Nettoyage des outliers (choix de la méthode de détection, choix de la méthode de traitement, visualiser l'impact puis appliquer globalement) :",
            style={"fontSize": "18px", "marginBottom": "10px"},
        ),
        dbc.Col([
            dbc.Button("🔄 Réinitialiser", id="out-reset", style={"display": "none", "marginTop": "6px", "marginBottom": "6px"}),
        ]),
        dbc.Row([
            dbc.Col([
                html.P(
                    "Sélectionnez les colonnes à traiter :",
                    style={"fontSize": "17px", "marginBottom": "10px", "marginTop": "10px"}
                ),

                dcc.Checklist(
                    id="out-columns",
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
            id="out-rules-container",
            className="mt-3"
        ),

        dbc.Button(
            "Appliquer",
            id="out-apply",
            color="primary",
            className="mt-3"
        ),

        html.Div(
            id="out-feedback",
            className="mt-2"
        ),

        dbc.Modal(
            [
                dbc.ModalHeader(dbc.ModalTitle(id="out-visual-title")),
                dbc.ModalBody(id="out-visual-body"),
                dbc.ModalFooter(
                    dbc.Button("Fermer", id="out-visual-close",color="secondary",)
                ),
            ],
            id="out-visual-modal",
            is_open=False,
            size="xl",
            scrollable=True,
            backdrop="static",
        )
    ])



def register_callbacks(app):
    @app.callback(
        Output("out-reset", "style"),
        Input("pipeline-store", "data"),
        Input("cleaning-subtabs", "active_tab")
    )
    def update_outliers_reset_button(pipeline, active_tab):
        has_outliers = any(
            step.get("step") == "outliers"
            for step in (pipeline or [])
        )

        return {
            "display": "inline-block"
            if active_tab == TAB_ID and has_outliers
            else "none",
            "marginTop": "6px",
            "marginBottom": "6px",
        }

    @app.callback(
        Output("out-columns", "options"),
        Output("out-columns", "value"),
        Output("out-columns", "labelStyle"),
        Input("original-parquet-path-store", "data"),
        Input("pipeline-store", "data"),
    )
    def populate_outliers_checklist(path, pipeline):
        if not path:
            return [], [], CHECKLIST_LABEL_STYLE

        df, is_spark = load_df(path)

        if df is None:
            return [], [], CHECKLIST_LABEL_STYLE

        columns = _numeric_cols(df, is_spark)

        options = [
            {
                "label": column,
                "value": column,
            }
            for column in columns
        ]

        existing_rules = get_outlier_rules_from_pipeline(pipeline)

        selected_columns = [
            column
            for column in columns
            if column in existing_rules
        ]

        label_style = get_dynamic_checklist_label_style(options)

        return options, selected_columns, label_style

    @app.callback(
        Output("out-rules-container", "children"),
        Input("out-columns", "value"),
        Input("pipeline-store", "data"),
        State("original-parquet-path-store", "data"),
    )
    def render_outlier_rules(selected_columns, pipeline, path):
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

        existing_rules = get_outlier_rules_from_pipeline(pipeline)
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
                selected_rule = existing_rules.get(column)

                rows.append(
                    build_outlier_rule_row(
                        column=column,
                        current_dtype=current_dtype,
                        selected_rule=selected_rule,
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
        Output({"type": "out-visualize", "column": MATCH}, "disabled",),
        Input({"type": "out-action", "column": MATCH},"value",),
    )
    def toggle_visualize_button(strategy):
        return not bool(strategy)

    @app.callback(
        Output({"type": "out-iqr-container", "column": MATCH}, "style"),
        Output({"type": "out-z-container", "column": MATCH}, "style"),
        Output({"type": "out-mad-container", "column": MATCH}, "style"),
        Input({"type": "out-method", "column": MATCH}, "value"),
    )
    def toggle_outlier_thresholds(method):
        return (
            {"display": "block"} if method == "iqr" else {"display": "none"},
            {"display": "block"} if method == "z" else {"display": "none"},
            {"display": "block"} if method == "mad" else {"display": "none"},
        )

    @app.callback(
        Output({"type": "out-constant-value", "column": MATCH}, "style"),
        Output({"type": "out-lower-bound", "column": MATCH}, "style"),
        Output({"type": "out-upper-bound", "column": MATCH}, "style"),
        Input({"type": "out-action", "column": MATCH}, "value"),
    )
    def toggle_outlier_action_inputs(action):
        return (
            {"display": "block"} if action == "constant" else {"display": "none"},
            {"display": "block"} if action == "custom_cap" else {"display": "none"},
            {"display": "block"} if action == "custom_cap" else {"display": "none"},
        )

    @app.callback(
        Output("out-visual-modal", "is_open"),
        Output("out-visual-body", "children"),
        Output("out-visual-title", "children"),
        Input({"type": "out-visualize", "column": ALL}, "n_clicks"),
        Input("out-visual-close", "n_clicks"),
        State({"type": "out-visualize", "column": ALL}, "id"),
        State({"type": "out-method", "column": ALL}, "id"),
        State({"type": "out-method", "column": ALL}, "value"),
        State({"type": "out-action", "column": ALL}, "id"),
        State({"type": "out-action", "column": ALL}, "value"),
        State({"type": "out-iqr-k", "column": ALL}, "id"),
        State({"type": "out-iqr-k", "column": ALL}, "value"),
        State({"type": "out-z", "column": ALL}, "id"),
        State({"type": "out-z", "column": ALL}, "value"),
        State({"type": "out-mad", "column": ALL}, "id"),
        State({"type": "out-mad", "column": ALL}, "value"),
        State({"type": "out-constant-value", "column": ALL}, "id"),
        State({"type": "out-constant-value", "column": ALL}, "value"),
        State({"type": "out-lower-bound", "column": ALL}, "id"),
        State({"type": "out-lower-bound", "column": ALL}, "value"),
        State({"type": "out-upper-bound", "column": ALL}, "id"),
        State({"type": "out-upper-bound", "column": ALL}, "value"),
        State("original-parquet-path-store", "data"),
        prevent_initial_call=True,
    )
    def open_outliers_preview(visualize_clicks, close_click, visualize_ids, method_ids, method_values, action_ids, action_values, iqr_ids, iqr_values, z_ids, z_values, mad_ids, mad_values, constant_ids, constant_values, lower_ids, lower_values, upper_ids,upper_values, path):
        triggered_id = ctx.triggered_id

        # Fermeture du modal
        if triggered_id == "out-visual-close":
            return False, no_update, no_update

        if not isinstance(triggered_id, dict):
            raise PreventUpdate

        if triggered_id.get("type") != "out-visualize":
            raise PreventUpdate

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

        column = triggered_id.get("column")

        def get_value(ids, values):
            for component_id, value in zip(ids or [], values or []):
                if (
                    isinstance(component_id, dict)
                    and component_id.get("column") == column
                ):
                    return value
            return None

        method = get_value(method_ids, method_values)
        action = get_value(action_ids, action_values)
        iqr_k = get_value(iqr_ids, iqr_values)
        z_threshold = get_value(z_ids, z_values)
        mad_threshold = get_value(mad_ids, mad_values)
        constant = get_value(constant_ids, constant_values)
        lower_bound = get_value(lower_ids, lower_values)
        upper_bound = get_value(upper_ids, upper_values)

        title = f"Prévisualisation — {column}"

        if not method:
            return (
                True,
                dbc.Alert(
                    "Veuillez sélectionner une méthode de détection.",
                    color="warning",
                ),
                title,
            )

        if not action:
            return (
                True,
                dbc.Alert(
                    "Veuillez sélectionner une stratégie de traitement.",
                    color="warning",
                ),
                title,
            )

        if action == "constant" and constant in (None, ""):
            return (
                True,
                dbc.Alert(
                    "Veuillez renseigner la constante de remplacement.",
                    color="warning",
                ),
                title,
            )

        if action == "custom_cap":
            if lower_bound in (None, "") or upper_bound in (None, ""):
                return (
                    True,
                    dbc.Alert(
                        "Veuillez renseigner les deux bornes personnalisées.",
                        color="warning",
                    ),
                    title,
                )

        if not path:
            return (
                True,
                format_warning("Aucun dataset original n'est disponible."),
                "Prévisualisation",
            )

        try:
            df, is_spark = load_df(path)

            if df is None:
                raise ValueError("Impossible de charger le dataset original.")

            rule = {
                "column": column,
                "method": method,
                "action": action,
                "iqr_k": float(iqr_k or 1.5),
                "z_threshold": float(z_threshold or 3.0),
                "mad_threshold": float(mad_threshold or 3.5),
                "constant": constant,
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
            }

            # Conversion du dataset original pour l'affichage
            if is_spark:
                df_before = df.select(column).toPandas()

                df_after_spark = apply_outliers(
                    df,
                    [rule],
                    is_spark=True,
                )

                df_after = df_after_spark.select(column).toPandas()

            else:
                df_before = df[[column]].copy()

                df_after = apply_outliers(
                    df.copy(),
                    [rule],
                    is_spark=False,
                )[[column]].copy()

            # Génération des graphiques
            before_box, before_hist = build_outlier_graphs(
                df_before,
                column=column,
                method=method,
                iqr_k=rule["iqr_k"],
                z_th=rule["z_threshold"],
                mad_th=rule["mad_threshold"],
                title_prefix="Avant — ",
            )

            after_box, after_hist = build_outlier_graphs(
                df_after,
                column=column,
                method=method,
                iqr_k=rule["iqr_k"],
                z_th=rule["z_threshold"],
                mad_th=rule["mad_threshold"],
                title_prefix="Après — ",
            )

            body = html.Div(
                [
                    dbc.Alert(
                        [
                            html.Strong("Stratégie simulée : "),
                            f"{action} — méthode : {method}",
                            html.Br(),
                            f"Nombre de lignes : "
                            f"{len(df_before)} → {len(df_after)}",
                        ],
                        color="info",
                    ),

                    html.H5("Avant traitement", className="mt-2"),

                    dbc.Row(
                        [
                            dbc.Col(
                                dcc.Graph(
                                    figure=before_box,
                                    config={"displayModeBar": False},
                                ),
                                xs=12,
                                lg=6,
                            ),
                            dbc.Col(
                                dcc.Graph(
                                    figure=before_hist,
                                    config={"displayModeBar": False},
                                ),
                                xs=12,
                                lg=6,
                            ),
                        ],
                        className="g-3",
                    ),

                    html.H5(
                        "Après traitement",
                        className="mt-4",
                    ),

                    dbc.Row(
                        [
                            dbc.Col(
                                dcc.Graph(
                                    figure=after_box,
                                    config={"displayModeBar": False},
                                ),
                                xs=12,
                                lg=6,
                            ),
                            dbc.Col(
                                dcc.Graph(
                                    figure=after_hist,
                                    config={"displayModeBar": False},
                                ),
                                xs=12,
                                lg=6,
                            ),
                        ],
                        className="g-3",
                    ),
                ]
            )

            return True, body, f"Impact du traitement — {column}"

        except Exception as exc:
            return (
                True,
                dbc.Alert(
                    f"Impossible de simuler la stratégie : {exc}",
                    color="danger",
                ),
                f"Erreur — {column}",
            )

    @app.callback(
        Output("pipeline-store", "data", allow_duplicate=True),
        Output("out-feedback", "children"),
        Input("out-apply", "n_clicks"),
        Input("out-reset", "n_clicks"),
        State("original-parquet-path-store", "data"),
        State("out-columns", "value"),
        State({"type": "out-method", "column": ALL}, "id"),
        State({"type": "out-method", "column": ALL}, "value"),
        State({"type": "out-action", "column": ALL}, "id"),
        State({"type": "out-action", "column": ALL}, "value"),
        State({"type": "out-iqr-k", "column": ALL}, "id"),
        State({"type": "out-iqr-k", "column": ALL}, "value"),
        State({"type": "out-z", "column": ALL}, "id"),
        State({"type": "out-z", "column": ALL}, "value"),
        State({"type": "out-mad", "column": ALL}, "id"),
        State({"type": "out-mad", "column": ALL}, "value"),
        State({"type": "out-constant-value", "column": ALL}, "id"),
        State({"type": "out-constant-value", "column": ALL}, "value"),
        State({"type": "out-lower-bound", "column": ALL}, "id"),
        State({"type": "out-lower-bound", "column": ALL}, "value"),
        State({"type": "out-upper-bound", "column": ALL}, "id"),
        State({"type": "out-upper-bound", "column": ALL}, "value"),
        State("pipeline-store", "data"),
        prevent_initial_call=True,
    )
    def on_apply_outliers_click(apply_clicks, reset_clicks, path, columns, method_ids, methods, action_ids, actions, iqr_ids, iqr_values, z_ids,z_values, mad_ids, mad_values, constant_ids, constant_values, lower_ids, lower_values, upper_ids, upper_values, pipeline):
        from app.modules.common.pipeline import add_step, reset_step

        pipeline = pipeline or []

        if not ctx.triggered_id:
            raise PreventUpdate

        # Réinitialisation
        if ctx.triggered_id == "out-reset":
            if not reset_clicks:
                raise PreventUpdate

            updated_pipeline = reset_step(
                pipeline,
                "outliers",
            )

            return (
                updated_pipeline,
                dbc.Alert(
                    "✅ Les stratégies de valeurs aberrantes ont été réinitialisées.",
                    color="info",
                ),
            )


        # Application dans le pipeline
        if ctx.triggered_id != "out-apply":
            raise PreventUpdate

        if not apply_clicks:
            raise PreventUpdate

        if not path:
            return (
                pipeline,
                format_warning("Aucun dataset chargé."),
            )

        def values_by_column(ids, values):
            result = {}

            for component_id, value in zip(ids or [], values or []):
                if (
                    isinstance(component_id, dict)
                    and component_id.get("column")
                ):
                    result[component_id["column"]] = value

            return result

        methods_by_column = values_by_column(method_ids, methods)
        actions_by_column = values_by_column(action_ids, actions)
        iqr_by_column = values_by_column(iqr_ids, iqr_values)
        z_by_column = values_by_column(z_ids, z_values)
        mad_by_column = values_by_column(mad_ids, mad_values)
        constants_by_column = values_by_column(
            constant_ids,
            constant_values,
        )
        lower_by_column = values_by_column(
            lower_ids,
            lower_values,
        )
        upper_by_column = values_by_column(
            upper_ids,
            upper_values,
        )

        rules = []

        columns = columns or []

        for column in columns:
            method = methods_by_column.get(column)
            action = actions_by_column.get(column)

            # Une ligne incomplète n'est pas enregistrée
            if not method and not action:
                continue

            if not method:
                return (
                    pipeline,
                    dbc.Alert(
                        f"Veuillez sélectionner une méthode pour « {column} ».",
                        color="warning",
                    ),
                )

            if not action:
                return (
                    pipeline,
                    dbc.Alert(
                        f"Veuillez sélectionner une stratégie pour « {column} ».",
                        color="warning",
                    ),
                )

            rule = {
                "column": column,
                "method": method,
                "action": action,
                "iqr_k": float(iqr_by_column.get(column) or 1.5),
                "z_threshold": float(
                    z_by_column.get(column) or 3.0
                ),
                "mad_threshold": float(
                    mad_by_column.get(column) or 3.5
                ),
                "constant": constants_by_column.get(column),
                "lower_bound": lower_by_column.get(column),
                "upper_bound": upper_by_column.get(column),
            }

            if action == "constant":
                if rule["constant"] in (None, ""):
                    return (
                        pipeline,
                        dbc.Alert(
                            f"Veuillez renseigner la constante pour « {column} ».",
                            color="warning",
                        ),
                    )

            if action == "custom_cap":
                if (
                    rule["lower_bound"] in (None, "")
                    or rule["upper_bound"] in (None, "")
                ):
                    return (
                        pipeline,
                        dbc.Alert(
                            f"Veuillez renseigner les deux bornes "
                            f"personnalisées pour « {column} ».",
                            color="warning",
                        ),
                    )

                try:
                    rule["lower_bound"] = float(rule["lower_bound"])
                    rule["upper_bound"] = float(rule["upper_bound"])
                except (TypeError, ValueError):
                    return (
                        pipeline,
                        dbc.Alert(
                            f"Les bornes de « {column} » doivent être numériques.",
                            color="warning",
                        ),
                    )

                if rule["lower_bound"] >= rule["upper_bound"]:
                    return (
                        pipeline,
                        dbc.Alert(
                            f"La borne minimale doit être inférieure "
                            f"à la borne maximale pour « {column} ».",
                            color="warning",
                        ),
                    )

            rules.append(rule)

        if not rules:
            return (
                pipeline,
                format_warning(
                    "Veuillez configurer au moins une règle de valeurs aberrantes."
                ),
            )

        try:
            df, is_spark = load_df(path)

            if df is None:
                return (
                    pipeline,
                    format_warning("Impossible de charger le dataset."),
                )

            # Validation de la configuration sans modifier le dataset original
            apply_outliers(
                df.copy() if not is_spark else df,
                rules,
                is_spark=is_spark,
            )

            new_step = {
                "step": "outliers",
                "params": rules,
            }

            updated_pipeline = add_step(
                pipeline,
                new_step,
            )

            return (
                updated_pipeline,
                dbc.Alert(
                    f"✅ {len(rules)} règle(s) de valeurs aberrantes enregistrée(s).",
                    color="success",
                ),
            )

        except Exception as exc:
            return (
                pipeline,
                format_warning(
                    f"Erreur lors de la configuration des outliers : {exc}"
                ),
            )