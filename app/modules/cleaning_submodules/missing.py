# app/modules/cleaning_submodules/missing.py
from __future__ import annotations
from typing import Any, Dict, List, Tuple
import pandas as pd
import dash
from dash import html, dcc, Input, Output, State, MATCH, ALL, no_update, callback_context as ctx 
import dash_bootstrap_components as dbc
import os
import shutil
from pathlib import Path

from app.modules.common.io import load_df_only, format_warning
from app.modules.common.ui import STYLE_DROPDOWN, CHECKLIST_STYLE, CHECKLIST_INPUT_STYLE, CHECKLIST_LABEL_STYLE

# Spark optionnel
try:
    from pyspark.sql import DataFrame as SparkDF
    from pyspark.sql import functions as F
    from pyspark.sql.functions import col as spark_col
    HAS_SPARK = True
except Exception:
    SparkDF = Any  # type: ignore
    F = None       # type: ignore
    spark_col = None  # type: ignore
    HAS_SPARK = False


TAB_ID = "clean-missing"

def get_tab() -> dbc.Tab:
    return dbc.Tab(label="Valeurs manquantes", tab_id=TAB_ID)

def get_layout() -> html.Div:
    return html.Div([
        html.P("Remplacement des valeurs manquantes, colonne par colonne :", style={"fontSize": "18px", "marginBottom": "10px"}),
        dbc.Row([
            dbc.Col([
                dcc.Store(id="cleaned-parquet-path-store", data=None, storage_type="session"),
                dcc.Store(id="miss-na-counts-store", data={}),
                html.P("Sélectionnez les colonnes à traiter :", style={"fontSize": "17px", "marginBottom": "8px"}),
                dcc.Checklist(
                    id="miss-columns",
                    options=[],   # rempli par callback
                    value=[],
                    style=CHECKLIST_STYLE,
                    inputStyle=CHECKLIST_INPUT_STYLE,  # selon ton thème
                    labelStyle=CHECKLIST_LABEL_STYLE,  # <-- style lisibilité
                ),
                html.Hr(),
                html.Div(id="miss-rows-container"),
                html.Div(id="miss-feedback", className="mt-2"),
                # dbc.Button("Appliquer", id="miss-apply", color="primary", className="mt-2"),
            ], width=12),
        ]),
    ])

def cleaned_from(active_path: str) -> str:
    p = Path(active_path)
    parent = p.parent if str(p.parent) not in ("", ".") else Path("data")
    name = p.name
    if name.startswith("cleaned_"):
        return str(p)
    return str(parent / f"cleaned_{name}")

def remove_parquet_path(path: str):
    if not path:
        return
    if os.path.isdir(path):
        shutil.rmtree(path, ignore_errors=True)
    elif os.path.exists(path):
        try:
            os.remove(path)
        except FileNotFoundError:
            pass

def _strategy_options_for_dtype(dtype: str) -> List[Dict[str, str]]:
    # dtype: "numeric", "categorical", "datetime", "text", "other"
    base = [{"label": "Constante", "value": "constant"}]
    if dtype == "numeric":
        return [
            {"label": "Moyenne", "value": "mean"},
            {"label": "Médiane", "value": "median"},
            {"label": "Zéro", "value": "zero"},
            {"label": "Interpolation (linéaire)", "value": "interpolate"},
            {"label": "Propagation (vers le bas)", "value": "ffill"},
            {"label": "Propagation (vers le haut)", "value": "bfill"},
            *base,
        ]
    if dtype == "datetime":
        return [
            {"label": "Propagation (vers le bas)", "value": "ffill"},
            {"label": "Propagation (vers le haut)", "value": "bfill"},
            *base,
        ]
    if dtype in ("categorical", "text"):
        return [
            {"label": "Mode", "value": "mode"},
            {"label": "Chaîne vide", "value": "empty"},
            {"label": "Propagation (vers le bas)", "value": "ffill"},
            {"label": "Propagation (vers le haut)", "value": "bfill"},
            *base,
        ]
    return [{"label": "Constante", "value": "constant"}]

def _infer_dtype_pandas(s: pd.Series) -> str:
    if pd.api.types.is_numeric_dtype(s):
        return "numeric"
    if pd.api.types.is_datetime64_any_dtype(s):
        return "datetime"
    if pd.api.types.is_categorical_dtype(s) or s.dtype == object:
        # on distinguera "text" vs "categorical" plus tard si besoin
        return "text"
    return "other"

def _infer_dtype_spark(df: SparkDF, colname: str) -> str:
    spark_type = dict(df.dtypes).get(colname, "")
    st = spark_type.lower()
    if any(t in st for t in ["int", "double", "float", "long", "short", "decimal"]):
        return "numeric"
    if "timestamp" in st or "date" in st:
        return "datetime"
    return "text"

def _build_row(colname: str, dtype_label: str, na_count: int) -> html.Div:
    return html.Div(
        id={"type": "miss-row", "col": colname},
        className="mb-2",
        children=[
            dbc.Row([
                dbc.Col(
                    html.Span(
                        f"{colname} — {na_count} NA",
                        style={
                            "backgroundColor": "#ffffff",
                            "color": "#000000",
                            "height": "36px",
                            "lineHeight": "36px",
                            "padding": "0 8px",
                            "borderRadius": "4px",
                            "fontFamily": "monospace",
                            "display": "inline-block",
                            "boxSizing": "border-box",
                        }
                    ),
                    width="auto",
                    className="me-3",
                ),
                dbc.Col(
                    dcc.Dropdown(
                        id={"type": "miss-strategy", "col": colname},
                        options=_strategy_options_for_dtype(dtype_label),
                        value="constant",
                        placeholder="Choisir une stratégie…",
                        style={**STYLE_DROPDOWN, "width": "220px", "marginBottom": "0px"},
                        clearable=False,
                    ),
                    width="auto",
                    className="me-2 align-self-center",
                ),
                dbc.Col(
                    dbc.Input(
                        id={"type": "miss-const", "col": colname},
                        placeholder="Valeur constante…",
                        # disabled=False,  # activé uniquement si strategy == constant
                    ),
                    id = {"type": "miss-const-wrap", "col": colname},
                    style={"display": "none"},
                ),
                dbc.Col(dbc.Button(
                    "Appliquer",
                    id={"type": "miss-apply", "col": colname},
                    color="primary",
                    size="sm"
                ), width=2),
            ], className="g-1 justify-content-start", align="center"),
        ]
    )

def register_callbacks(app):

    # 0) Remplir la checklist selon le parquet_path
    @app.callback(
        Output("miss-columns", "options"),
        Output("miss-columns", "value"),
        Output("miss-na-counts-store", "data"),
        Input("parquet-path-store", "data"),
        prevent_initial_call=False,
    )
    def _populate_checklist(parquet_path: str | None):
        df = load_df_only(parquet_path)
        if df is None:
            return [], [], {}
        if isinstance(df, pd.DataFrame):
            miss_counts = df.isna().sum().astype(int).to_dict()
            cols = list(df.columns)
        else:
            # Spark: attention coût potentiellement élevé
            cols = df.columns
            miss_counts = {c: df.filter(F.col(c).isNull()).count() for c in cols} if HAS_SPARK else {}
        options = [{"label": f"{c} (NA: {miss_counts.get(c, 0)})", "value": c} for c in cols]
        return options, [], miss_counts
    
    # 1) Générer/Supprimer les lignes par colonne sélectionnée
    @app.callback(
        Output("miss-rows-container", "children"),
        Input("miss-columns", "value"),
        State("parquet-path-store", "data"),
        State("miss-na-counts-store", "data"),
        prevent_initial_call=False,
    )
    def _render_rows(selected_cols: List[str], parquet_path: str | None, na_counts: Dict[str, int]):
        if not selected_cols:
            return []
        df = load_df_only(parquet_path)
        if df is None:
            return [format_warning("Aucune donnée chargée.")]
        rows = []
        for c in selected_cols:
            if isinstance(df, pd.DataFrame):
                dtype_label = _infer_dtype_pandas(df[c])
            else:
                dtype_label = _infer_dtype_spark(df, c) if HAS_SPARK else "text"
            rows.append(_build_row(c, dtype_label, na_counts.get(c, 0)))
        return rows

    # 2) Activer/désactiver le champ constante selon la stratégie
    @app.callback(
        Output({"type": "miss-const-wrap", "col": ALL}, "style"),
        Input({"type": "miss-strategy", "col": ALL}, "value"),
        prevent_initial_call=False
    )
    def _toggle_const_all(strategies):
        if not strategies:
            return no_update
        styles = []
        for v in strategies:
            styles.append({"display": "block"} if v == "constant" else {"display": "none"})
        return styles

    # 3) Appliquer les stratégies sélectionnées
    @app.callback(
        Output("miss-feedback", "children"),
        Output("parquet-path-store", "data", allow_duplicate=True),   # ← important
        Output("cleaned-parquet-path-store", "data"),
        Input({"type": "miss-apply", "col": ALL}, "n_clicks"),
        State({"type": "miss-apply", "col": ALL}, "id"),
        State({"type": "miss-strategy", "col": ALL}, "id"),
        State({"type": "miss-strategy", "col": ALL}, "value"),
        State({"type": "miss-const", "col": ALL}, "id"),
        State({"type": "miss-const", "col": ALL}, "value"),
        State("parquet-path-store", "data"),
        prevent_initial_call=True,
    )
    def _apply_missing_row(n_clicks_list, btn_ids, strat_ids, strat_vals, const_ids, const_vals, active_path):
        # Rien à faire si pas de clic
        if not n_clicks_list or not any(n_clicks_list):
            return no_update, no_update, no_update
        if not active_path:
            return dbc.Alert("Aucun parquet actif.", color="warning"), no_update, no_update

        # Colonne qui a déclenché le callback
        trig = ctx.triggered_id  # dict de la forme {"type":"miss-apply","col":"NomCol"}
        if not trig or "col" not in trig:
            raise dash.exceptions.PreventUpdate
        col_clicked = trig["col"]

        # Construire des maps col -> valeur pour stratégie et constante
        strat_map = {sid["col"]: sval for sid, sval in zip(strat_ids or [], strat_vals or [])}
        const_map = {cid["col"]: cval for cid, cval in zip(const_ids or [], const_vals or [])}

        strategy = strat_map.get(col_clicked)
        const_value = const_map.get(col_clicked)

        if not strategy:
            return dbc.Alert(f"Aucune stratégie sélectionnée pour {col_clicked}.", color="warning"), no_update, no_update

        # Charger le df actif
        df = load_df_only(active_path)
        if df is None:
            return dbc.Alert("Impossible de charger les données.", color="danger"), no_update, no_update

        # Appliquer la stratégie à la seule colonne cliquée
        try:
            if isinstance(df, pd.DataFrame):
                df_new, report = _apply_pandas(df, [col_clicked], {col_clicked: (strategy, const_value)})
            else:
                if not HAS_SPARK:
                    return dbc.Alert("Spark non disponible sur cet environnement.", color="warning"), no_update, no_update
                df_new, report = _apply_spark(df, [col_clicked], {col_clicked: (strategy, const_value)})
        except Exception as e:
            return dbc.Alert(f"Erreur d'application sur {col_clicked}: {e}", color="danger"), no_update, no_update

        # Sauvegarder vers cleaned_*
        out_path = cleaned_from(active_path)
        try:
            Path(os.path.dirname(out_path) or ".").mkdir(parents=True, exist_ok=True)
            if isinstance(df_new, pd.DataFrame):
                df_new.to_parquet(out_path, index=False)
            else:
                df_new.write.mode("overwrite").parquet(out_path)
        except Exception as e:
            return dbc.Alert(f"Imputation OK mais échec de la sauvegarde vers {out_path}: {e}", color="danger"), no_update, no_update

        # Feedback + bascule automatique du chemin actif sur le cleaned
        msg = report.get(col_clicked, f"{col_clicked}: transformation appliquée.")
        alert = dbc.Alert(f"{msg} → chemin actif: {out_path}", color="success")
        return alert, out_path, out_path

def _apply_pandas(df: pd.DataFrame, cols: List[str], strategies: Dict[str, Tuple[str, Any]]) -> Tuple[pd.DataFrame, Dict[str, str]]:
    out = df.copy()
    report: Dict[str, str] = {}
    for c in cols:
        strat, const = strategies.get(c, ("constant", None))
        before = out[c].isna().sum()
        dtype_label = _infer_dtype_pandas(out[c])

        if strat == "constant":
            out[c] = out[c].fillna(const)
        elif strat == "mean" and dtype_label == "numeric":
            out[c] = out[c].fillna(out[c].mean())
        elif strat == "median" and dtype_label == "numeric":
            out[c] = out[c].fillna(out[c].median())
        elif strat == "zero" and dtype_label == "numeric":
            out[c] = out[c].fillna(0)
        elif strat == "mode":
            mode_val = out[c].mode(dropna=True)
            fill_val = mode_val.iloc[0] if not mode_val.empty else const
            out[c] = out[c].fillna(fill_val)
        elif strat == "empty":
            out[c] = out[c].fillna("")
        elif strat == "ffill":
            out[c] = out[c].ffill()
        elif strat == "bfill":
            out[c] = out[c].bfill()
        elif strat == "interpolate" and dtype_label == "numeric":
            out[c] = out[c].interpolate(method="linear", limit_direction="both")
        else:
            # fallback constant
            out[c] = out[c].fillna(const)

        after = out[c].isna().sum()
        report[c] = f"NA avant: {before}, après: {after}, stratégie: {strat}"
    return out, report


def _apply_spark(sdf: SparkDF, cols: List[str], strategies: Dict[str, Tuple[str, Any]]) -> Tuple[SparkDF, Dict[str, str]]:
    # Implémentation Spark “best effort”
    out = sdf
    report: Dict[str, str] = {}
    for c in cols:
        strat, const = strategies.get(c, ("constant", None))
        dtype_label = _infer_dtype_spark(out, c)
        before = out.filter(F.col(c).isNull()).count()

        if strat == "constant":
            out = out.fillna({c: const})
        elif strat == "zero" and dtype_label == "numeric":
            out = out.fillna({c: 0})
        elif strat == "mean" and dtype_label == "numeric":
            mean_val = out.select(F.mean(F.col(c))).collect()[0][0]
            out = out.fillna({c: mean_val})
        elif strat == "median" and dtype_label == "numeric":
            # médiane via approxQuantile (rapide)
            median_val = out.approxQuantile(c, [0.5], 0.01)[0]
            out = out.fillna({c: median_val})
        elif strat == "mode":
            # mode simple: valeur la plus fréquente (peut être coûteux)
            mode_row = out.groupBy(F.col(c)).count().orderBy(F.desc("count")).limit(1).collect()
            mode_val = mode_row[0][0] if mode_row else const
            out = out.fillna({c: mode_val})
        elif strat in ("ffill", "bfill"):
            # ffill/bfill nécessitent une notion d’ordre; ici on ne l’a pas.
            # Option: laisser tel quel ou documenter qu’il faut une clé d’ordonnancement.
            # On fallback sur constante.
            out = out.fillna({c: const})
        elif strat == "empty":
            out = out.fillna({c: ""})
        elif strat == "interpolate" and dtype_label == "numeric":
            # Pas natif en Spark; fallback constant
            out = out.fillna({c: const})
        else:
            out = out.fillna({c: const})

        after = out.filter(F.col(c).isNull()).count()
        report[c] = f"NA avant: {before}, après: {after}, stratégie: {strat}"
    return out, report
