# app/modules/cleaning_submodules/missing.py
from __future__ import annotations
from typing import Any, Dict, List, Tuple, Optional

import os
import shutil
from pathlib import Path

import pandas as pd
import dash
from dash import html, dcc, Input, Output, State, ALL, no_update
from dash.exceptions import PreventUpdate

# Compat ctx: Dash >=2.9 -> dash.ctx ; sinon fallback sur callback_context
try:
    from dash import ctx
except Exception:
    from dash import callback_context as ctx  # type: ignore

import dash_bootstrap_components as dbc
import plotly.express as px

from app.modules.common.io import load_df_only, format_warning
from app.modules.common.ui import (
    STYLE_DROPDOWN,
    CHECKLIST_STYLE,
    CHECKLIST_INPUT_STYLE,
    CHECKLIST_LABEL_STYLE,
)

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
        html.P(
            "Remplacement des valeurs manquantes, colonne par colonne :",
            style={"fontSize": "18px", "marginBottom": "10px"},
        ),
        dbc.Row([
            dbc.Col([
                # Stores locaux à l'onglet
                dcc.Store(id="cleaned-parquet-path-store", data=None, storage_type="session"),
                dcc.Store(id="miss-ui-store", data=None, storage_type="memory"),
                dcc.Store(id="miss-na-counts-store", data={}),
                # Store utilisé pour bloquer les réouvertures fantômes du modal
                dcc.Store(id="miss-viz-ts-store", data=0, storage_type="memory"),

                html.P("Sélectionnez les colonnes à traiter :", style={"fontSize": "17px", "marginBottom": "8px"}),
                dcc.Checklist(
                    id="miss-columns",
                    options=[],
                    value=[],
                    style=CHECKLIST_STYLE,
                    inputStyle=CHECKLIST_INPUT_STYLE,
                    labelStyle=CHECKLIST_LABEL_STYLE,
                ),
                html.Hr(),
                html.Div(id="miss-rows-container"),
                html.Div(id="miss-feedback", className="mt-2"),

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
    # dtype: "numeric", "datetime", "text", "other"
    # On met "drop_row" en premier pour en faire la valeur par défaut
    base_always = [
        {"label": "Supprimer la ligne (NaN)", "value": "drop_row"},
        {"label": "Constante", "value": "constant"},
    ]
    if dtype == "numeric":
        return [
            {"label": "Supprimer la ligne (NaN)", "value": "drop_row"},
            {"label": "Moyenne", "value": "mean"},
            {"label": "Médiane", "value": "median"},
            {"label": "Zéro", "value": "zero"},
            {"label": "Interpolation (linéaire)", "value": "interpolate"},
            {"label": "Propagation (vers le bas)", "value": "ffill"},
            {"label": "Propagation (vers le haut)", "value": "bfill"},
            {"label": "Constante", "value": "constant"},
        ]
    if dtype == "datetime":
        return [
            {"label": "Supprimer la ligne (NaN)", "value": "drop_row"},
            {"label": "Propagation (vers le bas)", "value": "ffill"},
            {"label": "Propagation (vers le haut)", "value": "bfill"},
            {"label": "Constante", "value": "constant"},
        ]
    if dtype in ("text",):
        return [
            {"label": "Supprimer la ligne (NaN)", "value": "drop_row"},
            {"label": "Mode", "value": "mode"},
            {"label": "Chaîne vide", "value": "empty"},
            {"label": "Propagation (vers le bas)", "value": "ffill"},
            {"label": "Propagation (vers le haut)", "value": "bfill"},
            {"label": "Constante", "value": "constant"},
        ]
    return base_always


def _infer_dtype_pandas(s: pd.Series) -> str:
    if pd.api.types.is_numeric_dtype(s):
        return "numeric"
    if pd.api.types.is_datetime64_any_dtype(s):
        return "datetime"
    if pd.api.types.is_categorical_dtype(s) or s.dtype == object:
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
    is_numeric = (dtype_label == "numeric")
    return html.Div([
        dbc.Row(
            [
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
                    className="me-1",
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
                    className="me-1",
                ),
                dbc.Col(
                    dbc.Input(
                        id={"type": "miss-const", "col": colname},
                        placeholder="Valeur constante…",
                        type="number" if is_numeric else "text",
                        step="any" if is_numeric else None,
                        style={"width": "200px", "marginBottom": "0px"},
                    ),
                    id={"type": "miss-const-wrap", "col": colname},
                    style={"display": "none"},
                    width="auto",
                ),
                dbc.Col(
                    dbc.Button(
                        "Visualiser",
                        id={"type": "miss-visualize", "col": colname},
                        color="primary",
                        outline=True,
                    ),
                    width="auto",
                    className="me-1",
                ),
                dbc.Col(
                    dbc.Button(
                        "Appliquer",
                        id={"type": "miss-apply", "col": colname},
                        color="primary",
                    ),
                    width="auto",
                ),
            ],
            className="align-items-center gy-1 gx-2",
            justify="start",
        ),
        html.Hr(className="my-2"),
    ], id={"type": "miss-row", "col": colname})

def _to_number_or_raise(x) -> float | None:
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    xs = str(x).strip()
    if xs == "":
        return None
    # tolère virgule décimale
    xs = xs.replace(",", ".")
    return float(xs)  # laisse lever ValueError si non convertible


def _apply_pandas(df: pd.DataFrame, cols: List[str], strategies: Dict[str, Tuple[str, Any]]):
    out = df.copy()
    report: Dict[str, str] = {}
    new_na: Dict[str, int] = {}

    for c in cols:
        before = int(out[c].isna().sum()) if c in out.columns else 0
        try:
            if c not in out.columns:
                report[c] = "Colonne absente."
                new_na[c] = 0
                continue

            dtype_label = _infer_dtype_pandas(out[c])
            strat, const = strategies.get(c, ("drop_row", None))

            if strat == "drop_row":
                out = out[~out[c].isna()]
            elif strat == "constant":
                out[c] = out[c].fillna(const)
            elif strat == "mean" and pd.api.types.is_numeric_dtype(out[c]):
                out[c] = out[c].fillna(out[c].mean())
            elif strat == "median" and pd.api.types.is_numeric_dtype(out[c]):
                out[c] = out[c].fillna(out[c].median())
            elif strat == "zero" and pd.api.types.is_numeric_dtype(out[c]):
                out[c] = out[c].fillna(0)
            elif strat == "interpolate" and pd.api.types.is_numeric_dtype(out[c]):
                out[c] = out[c].interpolate(method="linear", limit_direction="both")
            elif strat in ("ffill", "bfill"):
                out[c] = getattr(out[c], strat)()
            elif strat == "mode":
                mode_val = out[c].mode(dropna=True)
                mode_val = mode_val.iloc[0] if not mode_val.empty else const
                out[c] = out[c].fillna(mode_val)
            elif strat == "empty":
                out[c] = out[c].fillna("")
            else:
                report[c] = f"Stratégie '{strat}' inconnue, aucune modification."
                after = int(out[c].isna().sum())
                new_na[c] = after
                continue

            after = int(out[c].isna().sum())
            report[c] = f"NA avant: {before}, après: {after}, stratégie: {strat}"
            new_na[c] = after

        except Exception as e:
            report[c] = f"Erreur pandas: {e}"
            new_na[c] = int(out[c].isna().sum())

    return out, report, new_na


def _apply_spark(df: SparkDF, cols: List[str], strategies: Dict[str, Tuple[str, Any]]):
    if not HAS_SPARK:
        raise RuntimeError("Spark non disponible")
    out = df
    report: Dict[str, str] = {}
    new_na: Dict[str, int] = {}

    for c in cols:
        try:
            before = out.filter(F.col(c).isNull()).count()
            strat, const = strategies.get(c, ("drop_row", None))

            if strat == "drop_row":
                out = out.filter(F.col(c).isNotNull())
            elif strat == "constant":
                out = out.fillna({c: const})
            elif strat in ("ffill", "bfill"):
                # Implémentations complètes ffill/bfill en Spark sont non-triviales.
                report[c] = f"Stratégie '{strat}' non supportée en Spark dans cette vue, aucune modification."
            elif strat in ("mean", "median", "zero", "interpolate", "mode", "empty"):
                # Simplification: seules 'drop_row' et 'constant' sont supportées ici
                report[c] = f"Stratégie '{strat}' non supportée en Spark dans cette vue, aucune modification."
            else:
                report[c] = f"Stratégie '{strat}' inconnue, aucune modification."

            after = out.filter(F.col(c).isNull()).count()
            if c not in report:
                report[c] = f"NA avant: {before}, après: {after}, stratégie: {strat}"
            new_na[c] = after
        except Exception as e:
            report[c] = f"Erreur Spark: {e}"
            new_na[c] = out.filter(F.col(c).isNull()).count()

    return out, report, new_na


def _preview_fig_pandas(df: pd.DataFrame, col: str):
    if col not in df.columns:
        return px.histogram(pd.Series([], name=col), title=f"{col} (absente)")
    s = df[col]
    na = int(s.isna().sum())
    if pd.api.types.is_numeric_dtype(s):
        nbins = 50 if s.size < 5000 else 60
        fig = px.histogram(s, nbins=nbins, title=f"Distribution de {col} (NA: {na})", height=300)
        return fig
    else:
        vc = s.astype(str).value_counts(dropna=False).nlargest(50)
        return px.bar(x=vc.index, y=vc.values,
                      title=f"Top modalités de {col} (NA: {na})",
                      labels={"x": col, "y": "count"}, height=300)


def _preview(parquet_info: Optional[str], col: str, strategy: Optional[str], const: Any):
    """
    Construit le titre + body pour le modal d'aperçu.
    On lit le dataset 'original' (chemin passé via original-parquet-path-store)
    et on applique virtuellement la stratégie uniquement pour la colonne ciblée.
    """
    strategy = strategy or "drop_row"  # valeur par défaut robuste
    df_before = load_df_only(parquet_info)
    if df_before is None:
        title = "Aperçu indisponible"
        body = dbc.Alert("Impossible de charger le dataset original.", color="warning")
        return title, body

    if isinstance(df_before, pd.DataFrame):
        df_after, _, _ = _apply_pandas(df_before, [col], {col: (strategy, const)})
        fig_before = _preview_fig_pandas(df_before, col)
        fig_after = _preview_fig_pandas(df_after, col)
    else:
        # Spark → collect sur la seule colonne pour tracer avec Plotly (pandas)
        sdf_after, _, _ = _apply_spark(df_before, [col], {col: (strategy, const)})
        pdf_before = df_before.select(col).toPandas()
        pdf_after = sdf_after.select(col).toPandas()
        fig_before = _preview_fig_pandas(pdf_before, col)
        fig_after = _preview_fig_pandas(pdf_after, col)

    title = f"Impact sur la distribution: {col}"
    body = html.Div([
        dbc.Row([
            dbc.Col(html.Div([html.H6("Avant"), dcc.Graph(figure=fig_before, style={"height": "38vh"})]), md=6),
            dbc.Col(html.Div([html.H6("Après (prévisualisation)"), dcc.Graph(figure=fig_after, style={"height": "38vh"})]), md=6),
        ], className="g-3")
    ])

    return title, body


def register_callbacks(app):

    # 0) Alimente la checklist et stocke les NA par colonne à partir du chemin actif
    @app.callback(
        Output("miss-columns", "options"),
        Output("miss-columns", "value"),
        Output("miss-na-counts-store", "data"),
        Input("parquet-path-store", "data"),
        prevent_initial_call=False,
    )
    def _populate_checklist(parquet_path: Optional[str]):
        df = load_df_only(parquet_path)
        if df is None:
            return [], [], {}
        if isinstance(df, pd.DataFrame):
            miss_counts = df.isna().sum().astype(int).to_dict()
            cols = list(df.columns)
        else:
            cols = df.columns
            miss_counts = {c: df.filter(F.col(c).isNull()).count()} if HAS_SPARK else {}
        options = [{"label": f"{c} (NA: {miss_counts.get(c, 0)})", "value": c} for c in cols]
        return options, [], miss_counts

    # 1) Génère/supprime les lignes en fonction de la sélection
    @app.callback(
        Output("miss-rows-container", "children"),
        Input("miss-columns", "value"),
        State("parquet-path-store", "data"),
        State("miss-na-counts-store", "data"),
        prevent_initial_call=False,
    )
    def _render_rows(selected_cols: List[str], parquet_path: Optional[str], na_counts: Dict[str, int]):
        if not selected_cols:
            return []
        df = load_df_only(parquet_path)
        if df is None:
            return [format_warning("Aucune donnée chargée.")]

        rows = []
        for c in selected_cols:
            if isinstance(df, pd.DataFrame):
                if c not in df.columns:
                    continue
                dtype_label = _infer_dtype_pandas(df[c])
            else:
                dtype_label = _infer_dtype_spark(df, c) if HAS_SPARK else "text"

            rows.append(_build_row(c, dtype_label, na_counts.get(c, 0)))
        return rows

    # 1-bis) Reset du timestamp mémorisé quand la sélection change (évite réouvertures)
    @app.callback(
        Output("miss-viz-ts-store", "data", allow_duplicate=True),
        Input("miss-columns", "value"),
        prevent_initial_call=True,
    )
    def _reset_ts_on_selection(_):
        # Réinitialise le détecteur de clics; évite les réouvertures fantômes
        return 0

    # 2) Affiche/cache le champ "constante" selon la stratégie sélectionnée
    @app.callback(
        Output({"type": "miss-const-wrap", "col": ALL}, "style"),
        Input({"type": "miss-strategy", "col": ALL}, "value"),
        prevent_initial_call=True
    )
    def _toggle_const_all(strategies):
        if strategies is None:
            raise PreventUpdate
        if isinstance(strategies, list) and len(strategies) == 0:
            return []
        if isinstance(strategies, list):
            return [{"display": "block"} if v == "constant" else {"display": "none"} for v in strategies]
        try:
            n = len(strategies)
        except Exception:
            raise PreventUpdate
        return [no_update] * n

    # 3) Visualisation Avant/Après (sans sauvegarder)
    def _vals_by_col(ids, vals):
        out = {}
        ids = ids or []
        vals = vals or []
        for i, _id in enumerate(ids):
            if isinstance(_id, dict) and "col" in _id:
                v = vals[i] if i < len(vals) else None
                out[_id["col"]] = v
        return out

    @app.callback(
        Output("miss-visual-modal", "is_open"),
        Output("miss-visual-body", "children"),
        Output("miss-visual-title", "children"),
        Output("miss-viz-ts-store", "data"),
        Input({"type": "miss-visualize", "col": ALL}, "n_clicks"),
        Input("miss-visual-close", "n_clicks"),
        State({"type": "miss-visualize", "col": ALL}, "id"),
        State({"type": "miss-strategy", "col": ALL}, "id"),
        State({"type": "miss-strategy", "col": ALL}, "value"),
        State({"type": "miss-const", "col": ALL}, "id"),
        State({"type": "miss-const", "col": ALL}, "value"),
        State("miss-viz-ts-store", "data"),
        State("parquet-path-store", "data"),
        prevent_initial_call=True,
    )
    def open_preview(clicks, close_click, btn_ids, strat_ids, strat_vals, const_ids, const_vals, last_count, active_parquet):
        # Sécurité sur None
        clicks = clicks or []
        btn_ids = btn_ids or []
        strat_ids, strat_vals = strat_ids or [], strat_vals or []
        const_ids, const_vals = const_ids or [], const_vals or []
        last_count = last_count or 0
        close_click = close_click or 0

        # Qui a déclenché ?
        trig = ctx.triggered_id

        # 1) Bouton Fermer → fermer, ne pas toucher au compteur
        if trig == "miss-visual-close":
            return False, no_update, no_update, last_count

        # 2) Clic sur un bouton Visualiser
        if isinstance(trig, dict) and trig.get("type") == "miss-visualize":
            # Compteur total de clics; sert d’edge detection globale
            total_clicks = sum((c or 0) for c in clicks)
            if total_clicks <= last_count:
                # Pas un nouveau clic
                return no_update, no_update, no_update, last_count

            col = trig.get("col")

            # Construire des maps col -> valeur
            def map_by_col(ids, vals):
                out = {}
                for i, _id in enumerate(ids):
                    if isinstance(_id, dict) and "col" in _id:
                        out[_id["col"]] = vals[i] if i < len(vals) else None
                return out

            strat_map = map_by_col(strat_ids, strat_vals)
            const_map = map_by_col(const_ids, const_vals)

            strategy = strat_map.get(col, "drop_row")
            const_value = const_map.get(col)

            # Charger et produire le contenu (isoler les exceptions)
            try:
                # Remplace load_df_only si tu utilises une autre fonction
                df = load_df_only(active_parquet)  # ← vérifie que active_parquet vient bien de parquet-path-store
                title, body = _preview(parquet_info=active_parquet, col=col, strategy=strategy, const=const_value)
            except Exception as e:
                title = f"Prévisualisation — {col}"
                body = dbc.Alert(f"Erreur dans _preview(): {e}", color="danger")

            return True, body, title, total_clicks

        # 3) Tout autre trigger (ne devrait pas arriver ici)
        return no_update, no_update, no_update, last_count

    # 4) Appliquer une stratégie sur la colonne cliquée
    @app.callback(
        Output("miss-feedback", "children"),
        Output("miss-ui-store", "data"),
        Output("parquet-path-store", "data", allow_duplicate=True),
        Output("cleaned-parquet-path-store", "data"),
        Input({"type": "miss-apply", "col": ALL}, "n_clicks"),
        State({"type": "miss-apply", "col": ALL}, "id"),
        State({"type": "miss-strategy", "col": ALL}, "id"),
        State({"type": "miss-strategy", "col": ALL}, "value"),
        State({"type": "miss-const", "col": ALL}, "id"),
        State({"type": "miss-const", "col": ALL}, "value"),
        State("miss-columns", "value"),  # sélection courante
        State("parquet-path-store", "data"),
        prevent_initial_call=True,
    )
    def _apply_missing_row(n_clicks_list, btn_ids, strat_ids, strat_vals, const_ids, const_vals,
                           checklist_value, active_path):
        # Pas de clic → rien
        if not n_clicks_list or not any(n_clicks_list):
            return no_update, no_update, no_update, no_update
        if not active_path:
            return dbc.Alert("Aucun parquet actif.", color="warning"), no_update, no_update, no_update

        # Quelle colonne a été cliquée ?
        trig = getattr(dash, "ctx", ctx).triggered_id
        if not isinstance(trig, dict) or trig.get("type") != "miss-apply":
            return no_update, no_update, no_update, no_update
        col_clicked = trig.get("col")

        # Mappings stratégie / constante (pour toutes les colonnes visibles)
        s_map = {i["col"]: v for i, v in zip(strat_ids or [], strat_vals or []) if isinstance(i, dict) and "col" in i}
        c_map = {i["col"]: v for i, v in zip(const_ids or [], const_vals or []) if isinstance(i, dict) and "col" in i}
        strategy = s_map.get(col_clicked, "drop_row")
        const_value = c_map.get(col_clicked, None)

        # État UI à persister
        ui_state = {
            "selected": checklist_value or [],
            "strategies": s_map,
            "constants": c_map,
        }

        # Charger le DF actif
        df = load_df_only(active_path)
        if df is None:
            return dbc.Alert("Impossible de charger le dataset actif.", color="warning"), ui_state, no_update, no_update

        # Appliquer
        try:
            if isinstance(df, pd.DataFrame):
                df_new, report, new_na = _apply_pandas(df, [col_clicked], {col_clicked: (strategy, const_value)})
                # Ici: si tu as une fonction de sauvegarde, appelle-la et mets à jour les stores de chemin.
                # Par sécurité (pas de code de persistence fourni), on se contente de feedback.
                msg = report.get(col_clicked, "Appliqué.")
                return dbc.Alert(f"{col_clicked}: {msg}", color="success"), ui_state, no_update, no_update
            else:
                if not HAS_SPARK:
                    return dbc.Alert("Spark non disponible sur cet environnement.", color="warning"), ui_state, no_update, no_update
                df_new, report, new_na = _apply_spark(df, [col_clicked], {col_clicked: (strategy, const_value)})
                msg = report.get(col_clicked, "Appliqué (Spark).")
                return dbc.Alert(f"{col_clicked}: {msg}", color="success"), ui_state, no_update, no_update
        except Exception as e:
            return dbc.Alert(f"Erreur d'application sur {col_clicked}: {e}", color="danger"), ui_state, no_update, no_update
