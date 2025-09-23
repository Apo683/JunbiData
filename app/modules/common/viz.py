# modules/common/viz.py
import dash
from dash import html, dcc, dash_table
import plotly.express as px
import plotly.io as pio

from app.modules.common.io import _apply_smart_hover

SAFE_TEMPLATE = "junbi" if "junbi" in pio.templates else "plotly_white"

def _get_common_layout(title, xaxis_title, yaxis_title, height=400):
    return {
        "title": {"text": title, "x": 0.5, "xanchor": "center"},
        "height": height,
        "margin": {"l": 50, "r": 20, "t": 60, "b": 30},
        "xaxis_title": xaxis_title,
        "yaxis_title": yaxis_title,
    }

def _generate_content(df, display_mode, x_col, y_col, title, xaxis_title, yaxis_title,
    sort_key=None, height=400, custom_layout=None, bargap=0.25,
    category_order=None, x_as_category=False,):
    # Cas table
    if display_mode == "table":
        return dash_table.DataTable(
            data=(df if df is not None else pd.DataFrame()).to_dict("records"),
            columns=[{"name": c, "id": c} for c in (df.columns if df is not None else [])],
            style_table={"overflowX": "auto"},
            style_cell={"textAlign": "left", "minWidth": "120px"},
        )

    if df is None or df.empty:
        return html.Div("Aucune donnée à afficher.")
    data = df.copy()
    if display_mode == "graph_ascending" and sort_key:
        data = data.sort_values(sort_key, ascending=True, kind="mergesort")
    elif display_mode == "graph_descending" and sort_key:
        data = data.sort_values(sort_key, ascending=False, kind="mergesort")
    if x_as_category:
        data[x_col] = data[x_col].astype(str)
    # Création robuste de la figure avec template sûr + fallback
    try:
        fig = px.bar(
            data,
            x=x_col,
            y=y_col,
            title=title,
            height=height,
            color=y_col,
            color_continuous_scale="Bluered_r",
            template=SAFE_TEMPLATE,  # <- toujours passer un nom de template
        )
    except Exception as e:
        # Fallback au cas improbable où le template serait indisponible à cet instant
        fig = px.bar(
            data,
            x=x_col,
            y=y_col,
            title=title,
            height=height,
            color=y_col,
            color_continuous_scale="Bluered_r",
            template="plotly_white",
        )
    if custom_layout:
        fig.update_layout(**custom_layout)
    fig.update_layout(bargap=bargap)
    if category_order is not None:
        fig.update_xaxes(type="category", categoryorder="array", categoryarray=[str(v) for v in category_order])
    elif x_as_category:
        fig.update_xaxes(type="category")

    _apply_smart_hover(fig, x_col, y_col)

    return dcc.Graph(figure=fig)