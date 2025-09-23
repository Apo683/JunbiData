# modules/common/ui.py
STYLE_DROPDOWN = {
    "width": "250px",
    "backgroundColor": "#ffffff",
    "color": "#000",
    "borderRadius": "5px",
    "marginBottom": "15px"
}
OPTIONS_DROPDOWN = [
    {"label": "Graphique (brut)", "value": "graph_raw"},
    {"label": "Graphique (tri croissant)", "value": "graph_ascending"},
    {"label": "Graphique (tri décroissant)", "value": "graph_descending"},
    {"label": "Tableau", "value": "table"}
]
CHECKLIST_STYLE = {"display": "flex", "flexWrap": "wrap", "justifyContent": "flex-start"}
CHECKLIST_INPUT_STYLE = {"marginRight": "6px"}
CHECKLIST_LABEL_STYLE = {
    "width": "230px", "textOverflow": "ellipsis", "overflow": "hidden",
    "whiteSpace": "nowrap", "display": "inline-block", "marginRight": "12px",
}