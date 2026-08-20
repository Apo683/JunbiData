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


def get_dynamic_checklist_label_style(options, min_width=25, max_width=55, extra_chars=3,):
    """
    Calcule une largeur adaptée aux labels d'une checklist.
    La largeur est exprimée en ch.
    """
    labels = [
        str(option.get("label", ""))
        for option in (options or [])
    ]

    max_length = max(
        (len(label) for label in labels),
        default=min_width
    )

    dynamic_width = min(
        max(max_length + extra_chars, min_width),
        max_width
    )

    return {
        **CHECKLIST_LABEL_STYLE,
        "width": f"{dynamic_width}ch",
    }
