# modules/common/pipeline.py

def get_pipeline(config):
    """
    Retourne toujours la liste des étapes du pipeline.
    Compatible avec l'ancien format et le nouveau.
    """
    if not config:
        return []

    # Nouveau format :
    # {"pipeline": [...], "metadata": {...}}
    if isinstance(config, dict):
        return config.get("pipeline", [])

    # Ancien format : [...]
    if isinstance(config, list):
        return config

    return []

def run_pipeline(config_df, config):
    pipeline = get_pipeline(config)

    for step in pipeline:
        step_name = step.get("step")
        params = step.get("params", [])

        if step_name == "formats":
            from app.modules.cleaning_submodules.formats import apply_formats
            config_df = apply_formats(config_df, params)

        elif step_name == "missing":
            from app.modules.cleaning_submodules.missing import apply_missing
            config_df = apply_missing(config_df, params)

        elif step_name == "duplicates":
            from app.modules.cleaning_submodules.duplicates import apply_duplicates
            config_df = apply_duplicates(config_df, params)

        elif step_name == "outliers":
            from app.modules.cleaning_submodules.outliers import apply_outliers
            config_df = apply_outliers(config_df, params)

    return config_df

def create_pipeline_config():
    return {
        "version": 1,
        "pipeline": [],
        "metadata": {
            "columns": {}
        }
    }


def add_step(config, new_step):
    config = config or create_pipeline_config()

    pipeline = get_pipeline(config)
    step_name = new_step.get("step")

    updated_pipeline = [
        step for step in pipeline
        if step.get("step") != step_name
    ]

    updated_pipeline.append(new_step)

    config["pipeline"] = updated_pipeline

    return config

def reset_step(config, step_name):
    config = config or create_pipeline_config()

    pipeline = get_pipeline(config)

    config["pipeline"] = [
        step for step in pipeline
        if step.get("step") != step_name
    ]

    return config


def add_metadata(config, column, values):
    config = config or create_pipeline_config()

    columns = config.setdefault("metadata", {}).setdefault("columns", {})

    columns[column] = {
        **columns.get(column, {}),
        **values
    }

    return config