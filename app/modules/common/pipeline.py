# modules/common/pipeline.py

def run_pipeline(df, pipeline):
    for step in pipeline:
        if step["step"] == "formats":
            from app.modules.cleaning_submodules.formats import apply_formats
            df = apply_formats(df, step["params"])

        elif step["step"] == "missing":
            from app.modules.cleaning_submodules.missing import apply_missing
            df = apply_missing(df, step["params"])

        elif step["step"] == "duplicates":
            from app.modules.cleaning_submodules.duplicates_clean import apply_duplicates
            df = apply_duplicates(df, step["params"])

        elif step["step"] == "outliers":
            from app.modules.cleaning_submodules.outliers_clean import apply_outliers
            df = apply_outliers(df, step["params"])

    return df

def add_step(pipeline, new_step):
    pipeline = pipeline or []

    step_name = new_step.get("step")

    updated_pipeline = [
        step
        for step in pipeline
        if step.get("step") != step_name
    ]

    print("Pipeline reçu par outliers :", pipeline)
    print("Nouvelle étape :", new_step)

    updated_pipeline.append(new_step)

    print("Pipeline après add_step :", updated_pipeline)

    return updated_pipeline

def reset_step(pipeline, step_name):
    """
    Supprime une étape précise du pipeline sans modifier le DataFrame original.
    """
    return [
        step for step in (pipeline or [])
        if step["step"] != step_name
    ]