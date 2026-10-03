import os

DEFAULT_TRACKING_URI = "sqlite:///mlflow.db"


def tracking_uri() -> str:
    """MLflow tracking URI: ``$MLFLOW_TRACKING_URI`` if set, otherwise a local SQLite store in the working directory.

    Resolved at call time on purpose: Lightning's ``MLFlowLogger`` reads the env var as a default argument at import.
    """
    return os.environ.get("MLFLOW_TRACKING_URI") or DEFAULT_TRACKING_URI
