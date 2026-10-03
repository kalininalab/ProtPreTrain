import os
from contextlib import nullcontext

DEFAULT_TRACKING_URI = "sqlite:///mlflow.db"


def tracking_uri() -> str:
    """MLflow tracking URI: ``$MLFLOW_TRACKING_URI`` if set, otherwise a local SQLite store in the working directory.

    Resolved at call time on purpose: Lightning's ``MLFlowLogger`` reads the env var as a default argument at import.
    """
    return os.environ.get("MLFLOW_TRACKING_URI") or DEFAULT_TRACKING_URI


def mlflow_logger(experiment: str):
    """Lightning ``MLFlowLogger`` for ``experiment`` on :func:`tracking_uri`.

    The store's tables and the experiment are created under a file lock: processes that open a fresh SQLite store at
    the same time otherwise race on MLflow's schema migration and experiment creation, and all of them fail.
    """
    import mlflow
    from filelock import FileLock
    from lightning.pytorch.loggers import MLFlowLogger

    uri = tracking_uri()
    lock = FileLock(uri.removeprefix("sqlite:///") + ".lock") if uri.startswith("sqlite:///") else nullcontext()
    with lock:
        client = mlflow.MlflowClient(uri)
        if client.get_experiment_by_name(experiment) is None:
            client.create_experiment(experiment)
        return MLFlowLogger(experiment_name=experiment, tracking_uri=uri)
