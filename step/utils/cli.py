import argparse


def str_to_bool(value: str) -> bool:
    """Command line inputs that are bools."""
    if isinstance(value, bool):
        return value
    if value.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif value.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def progress_bar():
    """Rich progress bar on a terminal; a sparse tqdm bar otherwise, so batch-job logs don't fill with redraws."""
    import sys

    import lightning.pytorch as pl

    if sys.stdout.isatty():
        return pl.callbacks.RichProgressBar()
    return pl.callbacks.TQDMProgressBar(refresh_rate=200)
