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


def keep_step_token(value: str) -> str:
    """Validate one ``--keep_ckpt_steps`` token: ``<int>``, ``final``, ``every=<int>`` or ``double=<int>``."""
    kind, _, n = value.partition("=")
    if value == "final" or (not n and kind.isdigit()) or (kind in ("every", "double") and n.isdigit() and int(n) > 0):
        return value
    raise argparse.ArgumentTypeError(f"bad step {value!r}: use an integer, 'final', 'every=N' or 'double=N'")


def resolve_keep_steps(tokens: list, total: int) -> list:
    """Expand ``--keep_ckpt_steps`` tokens into sorted optimizer steps in ``[0, total]``.

    ``every=N`` keeps N, 2N, ... and ``double=N`` keeps N, 2N, 4N, ... (log-spaced: early pretraining changes the
    model fastest); ``final`` is ``total``. Steps beyond ``total`` are dropped.
    """
    steps = set()
    for token in tokens:
        keep_step_token(token)
        kind, _, n = token.partition("=")
        if token == "final":
            steps.add(total)
        elif kind == "every":
            steps.update(range(int(n), total + 1, int(n)))
        elif kind == "double":
            step = int(n)
            while step <= total:
                steps.add(step)
                step *= 2
        else:
            steps.add(int(token))
    return sorted(s for s in steps if 0 <= s <= total)
