from beartype.claw import beartype_this_package

from . import data, utils
from .version import version as __version__

beartype_this_package()

__all__ = ["protpretrain", "__version__"]
