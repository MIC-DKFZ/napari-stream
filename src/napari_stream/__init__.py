try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"

from .sender import send

__all__ = []
