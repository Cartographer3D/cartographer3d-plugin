try:
    from cartographer.__version__ import version as __version__
except ImportError:
    __version__ = "unknown"

__all__ = ["__version__"]
