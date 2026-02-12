# Prefer installed pypto (e.g. pip install -e ../pypto); fallback to pypto_all/pypto/python.
import sys
from pathlib import Path

try:
    import pypto.language as pl
except ImportError:
    pl = None  # type: ignore[assignment]
    _here = Path(__file__).resolve().parent
    _pypto_all = _here.parent.parent
    _pypto_python = _pypto_all / "pypto" / "python"
    if _pypto_python.exists() and str(_pypto_python) not in sys.path:
        sys.path.insert(0, str(_pypto_python))
    try:
        import pypto.language as pl  # type: ignore[no-redef]
    except ImportError:
        pass

__all__ = ["pl"]
