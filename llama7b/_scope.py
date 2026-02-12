# Incore scope directive: compiler recognizes this to insert incore boundary.
# At runtime this is a no-op; pto-rt2 / compiler uses it for scope_begin/scope_end and memory accounting.
from contextlib import contextmanager
from typing import Generator


@contextmanager
def incore_scope() -> Generator[None, None, None]:
    """Mark an incore scope. Compiler inserts scope boundary; runtime may do scope_begin/scope_end."""
    yield
