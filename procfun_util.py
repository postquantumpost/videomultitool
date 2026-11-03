"""Shared helpers for processing functions."""

import sys


class LazyModel:
    """Load a detector on first use; if loading fails, warn once and disable it."""

    def __init__(self, name, loader):
        self.name = name
        self._loader = loader
        self._model = None
        self._failed = False

    def get(self):
        if self._model is None and not self._failed:
            try:
                self._model = self._loader()
            except Exception as exc:  # missing package, model download failure, ...
                self._failed = True
                print(f"Warning: {self.name} unavailable, skipping: {exc}", file=sys.stderr)
        return self._model


def corners_to_box(x0, y0, x1, y1):
    """Convert corner coordinates to an (x, y, width, height) box."""
    x0, y0, x1, y1 = (int(round(float(v))) for v in (x0, y0, x1, y1))
    return (x0, y0, x1 - x0, y1 - y0)
