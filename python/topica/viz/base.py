"""The backend-neutral panel contract.

Every panel computes its numbers once and exposes three renderers:

- ``.to_frame()`` -- a pandas DataFrame of the numbers behind the picture. Always
  available; this is the reviewer-armor and reproducibility guarantee.
- ``.to_png(path=None)`` -- a matplotlib figure (PNG/PDF/SVG by extension). The
  publication renderer; available for every panel.
- ``.to_html(path=None)`` -- an interactive (Altair) build. Available only for the
  panels where interaction genuinely helps; the rest raise a clear message.
"""

from __future__ import annotations


def _require(module: str, extra: str):
    """Import an optional backend (sub)module or raise a clear install hint."""
    import importlib

    try:
        return importlib.import_module(module)
    except ImportError as exc:  # pragma: no cover - exercised via the message
        top = module.split(".")[0]
        raise ImportError(
            f"this needs {top}; install it with `pip install topica[{extra}]`"
        ) from exc


class Panel:
    """A computed view with matplotlib / Altair / DataFrame renderers."""

    #: Short human title, used as the default figure/axes title.
    title: str = ""
    #: Whether ``to_html`` produces a genuine interactive build.
    interactive: bool = False

    # --- subclasses implement these -----------------------------------------
    def to_frame(self):
        """The numbers behind the picture, as a pandas DataFrame."""
        raise NotImplementedError

    def _figure(self, **kwargs):
        """Build and return a matplotlib ``Figure``."""
        raise NotImplementedError

    def _altair(self, **kwargs):
        raise NotImplementedError(
            f"{type(self).__name__} has no interactive build; use to_png() or to_frame()"
        )

    # --- the public renderers -----------------------------------------------
    def to_png(self, path: str | None = None, *, dpi: int = 150, **kwargs):
        """Render to matplotlib. Returns the ``Figure``; also saves to ``path``
        (``.png`` / ``.pdf`` / ``.svg`` by extension) when given."""
        fig = self._figure(**kwargs)
        if path is not None:
            fig.savefig(path, dpi=dpi, bbox_inches="tight")
        return fig

    # An explicit alias, since publication output is the headline.
    def to_pdf(self, path: str, *, dpi: int = 300, **kwargs):
        return self.to_png(path, dpi=dpi, **kwargs)

    def to_html(self, path: str | None = None, **kwargs):
        """Render the interactive (Altair) build. Returns the chart; saves to
        ``path`` when given. Raises for panels that are static-only."""
        chart = self._altair(**kwargs)
        if path is not None:
            chart.save(path)
        return chart

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.title!r})"
