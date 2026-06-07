"""The backend-neutral panel contract.

Every panel computes its numbers once and exposes three renderers:

- ``.to_frame()`` -- a pandas DataFrame of the numbers behind the picture. Always
  available; this is the reviewer-armor and reproducibility guarantee.
- ``.to_png(path=None)`` -- a matplotlib figure (PNG/PDF/SVG by extension). The
  publication renderer; available for every panel.
- ``.to_html(path=None)`` -- an interactive (Plotly) build. Available only for the
  panels where interaction genuinely helps; the rest raise a clear message.
"""

from __future__ import annotations

# Shared color idiom across panels, so a reader flipping between figures reads the
# same encoding everywhere: SEQ for magnitudes anchored at 0 (similarity, prevalence,
# P(w|t,g)), DIV for signed quantities centered at 0 (correlations, effects).
SEQ_CMAP = "viridis"
DIV_CMAP = "RdBu_r"


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
    """A computed view with matplotlib / Plotly / DataFrame renderers."""

    #: Short human title, used as the default figure/axes title.
    title: str = ""
    #: Whether ``to_html`` produces a genuine interactive (Plotly) build.
    interactive: bool = False

    # --- subclasses implement these -----------------------------------------
    def to_frame(self):
        """The numbers behind the picture, as a pandas DataFrame."""
        raise NotImplementedError

    def _draw(self, fig, **kwargs):
        """Draw the panel into the given matplotlib ``Figure`` or ``SubFigure``.
        Must create its own axes inside ``fig`` (so the panel composes into a
        dashboard's subfigure and stays vector). Does not call ``tight_layout``."""
        raise NotImplementedError

    def _figsize(self):
        """The standalone figure size (width, height) in inches."""
        return (6.0, 4.5)

    def _figure(self, *, figsize=None, **kwargs):
        """Build a standalone ``Figure`` around ``_draw``."""
        plt = _require("matplotlib.pyplot", "viz")
        fig = plt.figure(figsize=figsize or self._figsize())
        self._draw(fig, **kwargs)
        fig.tight_layout()
        return fig

    def _interactive(self, **kwargs):
        raise NotImplementedError(
            f"{type(self).__name__} has no interactive build; use to_png() or to_frame()"
        )

    # --- the public renderers -----------------------------------------------
    def to_png(self, path: str | None = None, *, dpi: int = 150, **kwargs):
        """Render to matplotlib. Returns the ``Figure``; also saves to ``path``
        (``.png`` / ``.pdf`` / ``.svg`` by extension) when given. Vector formats
        (``.pdf`` / ``.svg``) stay vector with selectable text."""
        fig = self._figure(**kwargs)
        if path is not None:
            fig.savefig(path, dpi=dpi, bbox_inches="tight")
        return fig

    # An explicit alias, since publication output is the headline.
    def to_pdf(self, path: str, *, dpi: int = 300, **kwargs):
        return self.to_png(path, dpi=dpi, **kwargs)

    def to_html(self, path: str | None = None, **kwargs):
        """Render the interactive (Plotly) build. Returns the figure; saves to
        ``path`` when given. Raises for panels that are static-only."""
        fig = self._interactive(**kwargs)
        if path is not None:
            fig.write_html(path)
        return fig

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.title!r})"
