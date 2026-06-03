# Installation

topica ships as a compiled wheel. No Rust toolchain or JVM required.

```bash
pip install topica
```

or, with [uv](https://github.com/astral-sh/uv):

```bash
uv pip install topica
```

The only runtime dependency is **NumPy**. A few features light up if you also
have optional packages installed:

| Optional package | Enables |
|------------------|---------|
| `pyLDAvis` | Interactive intertopic-distance charts via [`prepare_pyldavis`](../api/diagnostics.md) |
| `matplotlib` | The `plot=True` figure from [`quality_frontier`](../api/diagnostics.md) |
| `pandas` | Convenient tabular handling of effect tables and diagnostics |

## Requirements

- Python 3.9+
- A platform with a prebuilt wheel (Linux, macOS, Windows on x86-64 / arm64).

## Building from source

If you want to build from the repository (e.g. to hack on the Rust core) you'll
need a Rust toolchain and [maturin](https://github.com/PyO3/maturin):

```bash
git clone https://github.com/nealcaren/topica
cd topica
python -m venv .venv && source .venv/bin/activate
pip install maturin numpy pytest
maturin develop --release --features python
pytest
```
