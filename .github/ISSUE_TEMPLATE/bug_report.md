---
name: Bug report
about: Report incorrect behavior, a crash, or a result that disagrees with a reference
title: ""
labels: bug
---

**What happened**
A clear description of the bug.

**Reproducer**
A minimal script (model, parameters, and the smallest data that triggers it). A
fixed `seed=` helps.

```python
import topica
...
```

**Expected vs actual**
What you expected, and what you got (full traceback if it errored).

**Environment**
- topica version (`python -c "import topica; print(topica.__version__)"`):
- Python version and OS:
- Installed from wheel (`pip install topica`) or built from source:
