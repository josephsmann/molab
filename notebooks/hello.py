# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
# ]
# ///

import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md("""
    # Hello from molab

    A minimal reactive notebook used to verify the
    **local → GitHub → molab** round trip.

    Dependencies are declared inline (PEP 723) at the top of this
    file, so the notebook reproduces its own environment anywhere.
    """)
    return


@app.cell
def _(mo):
    n = mo.ui.slider(1, 50, value=12, label="n")
    n
    return (n,)


@app.cell
def _(mo, n):
    mo.md(f"`n` is **{n.value}**, and `n**2` is **{n.value ** 2}**.")
    return


@app.cell
def _():
    import os
    import platform
    import sys

    return os, platform, sys


@app.cell
def _(mo, os, platform, sys):
    def _mem_gb():
        try:
            return round(
                os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024**3, 1
            )
        except (ValueError, OSError, AttributeError):
            return None

    _rows = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpus": os.cpu_count(),
        "memory (GB)": _mem_gb(),
    }

    mo.md(
        "### Environment\n\n"
        "| | |\n|---|---|\n"
        + "\n".join(f"| {k} | `{v}` |" for k, v in _rows.items())
    )
    return


if __name__ == "__main__":
    app.run()
