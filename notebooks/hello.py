# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "altair==6.2.2",
#     "marimo",
#     "numpy==2.5.2",
#     "polars==1.43.2",
# ]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium", auto_download=["html"])


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
    mo.md(f"""
    `n` is **{n.value}**, and `n**2` is **{n.value ** 2}**.
    """)
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


@app.cell
def _(n):
    import altair as alt
    import numpy as np
    import polars as pl

    # Partial sums of the Basel series, reactive to the `n` slider above.
    _k = np.arange(1, n.value + 1)
    _df = pl.DataFrame({"k": _k, "partial": np.cumsum(1.0 / _k**2)})
    _limit = float(np.pi**2 / 6)

    _line = (
        alt.Chart(_df)
        .mark_line(
            color="#2a78d6",
            strokeWidth=2,
            point={"color": "#2a78d6", "size": 55, "filled": True},
        )
        .encode(
            x=alt.X("k:Q", title="terms summed"),
            y=alt.Y("partial:Q", title="partial sum", scale=alt.Scale(zero=False)),
            tooltip=[
                alt.Tooltip("k:Q", title="terms"),
                alt.Tooltip("partial:Q", title="sum", format=".6f"),
            ],
        )
    )

    _rule = (
        alt.Chart(pl.DataFrame({"y": [_limit]}))
        .mark_rule(color="#8a8a85", strokeWidth=1, strokeDash=[4, 4])
        .encode(y="y:Q")
    )

    _label = (
        alt.Chart(pl.DataFrame({"y": [_limit], "t": ["limit = pi^2/6 = 1.644934"]}))
        .mark_text(align="right", baseline="bottom", dx=-4, dy=-4,
                   color="#52514e", fontSize=11)
        .encode(x=alt.value(514), y="y:Q", text="t:N")
    )

    (
        (_line + _rule + _label)
        .properties(
            width=520,
            height=300,
            title="Basel series: partial sums of 1/k^2 converge to pi^2/6",
        )
        .configure_axis(
            gridColor="#e8e8e4", domainColor="#d4d4cf", tickColor="#d4d4cf",
            labelColor="#52514e", titleColor="#52514e",
        )
        .configure_view(stroke=None)
        .configure_title(color="#0b0b0b", fontSize=13, anchor="start")
    )
    return


if __name__ == "__main__":
    app.run()
