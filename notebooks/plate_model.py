# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
# ]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium", auto_download=["html"])


@app.cell
def _():
    import marimo as mo
    import ast
    import re
    from dataclasses import dataclass, field

    return ast, dataclass, field, mo, re


@app.cell
def _(mo):
    mo.md(r"""
    # Plate models, written as text

    A **plate model** is the compact drawing of a graphical probabilistic
    model: a circle per random variable, an arrow per conditional
    dependence, and a rectangle — the *plate* — around whatever repeats.

    Below is a small line-oriented syntax for them. It is deliberately
    shaped like the generative model rather than like a picture: `for`
    loops become plates, `~` statements become nodes, and edges fall out of
    which declared names appear on a right-hand side. The diagram redraws
    as you type, the same model exports to TikZ, daft, Graphviz DOT and
    Mermaid, and it will write you a **PyMC** or **Stan** scaffold to start
    fitting from.

    **Hover any node** to see exactly how it relates to its parents: the
    statement as you wrote it, the parents it depends on, the plates it
    sits in — with those parents and the arrows into it lit up.

    Everything here is pure Python with no third-party dependency, so it
    renders in the WASM preview as happily as in a sandbox.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## The syntax

    | you write | you get |
    |:---|:---|
    | `mu ~ Normal(0, 1)` | latent node, drawn as an open circle |
    | `y[i] ~ Normal(mu, sigma) obs` | observed node — shaded, subscript `i` |
    | `const alpha, beta` (or `alpha ~ const`) | fixed hyperparameter — a small solid dot |
    | `yhat[i] = a[j] + b * x[i]` | deterministic node — dashed outline |
    | `for i in 1..N:` + an indented block | a plate labelled `i = 1..N`; indent again to nest |
    | `cloudy -> rain -> wet` | plain edges, for when distributions don't matter |
    | `# ...` | comment |

    Two rules do most of the work:

    1. **A name on a right-hand side becomes a parent only if it is declared
       somewhere.** So `Normal`, `0.5` and loop indices are ignored, and you
       never have to write an edge twice.
    2. **Names render like maths.** `sigma_a` → σ<sub>a</sub>, `theta[d]` →
       θ<sub>d</sub>, `yhat` → ŷ, `x_bar` → x̄.
    """)
    return


@app.cell
def _(EXAMPLES, mo):
    example = mo.ui.dropdown(
        EXAMPLES, value="Latent Dirichlet allocation", label="Example"
    )
    direction = mo.ui.radio(
        ["top-down", "left-right"], value="top-down", inline=True, label="Flow"
    )
    zoom = mo.ui.slider(0.5, 2.0, 0.1, value=1.0, label="Zoom", show_value=True)
    mo.hstack([example, direction, zoom], justify="start", gap=2)
    return direction, example, zoom


@app.cell
def _(example, mo):
    source = mo.ui.code_editor(
        value=example.value,
        language="python",
        debounce=350,
        min_height=280,
        label="Model",
    )
    source
    return (source,)


@app.cell
def _(direction, mo, parse, source, to_svg, zoom):
    model = parse(source.value)
    svg = to_svg(
        model,
        direction="LR" if direction.value == "left-right" else "TB",
        scale=zoom.value,
    )
    _notes = (
        mo.md("\n".join(f"- ⚠️ {w}" for w in model.warnings))
        if model.warnings
        else mo.md("")
    )
    mo.vstack([mo.Html(f'<div style="overflow-x: auto">{svg}</div>'), _notes])
    return model, svg


@app.cell
def _(mo):
    pymc_runnable = mo.ui.switch(value=True, label="runnable")
    run_fit = mo.ui.run_button(label="▶ Simulate and fit", kind="success")
    mo.hstack(
        [
            mo.md("**PyMC tab** — "),
            pymc_runnable,
            mo.md("&nbsp;"),
            run_fit,
        ],
        justify="start",
        gap=0.6,
    )
    return pymc_runnable, run_fit


@app.cell
def _(
    mo,
    model,
    pymc_runnable,
    svg,
    to_daft,
    to_dot,
    to_mermaid,
    to_pymc,
    to_stan,
    to_tikz,
):
    def _tab(text, language, filename, mimetype="text/plain"):
        return mo.vstack([
            mo.download(
                text.encode(),
                filename=filename,
                mimetype=mimetype,
                label=f"Download {filename}",
            ),
            mo.md(f"```{language}\n{text}\n```"),
        ])

    mo.ui.tabs({
        "TikZ": _tab(to_tikz(model), "latex", "model.tex"),
        "daft": _tab(to_daft(model), "python", "model_daft.py"),
        "Graphviz": _tab(to_dot(model), "dot", "model.dot"),
        "PyMC": _tab(to_pymc(model, runnable=pymc_runnable.value), "python",
                     "model_pymc.py"),
        "Stan": _tab(to_stan(model), "stan", "model.stan"),
        "Mermaid": mo.vstack([
            mo.mermaid(to_mermaid(model)),
            _tab(to_mermaid(model), "text", "model.mmd"),
        ]),
        "SVG": mo.vstack([
            mo.download(
                svg.encode(),
                filename="model.svg",
                mimetype="image/svg+xml",
                label="Download model.svg",
            ),
            mo.md(f"`{len(svg)}` characters of self-contained SVG — no fonts or "
                  "scripts to ship with it."),
        ]),
    })
    return


@app.cell
def _(mo, model, run_fit, to_pymc):
    mo.stop(
        not run_fit.value,
        mo.md(
            """
            /// tip | Nothing fitted yet

            Press **▶ Simulate and fit** above. It runs the runnable version of
            the PyMC code — a dataset drawn from this very model, with the true
            parameters kept, then fitted back — and compares what came out with
            what went in. Needs `pymc` in the environment; a model with discrete
            latents can take a minute.
            ///
            """
        ),
    )

    _code = to_pymc(model, runnable=True)
    mo.stop(
        _code.startswith("# Nothing to run"),
        mo.md("**Nothing to fit.** " + _code.splitlines()[0].removeprefix("# ")),
    )

    _env = {}
    try:
        exec(compile(_code, "<generated pymc>", "exec"), _env)  # noqa: S102
        _failure = None
    except ImportError as _exc:
        _failure = mo.md(
            f"""
            **`{_exc.name}` is not installed here.** The generated code needs it:

            ```sh
            uv pip install pymc arviz
            ```
            """
        )
    except Exception as _exc:  # noqa: BLE001
        _failure = mo.md(
            f"""
            **The generated program did not run.**

            ```
            {type(_exc).__name__}: {_exc}
            ```

            Distribution arguments are passed through exactly as you wrote them,
            so a right-hand side that reads as a dependency list rather than a
            real call (`Categorical(z, A)`) will not execute. The **Stan** tab
            and the plain scaffold are unaffected.
            """
        )
    fit = None if _failure else dict(_env["results"], code=_code)
    _failure if _failure else mo.md("Fitted. Results below.")
    return (fit,)


@app.cell
def _(DISCRETE, fit, mo, model, split_call):
    mo.stop(fit is None, mo.md(""))

    import numpy as np
    import pandas as pd

    # Straight from the posterior samples: arviz would do this too, but
    # importing it drags in matplotlib, which this notebook does not need.
    _rhat = {}
    try:
        import arviz as az

        _rhat = {k: v.values for k, v in az.rhat(fit["idata"]).data_vars.items()}
    except Exception:  # noqa: BLE001 — r_hat is a nicety, not a requirement
        _rhat = {}

    # Discrete latent labels (the `z` of a mixture) are left out: their
    # posterior is over labels that swap between draws, so "did it recover the
    # truth" is not a question they can answer.
    labels = {
        _n.name for _n in model.nodes.values()
        if _n.kind == "latent" and split_call(_n.dist or "")[0].lower() in DISCRETE
    }

    _rows = []
    for _name, _samples in fit["idata"].posterior.data_vars.items():
        if _name in fit["data"] or _name not in fit["truth"] or _name in labels:
            continue                      # the data itself is not a parameter
        _drawn = np.asarray(_samples.values)
        _flat = _drawn.reshape(_drawn.shape[0] * _drawn.shape[1], -1)
        _true = np.asarray(fit["truth"][_name]).reshape(-1)
        _shaky = np.asarray(_rhat.get(_name, np.full(_flat.shape[1], np.nan))).reshape(-1)
        for _i, _index in enumerate(np.ndindex(_drawn.shape[2:])):
            _rows.append({
                "parameter": _name + ("[" + ",".join(map(str, _index)) + "]" if _index else ""),
                "true": round(float(_true[_i]), 3),
                "posterior mean": round(float(_flat[:, _i].mean()), 3),
                "sd": round(float(_flat[:, _i].std()), 3),
                "r_hat": round(float(_shaky[_i]), 3) if _shaky.size > _i else float("nan"),
            })

    recovered = pd.DataFrame(_rows)
    return labels, pd, recovered


@app.cell
def _(labels, mo, pd, recovered):
    mo.stop(recovered.empty, mo.md("*Nothing to compare — no parameters came back.*"))

    _span = [
        float(min(recovered["true"].min(), recovered["posterior mean"].min())),
        float(max(recovered["true"].max(), recovered["posterior mean"].max())),
    ]
    _within = int(
        (recovered["true"] - recovered["posterior mean"]).abs().le(recovered["sd"]).sum()
    )

    _aside = (
        " Discrete latent labels (" + ", ".join(sorted(labels)) + ") are left out:"
        " their labels swap between draws, so recovery is not a question they can"
        " answer." if labels else ""
    )

    try:
        import altair as alt

        _line = alt.Chart(pd.DataFrame({"v": _span})).mark_line(
            color="#c9c8bf", strokeWidth=2, strokeDash=[4, 4]
        ).encode(x="v:Q", y="v:Q")
        _bars = alt.Chart(recovered).transform_calculate(
            lo="datum['posterior mean'] - datum.sd",
            hi="datum['posterior mean'] + datum.sd",
        ).mark_rule(color="#2a78d6", strokeWidth=2, opacity=0.35).encode(
            x=alt.X("true:Q", title="true value (what was simulated)"),
            y=alt.Y("lo:Q", title="posterior mean ± 1 sd"),
            y2="hi:Q",
        )
        _dots = alt.Chart(recovered).mark_point(
            filled=True, size=90, color="#2a78d6", stroke="white", strokeWidth=1.5
        ).encode(
            x="true:Q",
            y=alt.Y("posterior mean:Q"),
            tooltip=["parameter", "true", "posterior mean", "sd", "r_hat"],
        )
        _picture = (_line + _bars + _dots).properties(
            width=460, height=320,
            title="Recovery: each parameter, what went in against what came out",
        )
    except ImportError:
        _picture = mo.md("*(altair not installed — table only)*")

    mo.vstack([
        mo.md(
            f"""
            ### Did it get the truth back?

            One dataset was drawn from this very model with a fixed seed, so
            every parameter has a known true value. Of **{len(recovered)}**
            parameters, **{_within}** land within one posterior sd of it.
            Points on the dashed line were recovered exactly; the model's own
            symmetries — exchangeable mixture components, say — push points off
            it however well the sampler did.{_aside}
            """
        ),
        mo.hstack([_picture, mo.ui.table(recovered, selection=None, page_size=12)],
                  widths=[1, 1], gap=1),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Under the hood

    The rest of the notebook is the implementation, in reading order:

    1. **Parse** — one pass over the lines. Indentation opens and closes
       plates, `~` / `=` / `->` declare nodes, and a second pass turns every
       declared name found on a right-hand side into an edge.
    2. **Layer** — each node sits one row below its deepest parent
       (longest-path layering); parentless nodes then sink to just above
       their children so hyperparameters don't float at the top.
    3. **Order and straighten** — barycentre sweeps put connected nodes
       near each other, reordering only *within* a plate so plate members
       stay contiguous, and edges spanning more than one row get invisible
       waypoints to bend around what's in the way.
    4. **Separate** — plates are nudged apart until no plate is drawn over
       a foreign plate or node. (The self-checks below assert exactly that,
       for every example, in both directions.)
    5. **Render** — SVG by string concatenation, and the same layout feeds
       the TikZ, daft, DOT and Mermaid exporters. The PyMC and Stan
       generators work from the parse instead: the plate tree becomes
       `dims` in one and `for` loops in the other.

    The hover behaviour is CSS, not JavaScript: every node, edge and
    tooltip gets an id, and one generated `svg:has(#node:hover) …` rule per
    node lights up that node's parents, its incoming arrows and its card.
    Cards are placed on whichever side of the node hides the least. It
    costs nothing at runtime and survives the download — open the exported
    `model.svg` in a browser and the hovering still works.
    """)
    return


@app.cell
def _():
    GREEK = {
        "alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ", "epsilon": "ε",
        "zeta": "ζ", "eta": "η", "theta": "θ", "iota": "ι", "kappa": "κ",
        "lambda": "λ", "mu": "μ", "nu": "ν", "xi": "ξ", "pi": "π", "rho": "ρ",
        "sigma": "σ", "tau": "τ", "upsilon": "υ", "phi": "φ", "chi": "χ",
        "psi": "ψ", "omega": "ω", "Gamma": "Γ", "Delta": "Δ", "Theta": "Θ",
        "Lambda": "Λ", "Xi": "Ξ", "Pi": "Π", "Sigma": "Σ", "Phi": "Φ",
        "Psi": "Ψ", "Omega": "Ω",
    }


    ACCENTS = {"hat": "̂", "bar": "̄", "tilde": "̃"}


    OBS_WORDS = {"obs", "observed", "@obs", "@observed", "data"}


    FIXED_WORDS = {"const", "fixed", "hyper"}
    return ACCENTS, FIXED_WORDS, GREEK, OBS_WORDS


@app.cell
def _(dataclass, field):
    @dataclass
    class Node:
        """One variable. `kind` is latent | obs | fixed | det."""

        name: str
        kind: str = "latent"
        sub: str = ""          # subscript from `x[i,j]`
        dist: str = ""         # right-hand side, verbatim
        plates: tuple = ()     # enclosing plate ids, outermost first
        order: int = 0         # declaration order


    @dataclass
    class Plate:
        pid: str
        var: str
        span: str
        parent: str | None = None
        depth: int = 0


    @dataclass
    class Model:
        nodes: dict = field(default_factory=dict)
        plates: dict = field(default_factory=dict)
        edges: list = field(default_factory=list)
        warnings: list = field(default_factory=list)

        def parents(self, name):
            return [s for s, d in self.edges if d == name]

        def children(self, name):
            return [d for s, d in self.edges if s == name]

    return Model, Node, Plate


@app.cell
def _(FIXED_WORDS, Model, Node, OBS_WORDS, Plate, re):
    IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z_0-9]*")


    FOR_RE = re.compile(r"^(?:for|plate)\s+([A-Za-z_]\w*)\s+in\s+(.+?)\s*:?\s*$", re.I)


    DECL_RE = re.compile(r"^([A-Za-z_]\w*)\s*(?:\[([^\]]*)\])?\s*(~|=)\s*(.*)$")


    BARE_RE = re.compile(r"^([A-Za-z_]\w*)\s*(?:\[([^\]]*)\])?\s*$")


    def split_marker(text):
        """Peel a trailing `obs` marker off a statement."""
        parts = text.rsplit(None, 1)
        if len(parts) == 2 and parts[1].lower() in OBS_WORDS:
            return parts[0].rstrip(), "obs"
        if text.strip().lower() in OBS_WORDS:
            return text, None
        return text, None


    def parse(src: str) -> Model:
        """Parse the plate DSL. Never raises — problems land in `model.warnings`."""
        model = Model()
        stack = []      # [(indent, plate_id)]
        pending = []    # (child, rhs) — edges resolved once all names are known
        seq = [0]

        def declare(name, sub, kind, dist):
            node = model.nodes.get(name)
            if node is None:
                seq[0] += 1
                node = Node(name, kind, sub, dist, tuple(p for _, p in stack), seq[0])
                model.nodes[name] = node
            else:
                if kind != "latent":
                    node.kind = kind
                node.sub = node.sub or sub
                node.dist = node.dist or dist
            return node

        for lineno, raw in enumerate(src.splitlines(), 1):
            line = raw.split("#", 1)[0].expandtabs(4).rstrip()
            if not line.strip():
                continue
            indent = len(line) - len(line.lstrip())
            while stack and indent <= stack[-1][0]:
                stack.pop()
            stmt = line.strip()

            m = FOR_RE.match(stmt)
            if m:
                pid = "p%03d" % (len(model.plates) + 1)
                model.plates[pid] = Plate(
                    pid, m.group(1), m.group(2).strip(),
                    stack[-1][1] if stack else None, len(stack),
                )
                stack.append((indent, pid))
                continue

            head = stmt.split(None, 1)[0].lower()
            if head in FIXED_WORDS and len(stmt.split(None, 1)) == 2:
                for name in re.split(r"[,\s]+", stmt.split(None, 1)[1]):
                    if BARE_RE.match(name or "x"):
                        declare(BARE_RE.match(name).group(1), "", "fixed", "")
                continue

            if "->" in stmt:
                chain = [c.strip() for c in stmt.split("->")]
                if all(BARE_RE.match(c) for c in chain):
                    for part in chain:
                        m2 = BARE_RE.match(part)
                        declare(m2.group(1), m2.group(2) or "", "latent", "")
                    for a, b in zip(chain, chain[1:]):
                        edge = (BARE_RE.match(a).group(1), BARE_RE.match(b).group(1))
                        if edge not in model.edges:
                            model.edges.append(edge)
                else:
                    model.warnings.append(f"line {lineno}: cannot read `{stmt}`")
                continue

            stmt, marker = split_marker(stmt)
            m = DECL_RE.match(stmt)
            if m:
                name, sub, op, rhs = m.group(1), m.group(2) or "", m.group(3), m.group(4).strip()
                kind = "det" if op == "=" else "latent"
                if rhs.lower() in FIXED_WORDS:
                    kind, rhs = "fixed", ""
                if marker == "obs":
                    kind = "obs"
                declare(name, sub, kind, rhs)
                if rhs:
                    pending.append((name, rhs))
                continue

            m = BARE_RE.match(stmt)
            if m:
                declare(m.group(1), m.group(2) or "", "obs" if marker else "latent", "")
                continue

            model.warnings.append(f"line {lineno}: cannot read `{stmt}`")

        # An identifier on a right-hand side is a parent iff it names a declared
        # node — so distribution names, loop indices and literals are ignored.
        for child, rhs in pending:
            for m in IDENT_RE.finditer(rhs):
                name = m.group(0)
                if rhs[m.end():m.end() + 1] == "(" or name == child:
                    continue
                if name in model.nodes and (name, child) not in model.edges:
                    model.edges.append((name, child))
        return model

    return IDENT_RE, parse


@app.cell
def _(ACCENTS, GREEK):
    def split_name(name):
        """`sigma_a` -> ('σ', 'a'); `yhat` -> ('ŷ', ''). Returns (display, subscript)."""
        parts = name.split("_")
        base, rest = parts[0], parts[1:]
        accent = ""
        while rest and rest[-1] in ACCENTS:
            accent = ACCENTS[rest.pop()]
        if not accent and base not in GREEK:
            for suffix, mark in ACCENTS.items():
                if base.endswith(suffix) and len(base) > len(suffix):
                    base, accent = base[: -len(suffix)], mark
                    break
        return GREEK.get(base, base) + accent, ",".join(rest)


    def node_label(node):
        """(display, subscript) for a node, merging name and bracket subscripts."""
        disp, sub = split_name(node.name)
        extra = ",".join(split_name(s.strip())[0] for s in node.sub.split(",") if s.strip())
        subs = [s for s in (sub, extra) if s]
        return disp, ",".join(subs)


    def tex_label(name, sub=""):
        """LaTeX for a node name, e.g. `sigma_a` -> `\\sigma_{a}`."""
        parts = name.split("_")
        base, rest = parts[0], parts[1:]
        accent = ""
        while rest and rest[-1] in ACCENTS:
            rest.pop()
            accent = "hat" if "hat" in name else ("bar" if "bar" in name else "tilde")
        if not accent and base not in GREEK:
            for suffix in ACCENTS:
                if base.endswith(suffix) and len(base) > len(suffix):
                    base, accent = base[: -len(suffix)], suffix
                    break
        core = f"\\{base}" if base in GREEK else base
        if accent:
            core = f"\\{accent}{{{core}}}"
        subs = [s for s in (",".join(rest), sub) if s]
        return core + (f"_{{{','.join(subs)}}}" if subs else "")

    return node_label, split_name, tex_label


@app.cell
def _():
    NODE_R = 24


    X_GAP = 44


    Y_GAP = 104


    PAD = 15


    MARGIN = 18


    def rank_nodes(model):
        """Longest-path layering: a node sits one level below its deepest parent."""
        rank, state = {}, {}

        def visit(name):
            if state.get(name) == "done":
                return rank[name]
            if state.get(name) == "open":
                note = f"cycle through `{name}` — a plate model should be acyclic"
                if note not in model.warnings:
                    model.warnings.append(note)
                return 0
            state[name] = "open"
            parents = [p for p in model.parents(name) if p in model.nodes]
            rank[name] = 1 + max([visit(p) for p in parents], default=-1)
            state[name] = "done"
            return rank[name]

        for name in model.nodes:
            visit(name)

        # Parentless nodes (hyperparameters, data) sink to just above their
        # children, so they sit next to what they feed instead of floating at the top.
        for _ in range(3):
            for name in model.nodes:
                kids = [rank[c] for c in model.children(name) if c in rank]
                if kids and not model.parents(name):
                    rank[name] = min(kids) - 1
        floor = min(rank.values(), default=0)
        return {n: r - floor for n, r in rank.items()}


    def shared_plates(a, b):
        """Longest common plate prefix of two plate paths."""
        out = []
        for pa, pb in zip(a, b):
            if pa != pb:
                break
            out.append(pa)
        return tuple(out)


    def row_gap(a, b):
        """Minimum centre distance between two neighbouring cells in a row."""
        apart = len(a["plates"]) + len(b["plates"]) - 2 * len(shared_plates(a["plates"], b["plates"]))
        return a["hw"] + b["hw"] + X_GAP - 8 + 22 * apart


    def build_cells(model, rank):
        """Real nodes plus dummy waypoints for edges spanning more than one rank."""
        cells, rows = {}, {}
        for name, node in model.nodes.items():
            cells[name] = {
                "key": name, "kind": "node", "node": node, "rank": rank[name],
                "plates": node.plates, "hw": NODE_R, "order": node.order,
                "up": [], "down": [], "x": 0.0,
            }
        routes = {}
        for src, dst in model.edges:
            if src not in cells or dst not in cells:
                continue
            chain = [src]
            for r in range(rank[src] + 1, rank[dst]):
                key = f"~{src}>{dst}@{r}"
                cells[key] = {
                    "key": key, "kind": "dummy", "node": None, "rank": r,
                    "plates": shared_plates(model.nodes[src].plates, model.nodes[dst].plates),
                    "hw": 5, "order": model.nodes[src].order, "up": [], "down": [], "x": 0.0,
                }
                chain.append(key)
            chain.append(dst)
            routes[(src, dst)] = chain[1:-1]
            for a, b in zip(chain, chain[1:]):
                cells[a]["down"].append(b)
                cells[b]["up"].append(a)
        for cell in cells.values():
            rows.setdefault(cell["rank"], []).append(cell)
        for row in rows.values():
            row.sort(key=lambda c: (c["plates"], c["order"], c["key"]))
        return cells, rows, routes


    def order_rows(cells, rows):
        """Barycentre sweeps, reordering only inside a plate group."""
        index = {c["key"]: i for row in rows.values() for i, c in enumerate(row)}
        for sweep in range(8):
            for r in sorted(rows, reverse=sweep % 2 == 1):
                side = "up" if sweep % 2 == 0 else "down"

                def bary(cell):
                    nbrs = [index[n] for n in cell[side] if n in index]
                    return sum(nbrs) / len(nbrs) if nbrs else index[cell["key"]]

                rows[r].sort(key=lambda c: (c["plates"], bary(c), c["order"]))
                for i, cell in enumerate(rows[r]):
                    index[cell["key"]] = i

    return (
        MARGIN,
        NODE_R,
        PAD,
        Y_GAP,
        build_cells,
        order_rows,
        rank_nodes,
        row_gap,
    )


@app.cell
def _(
    MARGIN,
    NODE_R,
    PAD,
    Y_GAP,
    build_cells,
    order_rows,
    rank_nodes,
    row_gap,
):
    def pull_weight(cell, other):
        """How hard `other` pulls `cell` sideways: same plate first."""
        if other["kind"] != "node":
            return 0.35
        return 1.0 if other["plates"] == cell["plates"] else 0.2


    def straighten(cells, rows):
        """Pack each row left to right, then pull rows toward their neighbours."""
        for row in rows.values():
            x = 0.0
            for i, cell in enumerate(row):
                x += row_gap(row[i - 1], cell) if i else 0.0
                cell["x"] = x

        for sweep in range(10):
            for r in sorted(rows, reverse=sweep % 2 == 1):
                row = rows[r]
                side = "up" if sweep % 2 == 0 else "down"
                want = []
                for cell in row:
                    # A node is pulled hardest by neighbours on its own plate, so
                    # plate members stay in a column instead of trailing after
                    # whatever hyperparameter happens to point at them.
                    pull = [(cells[n]["x"], pull_weight(cell, cells[n]))
                            for n in cell[side] if n in cells]
                    total = sum(w for _, w in pull)
                    want.append(sum(x * w for x, w in pull) / total if total else cell["x"])
                x = want[0]
                for i, cell in enumerate(row):
                    if i:
                        x = max(want[i], x + row_gap(row[i - 1], cell))
                    cell["x"] = x
                drift = sum(w - c["x"] for w, c in zip(want, row)) / len(row)
                for cell in row:
                    cell["x"] += drift

        left = min(c["x"] - c["hw"] for c in cells.values())
        for cell in cells.values():
            cell["x"] -= left


    def nested_plates(model, a, b):
        """True if either plate contains the other."""
        for x, y in ((a, b), (b, a)):
            while x:
                if x == y:
                    return True
                x = model.plates[x].parent
        return False


    def overlap(a, b):
        """Overlap of two boxes as (dx, dy); positive on both means they clash."""
        return (min(a[2], b[2]) - max(a[0], b[0]), min(a[3], b[3]) - max(a[1], b[1]))


    def cell_box(cell):
        return (cell["x"] - cell["hw"], cell["y"] - cell["hw"],
                cell["x"] + cell["hw"], cell["y"] + cell["hw"])


    def find_clash(model, cells, rects, ax):
        """First plate drawn over a foreign plate or node, as (what, key, shift)."""
        lo, hi = ax, ax + 2
        for a, b in ((a, b) for a in rects for b in rects if a < b):
            if nested_plates(model, a, b):
                continue
            near, far = sorted((rects[a], rects[b]), key=lambda r: r[lo])
            if min(overlap(near, far)) > 0:
                return "plate", (a if rects[a] is far else b), near[hi] + MARGIN - far[lo]
        for pid, rect in rects.items():
            middle = (rect[lo] + rect[hi]) / 2
            for cell in cells.values():
                if cell["kind"] != "node" or pid in cell["plates"]:
                    continue
                box = cell_box(cell)
                if min(overlap(rect, box)) <= 0:
                    continue
                if box[lo] < middle:
                    return "plate", pid, box[hi] + MARGIN - rect[lo]
                return "cell", cell["key"], rect[hi] + MARGIN - box[lo]
        return None


    def compact_rows(cells, rows, key):
        """Restore minimum spacing inside each row, pushing one way only."""
        for row in rows.values():
            for i, cell in enumerate(row[1:], 1):
                cell[key] = max(cell[key], row[i - 1][key] + row_gap(row[i - 1], cell))


    def separate(model, cells, rows, ax):
        """Nudge plates apart until none is drawn over a stranger."""
        key = "xy"[ax]
        rects = plate_rects(model, cells)
        for _ in range(10):
            clash = find_clash(model, cells, rects, ax)
            if not clash:
                break
            what, target, shift = clash
            if shift <= 0:
                break
            for cell in cells.values():
                moved = (target in cell["plates"] if what == "plate"
                         else cell[key] >= cells[target][key])
                if moved:
                    cell[key] += shift
            compact_rows(cells, rows, key)
            rects = plate_rects(model, cells)
        return rects


    def plate_rects(model, cells):
        """A rectangle wrapping each plate's members and its nested plates."""
        rects = {}

        def rect_for(pid):
            if pid in rects:
                return rects[pid]
            boxes = [
                (c["x"] - c["hw"], c["y"] - c["hw"], c["x"] + c["hw"], c["y"] + c["hw"])
                for c in cells.values() if pid in c["plates"]
            ]
            boxes += [r for r in (rect_for(c) for c, p in model.plates.items()
                                  if p.parent == pid) if r]
            if not boxes:
                return None
            rects[pid] = (
                min(b[0] for b in boxes) - PAD, min(b[1] for b in boxes) - PAD,
                max(b[2] for b in boxes) + PAD, max(b[3] for b in boxes) + PAD + 16,
            )
            return rects[pid]

        for pid in model.plates:
            rect_for(pid)
        return {p: r for p, r in rects.items() if r}


    def layout(model, direction="TB"):
        """Positions for nodes, waypoints for edges, rectangles for plates."""
        rank = rank_nodes(model)
        cells, rows, routes = build_cells(model, rank)
        if not cells:
            return {}, {}, {}, [0, 0, 0, 0]
        order_rows(cells, rows)
        straighten(cells, rows)

        y, ys = 0.0, {}
        for r in sorted(rows):
            if r in ys:
                continue
            if ys:
                changed = {p for c in rows[r] for p in c["plates"]} ^ {
                    p for c in rows[r - 1] for p in c["plates"]
                }
                y += Y_GAP + (26 if changed else 0)
            ys[r] = y
        for cell in cells.values():
            cell["y"] = ys[cell["rank"]]

        if direction == "LR":
            for cell in cells.values():
                cell["x"], cell["y"] = cell["y"], cell["x"]
        rects = separate(model, cells, rows, 1 if direction == "LR" else 0)
        pos = {c["key"]: (c["x"], c["y"]) for c in cells.values()}
        bends = {e: [pos[k] for k in chain] for e, chain in routes.items()}

        box = [
            min(pos[n][0] for n in model.nodes) - NODE_R,
            min(pos[n][1] for n in model.nodes) - NODE_R,
            max(pos[n][0] for n in model.nodes) + NODE_R,
            max(pos[n][1] for n in model.nodes) + NODE_R,
        ]
        for x1, y1, x2, y2 in rects.values():
            box = [min(box[0], x1), min(box[1], y1), max(box[2], x2), max(box[3], y2)]
        return pos, bends, rects, box

    return layout, nested_plates, overlap


@app.cell
def _(NODE_R, node_label):
    SERIF = "Georgia, 'Times New Roman', 'DejaVu Serif', serif"


    INK = "#1f2328"


    SHADE = "#c8ced7"


    HOVER = "#b23b2e"


    MONO = "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"


    TIP_SIZE = 12.0


    TIP_LEAD = 16.0


    def xml_escape(text):
        return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


    def fit_size(head, sub, limit):
        """Largest font size (<= 17) whose label still fits inside the circle."""
        width = 0.6 * len(head) + 0.42 * len(sub)
        return max(8.5, min(17.0, limit / max(width, 0.6)))


    def svg_label(node, x, y, size=None, anchor="middle"):
        head, sub = node_label(node)
        size = size or fit_size(head, sub, 2 * (NODE_R - 4))
        body = xml_escape(head)
        if sub:
            body += (f'<tspan dy="{size * 0.3:.1f}" font-size="{size * 0.66:.1f}">'
                     f'{xml_escape(sub)}</tspan>')
        return (f'<text x="{x:.1f}" y="{y + size * 0.34:.1f}" font-size="{size:.1f}" '
                f'font-style="italic" text-anchor="{anchor}" fill="{INK}">{body}</text>')


    def trim_ends(points, r1, r2):
        """Shorten a polyline so it starts and ends on the node boundaries."""
        def step(a, b, dist):
            vx, vy = b[0] - a[0], b[1] - a[1]
            norm = (vx * vx + vy * vy) ** 0.5 or 1.0
            return (a[0] + vx / norm * dist, a[1] + vy / norm * dist)

        points = list(points)
        points[0] = step(points[0], points[1], r1)
        points[-1] = step(points[-1], points[-2], r2)
        return points

    return (
        HOVER,
        INK,
        MONO,
        SERIF,
        SHADE,
        TIP_LEAD,
        TIP_SIZE,
        svg_label,
        trim_ends,
        xml_escape,
    )


@app.cell
def _(
    HOVER,
    INK,
    MONO,
    NODE_R,
    SERIF,
    SHADE,
    TIP_LEAD,
    TIP_SIZE,
    layout,
    split_name,
    svg_label,
    trim_ends,
    xml_escape,
):
    def uid_for(model):
        """A stable id prefix, so several diagrams can share a page."""
        text = "|".join(sorted(model.nodes)) + "||" + "|".join(f"{s}>{d}" for s, d in model.edges)
        digest = 2166136261
        for ch in text:
            digest = ((digest ^ ord(ch)) * 16777619) & 0xFFFFFFFF
        return f"pm{digest:08x}"


    def relation_lines(model, node):
        """What hovering a node reveals: its statement, its parents, its plates."""
        head = node.name + (f"[{node.sub}]" if node.sub else "")
        lines = [f"{head} {'=' if node.kind == 'det' else '~'} {node.dist}"
                 if node.dist else head]
        parents = model.parents(node.name)
        lines.append("parents: " + (", ".join(parents) if parents else "none (root)"))
        if node.plates:
            lines.append("inside: " + " · ".join(
                f"{model.plates[p].var} = {model.plates[p].span}" for p in node.plates))
        lines.append({"obs": "observed", "det": "deterministic",
                      "fixed": "fixed hyperparameter"}.get(node.kind, "latent"))
        return lines


    def place_tip(box_w, box_h, x, y, canvas, avoid):
        """Put the card beside the node, on whichever side hides the least."""
        width, height = canvas
        corners = [
            (x + NODE_R + 12, y - box_h / 2),
            (x - NODE_R - 12 - box_w, y - box_h / 2),
            (x - box_w / 2, y + NODE_R + 14),
            (x - box_w / 2, y - NODE_R - 14 - box_h),
        ]
        best, best_cost = None, None
        for left, top in corners:
            left = min(max(left, 6), max(width - box_w - 6, 6))
            top = min(max(top, 6), max(height - box_h - 6, 6))
            card = (left, top, left + box_w, top + box_h)
            cost = sum(
                weight * max(0.0, min(card[2], b[2]) - max(card[0], b[0]))
                * max(0.0, min(card[3], b[3]) - max(card[1], b[1]))
                for b, weight in avoid
            )
            if best_cost is None or cost < best_cost:
                best, best_cost = (left, top), cost
        return best


    def tooltip_card(uid, name, lines, x, y, canvas, avoid):
        """A card pinned near the node, hidden until that node is hovered."""
        box_w = 7.3 * max(len(line) for line in lines) + 20
        box_h = TIP_LEAD * len(lines) + 13
        left, top = place_tip(box_w, box_h, x, y, canvas, avoid)
        rows = "".join(
            f'<text x="{left + 10:.1f}" y="{top + 19 + i * TIP_LEAD:.1f}" '
            f'font-size="{TIP_SIZE}" fill="{INK if i == 0 else "#5b616b"}">'
            f"{xml_escape(line)}</text>"
            for i, line in enumerate(lines)
        )
        return (f'<g id="{uid}-t-{name}" class="tip" font-family="{MONO}">'
                f'<rect x="{left:.1f}" y="{top:.1f}" width="{box_w:.1f}" '
                f'height="{box_h:.1f}" rx="6" fill="#ffffff" stroke="{HOVER}" '
                f'stroke-width="1.1"/>{rows}</g>')


    def hover_css(uid, model):
        """One rule per node: light up its incoming edges, parents and tooltip."""
        rules = [
            f"#{uid} .tip{{opacity:0;pointer-events:none}}",
            f"#{uid} .ring{{opacity:0;pointer-events:none;fill:none;"
            f"stroke:{HOVER};stroke-width:2.4}}",
            f"#{uid} .node{{cursor:default}}",
        ]
        for name in model.nodes:
            seen = f"#{uid}:has(#{uid}-n-{name}:hover)"
            lit = [f"{seen} #{uid}-t-{name}{{opacity:1}}",
                   f"{seen} #{uid}-r-{name}{{opacity:1}}"]
            parents = [p for p in model.parents(name) if p in model.nodes]
            if parents:
                lit.append(",".join(f"{seen} #{uid}-e-{p}-{name}" for p in parents)
                           + f"{{stroke:{HOVER};stroke-width:2.6;"
                             f"marker-end:url(#{uid}-lit)}}")
                lit.append(",".join(f"{seen} #{uid}-r-{p}" for p in parents)
                           + "{opacity:1}")
            rules.extend(lit)
        return "<style>" + "".join(rules) + "</style>"


    def to_svg(model, direction="TB", scale=1.0, title=""):
        """Render the model as a standalone, self-contained SVG document."""
        if not model.nodes:
            return ('<svg xmlns="http://www.w3.org/2000/svg" width="340" height="70">'
                    f'<text x="14" y="40" font-family="{SERIF}" font-size="15" '
                    'fill="#8b9096">nothing to draw yet</text></svg>')

        uid = uid_for(model)
        pos, bends, rects, box = layout(model, direction)
        margin = 24
        top = margin + (32 if title else 0)
        w, h = (box[2] - box[0]) + 2 * margin, (box[3] - box[1]) + margin + top
        dx, dy = margin - box[0], top - box[1]
        at = lambda p: (p[0] + dx, p[1] + dy)  # noqa: E731
        out = []

        if title:
            out.append(f'<text x="{w / 2:.1f}" y="32" font-size="18" text-anchor="middle" '
                       f'fill="{INK}">{xml_escape(title)}</text>')

        for pid, rect in sorted(rects.items(), key=lambda kv: model.plates[kv[0]].depth):
            x1, y1 = at(rect[:2])
            x2, y2 = at(rect[2:])
            plate = model.plates[pid]
            out.append(f'<rect x="{x1:.1f}" y="{y1:.1f}" width="{x2 - x1:.1f}" '
                       f'height="{y2 - y1:.1f}" rx="7" fill="none" stroke="#7c828d" '
                       f'stroke-width="1.2"/>')
            out.append(f'<text x="{x2 - 9:.1f}" y="{y2 - 9:.1f}" font-size="13" '
                       f'font-style="italic" text-anchor="end" fill="#5b616b">'
                       f'{xml_escape(split_name(plate.var)[0])} = {xml_escape(plate.span)}</text>')

        for src, dst in model.edges:
            if src not in pos or dst not in pos:
                continue
            points = [at(pos[src])] + [at(p) for p in bends.get((src, dst), [])] + [at(pos[dst])]
            r1 = 6 if model.nodes[src].kind == "fixed" else NODE_R
            r2 = (6 if model.nodes[dst].kind == "fixed" else NODE_R) + 8
            path = " ".join(f"{x:.1f},{y:.1f}" for x, y in trim_ends(points, r1, r2))
            out.append(f'<polyline id="{uid}-e-{src}-{dst}" points="{path}" fill="none" '
                       f'stroke="{INK}" stroke-width="1.35" stroke-linejoin="round" '
                       f'marker-end="url(#{uid}-arrow)"/>')

        def circle_box(name, pad=0):
            cx, cy = at(pos[name])
            r = (6 if model.nodes[name].kind == "fixed" else NODE_R) + pad
            return (cx - r, cy - r, cx + r, cy + r)

        tips, rings = [], []
        for name, node in model.nodes.items():
            x, y = at(pos[name])
            radius = 6 if node.kind == "fixed" else NODE_R
            if node.kind == "fixed":
                body = (f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.5" fill="{INK}"/>'
                        + svg_label(node, x, y - 15, size=14))
            else:
                dash = ' stroke-dasharray="5 3"' if node.kind == "det" else ""
                body = (f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{NODE_R}" '
                        f'fill="{SHADE if node.kind == "obs" else "#ffffff"}" '
                        f'stroke="{INK}" stroke-width="1.4"{dash}/>' + svg_label(node, x, y))
            out.append(f'<g id="{uid}-n-{name}" class="node">{body}</g>')
            rings.append(f'<circle id="{uid}-r-{name}" class="ring" cx="{x:.1f}" '
                         f'cy="{y:.1f}" r="{radius + 5}"/>')
            kin = set(model.parents(name)) | {name}
            avoid = [(circle_box(other, 6), 5.0 if other in kin else 1.0)
                     for other in model.nodes]
            for parent in model.parents(name):          # keep the lit edges visible
                if parent in pos:
                    route = [at(pos[parent])] + [at(b) for b in bends.get((parent, name), [])]
                    route.append((x, y))
                    avoid.append(((min(p[0] for p in route) - 4, min(p[1] for p in route) - 4,
                                   max(p[0] for p in route) + 4, max(p[1] for p in route) + 4),
                                  2.0))
            tips.append(
                tooltip_card(uid, name, relation_lines(model, node), x, y, (w, h), avoid)
            )

        def marker(name, colour):
            return (f'<marker id="{name}" viewBox="0 0 10 10" refX="9" refY="5" '
                    'markerWidth="6.5" markerHeight="6.5" orient="auto-start-reverse">'
                    f'<path d="M 0 0 L 10 5 L 0 10 z" fill="{colour}"/></marker>')

        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" id="{uid}" '
            f'viewBox="0 0 {w:.0f} {h:.0f}" width="{w * scale:.0f}" '
            f'height="{h * scale:.0f}" font-family="{SERIF}">'
            f'<defs>{marker(uid + "-arrow", INK)}{marker(uid + "-lit", HOVER)}</defs>'
            + hover_css(uid, model)
            + '<rect width="100%" height="100%" fill="#ffffff" rx="8"/>'
            + "".join(out + rings + tips) + "</svg>"
        )

    return (to_svg,)


@app.cell
def _(layout, node_label, tex_label):
    def to_dot(model):
        """Graphviz DOT — plates become `cluster_` subgraphs."""
        lines = ["digraph plate {", "  rankdir=TB;", '  node [fontname="serif"];']

        def declare(node, indent):
            head, sub = node_label(node)
            label = head + (f"_{sub}" if sub else "")
            if node.kind == "fixed":
                return f'{indent}{node.name} [shape=point, width=0.12, xlabel="{label}"];'
            style = {"obs": 'shape=circle, style=filled, fillcolor="#c8ced7"',
                     "det": "shape=circle, style=dashed"}.get(node.kind, "shape=circle")
            return f'{indent}{node.name} [label="{label}", {style}];'

        def emit(parent, indent):
            for pid, plate in model.plates.items():
                if plate.parent != parent:
                    continue
                lines.append(f"{indent}subgraph cluster_{pid} {{")
                lines.append(f'{indent}  label="{plate.var} = {plate.span}"; '
                             "labelloc=b; labeljust=r; style=rounded; color=\"#7c828d\";")
                for node in model.nodes.values():
                    if node.plates and node.plates[-1] == pid:
                        lines.append(declare(node, indent + "  "))
                emit(pid, indent + "  ")
                lines.append(f"{indent}}}")

        for node in model.nodes.values():
            if not node.plates:
                lines.append(declare(node, "  "))
        emit(None, "  ")
        lines += [f"  {s} -> {d};" for s, d in model.edges]
        return "\n".join(lines + ["}"])


    def to_tikz(model):
        """TikZ for the `bayesnet` package, positioned from this layout."""
        pos, _bends, _rects, box = layout(model)
        lines = ["% \\usepackage{tikz}", "% \\usetikzlibrary{bayesnet}",
                 "\\begin{tikzpicture}"]
        for name, node in model.nodes.items():
            style = {"obs": "obs", "det": "det", "fixed": "const"}.get(node.kind, "latent")
            x, y = pos[name][0] / 62.0, -pos[name][1] / 62.0
            lines.append(f"  \\node[{style}] ({name}) at ({x:.2f},{y:.2f}) "
                         f"{{${tex_label(node.name, node.sub)}$}};")
        lines += [f"  \\edge {{{s}}} {{{d}}};" for s, d in model.edges]
        for pid, plate in model.plates.items():
            members = " ".join(f"({n.name})" for n in model.nodes.values() if pid in n.plates)
            lines.append(f"  \\plate {{{pid}}} {{{members}}} "
                         f"{{${tex_label(plate.var)} = {plate.span}$}};")
        return "\n".join(lines + ["\\end{tikzpicture}"])


    def to_daft(model):
        """Python that rebuilds the figure with daft (matplotlib)."""
        pos, _bends, rects, box = layout(model)
        unit, base = 74.0, box[3]
        lines = ["import daft", "", "pgm = daft.PGM()"]
        for name, node in model.nodes.items():
            flag = {"obs": ", observed=True", "fixed": ", fixed=True"}.get(node.kind, "")
            lines.append(f'pgm.add_node("{name}", r"${tex_label(node.name, node.sub)}$", '
                         f"{pos[name][0] / unit:.2f}, {(base - pos[name][1]) / unit:.2f}{flag})")
        lines += [f'pgm.add_edge("{s}", "{d}")' for s, d in model.edges]
        for pid, plate in model.plates.items():
            x1, y1, x2, y2 = rects[pid]
            lines.append(f"pgm.add_plate([{x1 / unit:.2f}, {(base - y2) / unit:.2f}, "
                         f"{(x2 - x1) / unit:.2f}, {(y2 - y1) / unit:.2f}], "
                         f'label=r"${tex_label(plate.var)} = {plate.span}$", shift=-0.1)')
        return "\n".join(lines + ["pgm.render()", 'pgm.savefig("model.png", dpi=200)'])


    def to_mermaid(model):
        """Mermaid flowchart — plates become subgraphs."""
        lines = ["flowchart TB"]
        shapes = {"obs": ("([", "])"), "det": ("{{", "}}"), "fixed": ("[", "]")}

        def declare(node, indent):
            head, sub = node_label(node)
            open_, close = shapes.get(node.kind, ("((", "))"))
            return f'{indent}{node.name}{open_}"{head}{"_" + sub if sub else ""}"{close}'

        def emit(parent, indent):
            for pid, plate in model.plates.items():
                if plate.parent != parent:
                    continue
                lines.append(f'{indent}subgraph {pid}["{plate.var} = {plate.span}"]')
                for node in model.nodes.values():
                    if node.plates and node.plates[-1] == pid:
                        lines.append(declare(node, indent + "  "))
                emit(pid, indent + "  ")
                lines.append(f"{indent}end")

        for node in model.nodes.values():
            if not node.plates:
                lines.append(declare(node, "  "))
        emit(None, "  ")
        return "\n".join(lines + [f"  {s} --> {d}" for s, d in model.edges])

    return to_daft, to_dot, to_mermaid, to_tikz


@app.cell
def _(rank_nodes, re):
    PYMC_DATA = "_data"     # suffix for the arrays an observed node is fitted to


    DEMO_SUPPORT = 3        # assumed length of a vector draw nothing pins down


    # Distributions whose draw is itself a vector: the plates cannot tell you
    # how long that vector is, so the generated dims are one short.
    MULTIVARIATE = {"dirichlet", "mvnormal", "multivariatenormal", "multinomial"}


    DISCRETE = {"categorical", "bernoulli", "binomial", "poisson", "multinomial",
                "geometric", "negbinomial", "negativebinomial", "betabinomial"}


    # DSL distribution -> (Stan name, arguments prepended, Stan type of a draw)
    STAN_DIST = {
        "normal": ("normal", "", "real"),
        "halfnormal": ("normal", "0, ", "real<lower=0>"),
        "cauchy": ("cauchy", "", "real"),
        "halfcauchy": ("cauchy", "0, ", "real<lower=0>"),
        "studentt": ("student_t", "", "real"),
        "uniform": ("uniform", "", "real"),
        "lognormal": ("lognormal", "", "real<lower=0>"),
        "exponential": ("exponential", "", "real<lower=0>"),
        "gamma": ("gamma", "", "real<lower=0>"),
        "invgamma": ("inv_gamma", "", "real<lower=0>"),
        "beta": ("beta", "", "real<lower=0, upper=1>"),
        "dirichlet": ("dirichlet", "", "simplex"),
        "categorical": ("categorical", "", "int<lower=1>"),
        "bernoulli": ("bernoulli", "", "int<lower=0, upper=1>"),
        "binomial": ("binomial", "", "int<lower=0>"),
        "poisson": ("poisson", "", "int<lower=0>"),
        "multinomial": ("multinomial", "", "int<lower=0>"),
        "mvnormal": ("multi_normal", "", "vector"),
    }


    def split_call(text):
        """`Normal(0, 1)` -> ("Normal", "0, 1"); anything else -> (text, None)."""
        match = re.match(r"^\s*([A-Za-z_]\w*)\s*\((.*)\)\s*$", text, re.S)
        return (match.group(1), match.group(2)) if match else (text.strip(), None)


    def scan_refs(text):
        """Walk a right-hand side, yielding (literal, name, index) pieces."""
        i = 0
        while i < len(text):
            match = re.match(r"[A-Za-z_]\w*", text[i:])
            if not match:
                yield text[i], None, None
                i += 1
                continue
            name = match.group(0)
            i += len(name)
            index = None
            if i < len(text) and text[i] == "[":
                depth, j = 0, i
                while j < len(text):
                    depth += 1 if text[j] == "[" else -1 if text[j] == "]" else 0
                    if depth == 0:
                        break
                    j += 1
                index, i = text[i + 1: j], j + 1
            yield None, name, index


    def dims_of(model, node):
        return tuple(model.plates[p].var for p in node.plates)


    def plate_size(plate):
        """The Stan/PyMC name for how far a plate runs."""
        tail = plate.span.split("..")[-1].strip()
        return tail if re.fullmatch(r"[A-Za-z_]\w*", tail) else f"n_{plate.var}"


    def in_dependency_order(model):
        """Nodes in dependency order — parents before children."""
        rank = rank_nodes(model)
        return sorted(model.nodes.values(), key=lambda n: (rank[n.name], n.order))


    def plate_tree(model, pid, keep):
        """(order, kind, payload) items directly inside a plate, in source order."""
        items = [(n.order, "node", n) for n in model.nodes.values()
                 if (n.plates[-1] if n.plates else None) == pid and keep(n)]
        for cid, plate in model.plates.items():
            if plate.parent != pid:
                continue
            inside = [n.order for n in model.nodes.values() if cid in n.plates and keep(n)]
            if inside:
                items.append((min(inside), "plate", cid))
        return [item[1:] for item in sorted(items)]

    return (
        DEMO_SUPPORT,
        DISCRETE,
        MULTIVARIATE,
        PYMC_DATA,
        STAN_DIST,
        dims_of,
        in_dependency_order,
        plate_size,
        plate_tree,
        scan_refs,
        split_call,
    )


@app.cell
def _(
    DEMO_SUPPORT,
    MULTIVARIATE,
    PYMC_DATA,
    dims_of,
    in_dependency_order,
    plate_size,
    re,
    scan_refs,
    split_call,
):
    def is_multivariate(node):
        return split_call(node.dist or "")[0].lower() in MULTIVARIATE


    def support_sizes(model):
        """How long each multivariate node's draw is, guessed from how it is used.

        A `Categorical` over some simplex draws a label; if that label indexes a
        node, the simplex is exactly as long as that node's first plate. Returns
        the size expressions, and the names where nothing pinned it down.
        """
        guess = {n.name: str(DEMO_SUPPORT) for n in model.nodes.values() if is_multivariate(n)}
        guessed = set(guess)
        for chooser in model.nodes.values():
            head, args = split_call(chooser.dist or "")
            if head.lower() != "categorical" or not args:
                continue
            picks = [n for _lit, n, _idx in scan_refs(args) if n in guess]
            for other in model.nodes.values():
                for _lit, name, index in scan_refs(other.dist or ""):
                    if not index or chooser.name not in index or name not in model.nodes:
                        continue
                    target = model.nodes[name]
                    if target.plates:
                        for pick in picks:
                            guess[pick] = plate_size(model.plates[target.plates[0]])
                            guessed.discard(pick)
        return guess, guessed


    def demo_sizes(model):
        """Concrete plate lengths for the runnable version.

        A plate that something indexes into (mixture components, topics) stays
        small; the plates that only count observations get more room.
        """
        linked = set()
        for node in model.nodes.values():
            for _literal, name, index in scan_refs(node.dist or ""):
                if not index or name not in model.nodes:
                    continue
                labels = [n for _l, n, _i in scan_refs(index) if n in model.nodes]
                target = model.nodes[name]
                if labels and target.plates:
                    linked.add(target.plates[0])
        return {pid: "3" if pid in linked else ("8" if plate.depth == 0 else "4")
                for pid, plate in model.plates.items()}


    def shape_note(model, node):
        dims = dims_of(model, node)
        return f", over {' x '.join(dims)}" if dims else ""


    def demo_value(model, node, support):
        """A concrete stand-in for a given: the shape matters, the value rarely does."""
        for child in model.children(node.name):
            if child in support:
                return f"np.ones({support[child]})"
        shape = ", ".join(plate_size(model.plates[p]) for p in node.plates)
        return f"rng.normal(size=({shape},))" if shape else "1.0"


    def pymc_ref(model, target, name, index):
        """How `name` (optionally indexed) should be written inside `target`."""
        node = model.nodes[name]
        if index:
            picks = [n for _lit, n, _idx in scan_refs(index) if n in model.nodes]
            if picks:      # mixture-style indexing: phi[z[d,n]] -> phi[z]
                inner = ", ".join(pymc_ref(model, target, p, None) for p in picks)
                return f"{name}[{inner}]"
        mine, theirs = target.plates, node.plates
        if not theirs or theirs == mine:
            return name
        if mine[: len(theirs)] == theirs:      # outer plate: broadcast into place
            axes = [":"] * len(theirs) + ["None"] * (len(mine) - len(theirs))
            if is_multivariate(node):
                axes.append(":")               # keep the support axis last
            return f"{name}[{', '.join(axes)}]"
        return f"{name}[{model.plates[theirs[-1]].var}_idx]"


    def pymc_expr(model, target, text):
        """Rewrite a right-hand side into vectorised PyMC."""
        out = []
        for literal, name, index in scan_refs(text):
            if literal is not None:
                out.append(literal)
            elif name in model.nodes:
                out.append(pymc_ref(model, target, name, index))
            else:
                out.append(name + (f"[{index}]" if index is not None else ""))
        return "".join(out)


    def pymc_dims(model, node):
        """`dims=` for a node: its plates, plus a support axis if it draws a vector."""
        dims = list(dims_of(model, node))
        if is_multivariate(node):
            dims.append(f"{node.name}_dim")
        if not dims:
            return ""
        if len(dims) == 1:
            return f', dims="{dims[0]}"'
        return ", dims=(" + ", ".join(f'"{d}"' for d in dims) + ")"


    def pymc_body(model, indent="        "):
        """One PyMC statement per node, parents first."""
        lines, todo = [], []
        for node in in_dependency_order(model):
            if node.kind == "fixed":
                continue
            dims = pymc_dims(model, node)
            if node.kind == "det":
                expr = pymc_expr(model, node, node.dist or "...")
                lines.append(f'{indent}{node.name} = pm.Deterministic("{node.name}", {expr}{dims})')
                continue
            head, args = split_call(node.dist or "Flat()")
            args = pymc_expr(model, node, args) if args else ""
            call = f'"{node.name}"' + (f", {args}" if args else "")
            if node.kind == "obs":
                call += f', observed=None if data is None else data["{node.name}"]'
            lines.append(f"{indent}{node.name} = pm.{head}({call}{dims})")
            if not args:
                todo.append(f"{node.name} has no distribution — pm.Flat() stands in")
        for index in dict.fromkeys(re.findall(r"\b(\w+)_idx\b", "\n".join(lines))):
            todo.append(f"{index}_idx — an integer array saying which {index} each row belongs to")
        return lines, todo


    def to_pymc(model, runnable=False):
        """PyMC for the model.

        The scaffold leaves every size and every array the diagram never knew about
        as `...`. The runnable version fills those in, simulates one dataset from
        the model itself — so the true parameters are known — and fits it back.
        """
        if not model.nodes:
            return "# nothing to build yet"

        silent = [n.name for n in model.nodes.values()
                  if n.kind in ("latent", "obs") and not n.dist]
        if runnable and silent:
            return (
                "# Nothing to run: " + ", ".join(silent) + " have no distribution.\n"
                "# Arrows alone say who depends on whom — enough to draw the diagram,\n"
                "# not enough to simulate or fit. Give the nodes distributions, e.g.\n"
                "# `wet ~ Bernoulli(p)`, and this becomes a program you can run.\n\n"
                + to_pymc(model, runnable=False)
            )

        support, guessed = support_sizes(model)
        body, todo = pymc_body(model)
        observed = [n.name for n in model.nodes.values() if n.kind == "obs"]
        todo += [f"{name}_dim is a guess ({support[name]}) — the plates never said how "
                 f"long a {name} draw is" for name in guessed]
        for node in model.nodes.values():
            lengths = {support[c] for c in model.children(node.name) if c in support}
            if len(lengths) > 1:
                todo.append(f"{node.name} feeds draws of different lengths "
                            f"({', '.join(sorted(lengths))}) — one value cannot serve both")

        coords = [f'    "{plate.var}": range({plate_size(plate)}),'
                  for plate in model.plates.values()]
        coords += [f'    "{name}_dim": range({support[name]}),' for name in support]

        if runnable:
            sizes = demo_sizes(model)
            givens = [f"{n.name} = {demo_value(model, n, support)}   # given"
                      for n in model.nodes.values() if n.kind == "fixed"]
            if any("rng." in given for given in givens):
                givens.insert(0, "rng = np.random.default_rng(seed)")
        else:
            givens = [f"{plate_size(p)} = ...  # how far {p.var} runs ({p.span})"
                      for p in model.plates.values()]
            givens += [f"{n.name} = ...  # given" + shape_note(model, n)
                       for n in model.nodes.values() if n.kind == "fixed"]
            givens += [f"{n.name}{PYMC_DATA} = ...  # observed" + shape_note(model, n)
                       for n in model.nodes.values() if n.kind == "obs"]
        givens = list(dict.fromkeys(givens))

        if not runnable:
            lines = [
                "import pymc as pm",
                "",
                "# Generated from a plate model — the structure is exact; every `...`",
                "# is data or a size the diagram never knew about. Pasted into a marimo",
                "# cell, the names below join the notebook's graph, so keep them clear of",
                "# names it already owns.",
            ]
            lines += [f"# TODO: {note}" for note in todo]
            lines += [""] + givens + [""]
            if coords:
                lines += ["coords = {"] + coords + ["}", ""]
            lines += [
                "def build(data=None):",
                '    """The plate model. Pass data to condition on it, omit it to simulate."""',
                "    with pm.Model(" + ("coords=coords" if coords else "") + ") as built:",
            ]
            lines += body or ["        pass"]
            lines += [
                "    return built",
                "",
                "_model = build({" + ", ".join(f'"{n}": {n}{PYMC_DATA}' for n in observed) + "})",
            ]
            return "\n".join(lines)

        # Runnable: everything lives inside one function, so the whole program can be
        # pasted into a marimo cell without a single name joining the notebook's graph.
        plates = [(plate_size(p), sizes[pid]) for pid, p in model.plates.items()]
        signature = ", ".join([f"{name}={value}" for name, value in dict(plates).items()]
                              + ["seed=0"])
        lines = [f"def simulate_and_fit({signature}):",
                 '    """Simulate one dataset from this model, then fit it back.',
                 "",
                 "    The plates fix the structure; the sizes above are stand-ins. Every",
                 "    name is local, so this drops into a marimo cell as it stands.",
                 '    """']
        lines += [f"    # TODO: {note}" for note in todo]
        lines += ["    import numpy as np", "    import pymc as pm", ""]
        lines += ["    " + given for given in givens]
        if coords:
            lines += ["", "    coords = {"] + ["    " + c for c in coords] + ["    }"]
        lines += [
            "",
            "    def build(data=None):",
            "        with pm.Model(" + ("coords=coords" if coords else "") + ") as built:",
        ]
        lines += ["    " + statement for statement in body] or ["            pass"]
        lines += [
            "        return built",
            "",
            "    # One dataset drawn from the model itself, so the truth is known.",
            "    prior = build()",
            "    truth = dict(zip([v.name for v in prior.free_RVs],",
            "                     pm.draw(prior.free_RVs, random_seed=seed)))",
            "    data = {" + ", ".join(f'"{n}": truth["{n}"]' for n in observed) + "}",
            "",
            "    fitted = build(data)          # the same model, conditioned on it",
            "    idata = pm.sample(",
            "        draws=500, tune=500, chains=2, cores=1,",
            "        random_seed=seed, progressbar=False, model=fitted,",
            "    )",
            '    return {"idata": idata, "truth": truth, "data": data, "model": fitted}',
            "",
            "",
            "results = simulate_and_fit()",
        ]
        return "\n".join(lines)

    return (to_pymc,)


@app.cell
def _(
    DISCRETE,
    IDENT_RE,
    STAN_DIST,
    dims_of,
    plate_size,
    plate_tree,
    scan_refs,
    split_call,
):
    def stan_ref(model, name, index):
        """Index a reference the way the plates say it should be indexed."""
        node = model.nodes[name]
        if index and any(n in model.nodes for n in IDENT_RE.findall(index)):
            return f"{name}[{stan_expr(model, index)}]"     # e.g. phi[z[d, n]]
        dims = dims_of(model, node)
        return f"{name}[{', '.join(dims)}]" if dims else name


    def stan_expr(model, text):
        """Rewrite a right-hand side so every reference carries its plate indices."""
        out = []
        for literal, name, index in scan_refs(text):
            if literal is not None:
                out.append(literal)
            elif name in model.nodes:
                out.append(stan_ref(model, name, index))
            else:
                out.append(name + (f"[{index}]" if index is not None else ""))
        return "".join(out)


    def stan_dist(model, node):
        """(call, type) for a node's distribution, falling back to a lower-cased guess."""
        head, args = split_call(node.dist or "")
        key = head.lower().replace("_", "")
        name, prefix, kind = STAN_DIST.get(key, (head.lower(), "", "real"))
        if args is None:
            return "", kind
        return f"{name}({prefix}{stan_expr(model, args)})", kind


    def stan_decl(model, node, kind=None):
        """`array[D, N] int<lower=1> w;` — a declaration with the plate shape."""
        kind = kind or stan_dist(model, node)[1]
        if kind == "simplex":
            kind = f"simplex[dim_{node.name}]"
        elif kind == "vector":
            kind = f"vector[dim_{node.name}]"
        shape = ", ".join(plate_size(model.plates[p]) for p in node.plates)
        return f"array[{shape}] {kind} {node.name};" if shape else f"{kind} {node.name};"


    def fixed_type(model, node):
        """A given feeding a simplex- or vector-valued draw is itself a vector."""
        for child in model.children(node.name):
            kind = stan_dist(model, model.nodes[child])[1]
            if kind in ("simplex", "vector"):
                return f"vector[dim_{child}]"
        return "real"


    def stan_body(model, keep, pid=None, indent="  "):
        """The sampling statements, wrapped in the loops the plates describe."""
        lines = []
        for kind, payload in plate_tree(model, pid, keep):
            if kind == "plate":
                plate = model.plates[payload]
                lines.append(f"{indent}for ({plate.var} in 1:{plate_size(plate)}) {{")
                lines += stan_body(model, keep, payload, indent + "  ")
                lines.append(f"{indent}}}")
                continue
            node = payload
            dims = dims_of(model, node)
            head = f"{node.name}[{', '.join(dims)}]" if dims else node.name
            if node.kind == "det":
                lines.append(f"{indent}{head} = {stan_expr(model, node.dist)};")
                continue
            call, _kind = stan_dist(model, node)
            flag = "   // discrete latent — marginalise this out" if (
                node.kind == "latent" and split_call(node.dist or "")[0].lower() in DISCRETE
            ) else ""
            lines.append(f"{indent}{head} ~ {call};{flag}" if call
                         else f"{indent}// {head}: no distribution given")
        return lines


    def to_stan(model):
        """A Stan program: the plates become loops, so the structure is faithful."""
        if not model.nodes:
            return "// nothing to build yet"

        discrete = [n for n in model.nodes.values() if n.kind == "latent"
                    and split_call(n.dist or "")[0].lower() in DISCRETE]
        notes = ["// Generated from a plate model — sizes and data are yours to fill in,",
                 "// and array shapes assume rectangular plates."]
        if discrete:
            notes.append("// Stan has no discrete parameters: marginalise "
                         + ", ".join(n.name for n in discrete)
                         + " out (see the mixture chapter of the user's guide)")

        data = [f"  int<lower=1> {plate_size(p)};" for p in model.plates.values()]
        data += [f"  int<lower=1> dim_{n.name};   // size of the simplex {n.name} lives on"
                 for n in model.nodes.values() if stan_dist(model, n)[1] == "simplex"]
        data += [f"  {stan_decl(model, n, fixed_type(model, n))}   // fixed"
                 for n in model.nodes.values() if n.kind == "fixed"]
        data += [f"  {stan_decl(model, n)}   // observed"
                 for n in model.nodes.values() if n.kind == "obs"]

        params, latent = [], [n for n in model.nodes.values() if n.kind == "latent"]
        for node in latent:
            line = f"  {stan_decl(model, node)}"
            params.append(f"  // {line.strip()}   // discrete — marginalise"
                          if node in discrete else line)

        blocks = ["\n".join(notes), ""]
        blocks.append("data {\n" + "\n".join(dict.fromkeys(data)) + "\n}")
        if params:
            blocks.append("parameters {\n" + "\n".join(params) + "\n}")
        dets = [n for n in model.nodes.values() if n.kind == "det"]
        if dets:
            blocks.append("transformed parameters {\n"
                          + "\n".join(f"  {stan_decl(model, n, 'real')}" for n in dets)
                          + "\n" + "\n".join(stan_body(model, lambda n: n.kind == "det"))
                          + "\n}")
        blocks.append("model {\n"
                      + "\n".join(stan_body(model, lambda n: n.kind in ("latent", "obs")))
                      + "\n}")
        return "\n".join(blocks)

    return (to_stan,)


@app.cell
def _():
    EXAMPLES = {
        "Latent Dirichlet allocation": """\
    # Latent Dirichlet allocation (Blei, Ng & Jordan)
    const alpha, beta

    for k in 1..K:
        phi[k] ~ Dirichlet(beta)

    for d in 1..D:
        theta[d] ~ Dirichlet(alpha)
        for n in 1..N_d:
            z[d,n] ~ Categorical(theta[d])
            w[d,n] ~ Categorical(phi[z[d,n]]) obs
    """,
        "Gaussian mixture": """\
    # Finite Gaussian mixture
    const alpha
    pi ~ Dirichlet(alpha)

    for k in 1..K:
        mu[k] ~ Normal(0, 10)
        sigma[k] ~ HalfNormal(1)

    for i in 1..N:
        z[i] ~ Categorical(pi)
        x[i] ~ Normal(mu[z[i]], sigma[z[i]]) obs
    """,
        "Hierarchical regression": """\
    # Partially pooled ("varying intercept") regression
    mu_a ~ Normal(0, 5)
    sigma_a ~ HalfNormal(2)
    sigma_y ~ HalfNormal(2)
    b ~ Normal(0, 1)

    for j in 1..J:
        a[j] ~ Normal(mu_a, sigma_a)
        for i in 1..n_j:
            x[i] ~ const
            yhat[i] = a[j] + b * x[i]
            y[i] ~ Normal(yhat[i], sigma_y) obs
    """,
        "Hidden Markov model": """\
    # Hidden Markov model, unrolled three steps
    # (a plate can show repetition, but not the chain itself)
    const alpha

    for k in 1..K:
        A[k] ~ Dirichlet(alpha)          # transition row for state k
        B[k] ~ Dirichlet(alpha)          # what state k emits

    pi ~ Dirichlet(alpha)
    z_1 ~ Categorical(pi)
    z_2 ~ Categorical(A[z_1])
    z_3 ~ Categorical(A[z_2])
    y_1 ~ Categorical(B[z_1]) obs
    y_2 ~ Categorical(B[z_2]) obs
    y_3 ~ Categorical(B[z_3]) obs
    """,
        "Sprinkler (plain edges)": """\
    # No distributions, just structure
    cloudy -> sprinkler
    cloudy -> rain
    sprinkler -> wet
    rain -> wet
    wet obs
    """,
        "Empty": "",
    }
    return (EXAMPLES,)


@app.cell
def _(
    EXAMPLES,
    NODE_R,
    ast,
    layout,
    mo,
    nested_plates,
    overlap,
    parse,
    to_daft,
    to_dot,
    to_mermaid,
    to_pymc,
    to_stan,
    to_svg,
    to_tikz,
):
    def _run_checks():
        results = []

        def check(name):
            def wrap(fn):
                try:
                    fn()
                    results.append((name, "✅", ""))
                except Exception as exc:  # noqa: BLE001
                    results.append((name, "❌", f"{type(exc).__name__}: {exc}"))
                return fn
            return wrap

        lda = parse(EXAMPLES["Latent Dirichlet allocation"])

        @check("LDA parses to 6 nodes, 3 plates, 5 edges")
        def _():
            assert (len(lda.nodes), len(lda.plates), len(lda.edges)) == (6, 3, 5)
            assert not lda.warnings

        @check("`obs` marks the word node as observed")
        def _():
            assert lda.nodes["w"].kind == "obs"
            assert lda.nodes["alpha"].kind == "fixed"

        @check("edges come from declared names only")
        def _():
            assert ("theta", "z") in lda.edges and ("z", "w") in lda.edges
            assert not any(s == "Categorical" for s, _ in lda.edges)

        @check("nesting follows indentation")
        def _():
            assert lda.nodes["z"].plates == ("p002", "p003")
            assert lda.plates["p003"].parent == "p002"

        @check("deterministic and accented names")
        def _():
            hier = parse(EXAMPLES["Hierarchical regression"])
            assert hier.nodes["yhat"].kind == "det"
            assert to_svg(hier).count("y\u0302") == 1  # y + combining hat

        @check("no plate is drawn over a foreign plate or node")
        def _():
            for name, text in EXAMPLES.items():
                model = parse(text)
                if not model.nodes:
                    continue
                for way in ("TB", "LR"):
                    pos, _bends, rects, _box = layout(model, way)
                    for a in rects:
                        for b in rects:
                            if a < b and not nested_plates(model, a, b):
                                assert min(overlap(rects[a], rects[b])) <= 0, \
                                    f"{name}/{way}: plates {a} and {b} overlap"
                        for node in model.nodes.values():
                            if a in node.plates:
                                continue
                            x, y = pos[node.name]
                            box = (x - NODE_R, y - NODE_R, x + NODE_R, y + NODE_R)
                            assert min(overlap(rects[a], box)) <= 0, \
                                f"{name}/{way}: {node.name} sits in plate {a}"

        @check("every example survives every exporter")
        def _():
            for text in EXAMPLES.values():
                model = parse(text)
                for export in (to_svg, to_dot, to_tikz, to_daft, to_mermaid,
                               to_pymc, to_stan):
                    if model.nodes or export is to_svg:
                        assert export(model).strip()

        @check("hovering is wired up for every node and edge")
        def _():
            svg = to_svg(lda)
            assert svg.count('class="tip"') == len(lda.nodes)
            assert svg.count('class="ring"') == len(lda.nodes)
            for s, d in lda.edges:
                assert f"-e-{s}-{d}" in svg
            for name in lda.nodes:
                assert f"-n-{name}:hover)" in svg
            assert "Categorical(phi[z[d,n]])" in svg   # the statement, verbatim

        @check("generated PyMC is valid Python, and indexes the way it should")
        def _():
            for text in EXAMPLES.values():
                compile(to_pymc(parse(text)), "<pymc>", "exec")
            assert 'pm.Categorical("w", phi[z]' in to_pymc(lda)   # mixture indexing
            hier = to_pymc(parse(EXAMPLES["Hierarchical regression"]))
            assert "a[:, None]" in hier          # outer plate broadcast into inner
            assert 'data["y"]' in hier
            assert "_model = build(" in hier     # cell-local when pasted into marimo

        @check("the runnable version simulates its own data, with no gaps left")
        def _():
            for name, text in EXAMPLES.items():
                model = parse(text)
                code = to_pymc(model, runnable=True)
                compile(code, "<runnable>", "exec")
                if not model.nodes or code.startswith("# Nothing to run"):
                    continue
                assert "pm.draw(" in code and "pm.sample(" in code
                assert " = ..." not in code, f"{name} still has a placeholder"
                # every name stays inside the function, so it can be pasted into
                # a marimo cell without colliding with the notebook's own graph
                _top = ast.parse(code).body
                assert {type(_n).__name__ for _n in _top} == {"FunctionDef", "Assign"}
                assert [_t.id for _n in _top if isinstance(_n, ast.Assign)
                        for _t in _n.targets] == ["results"]
            sprinkler = to_pymc(parse(EXAMPLES["Sprinkler (plain edges)"]), runnable=True)
            assert sprinkler.startswith("# Nothing to run")   # arrows alone cannot fit

        @check("plate sizes and vector lengths line up in the runnable version")
        def _():
            code = to_pymc(lda, runnable=True)
            assert '"theta_dim": range(K)' in code   # a topic mix is K long
            assert 'pm.Dirichlet("theta", alpha, dims=("d", "theta_dim"))' in code
            assert "theta[:, None, :]" in code       # broadcast, support axis last
            assert "alpha = np.ones(K)" in code

        @check("generated Stan has the blocks, loops and warnings it needs")
        def _():
            stan = to_stan(lda)
            for block in ("data {", "parameters {", "model {"):
                assert block in stan
            assert stan.count("{") == stan.count("}")
            assert "for (d in 1:D) {" in stan and "for (n in 1:N_d) {" in stan
            assert "marginalise z out" in stan   # Stan has no discrete parameters
            hier = to_stan(parse(EXAMPLES["Hierarchical regression"]))
            assert "yhat[j, i] = a[j] + b * x[j, i];" in hier   # indexed by plate
            assert "array[J, n_j] real y;" in hier

        @check("junk lines warn instead of raising")
        def _():
            broken = parse("a ~ Normal(0, 1)\n)(*&^\nb ~ Normal(a, 1)")
            assert len(broken.warnings) == 1
            assert ("a", "b") in broken.edges

        @check("a cycle is reported, not hung on")
        def _():
            looped = parse("a -> b\nb -> a")
            assert to_svg(looped)          # the warning is raised while laying out
            assert looped.warnings

        return results

    _rows = _run_checks()
    mo.md(
        "### Self-checks\n\n| | | |\n|:--|:--|:--|\n"
        + "\n".join(f"| {mark} | {name} | {note} |" for name, mark, note in _rows)
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Is there already a markup language for plate models?

    Not one standard one. There are three families, and the exports above
    speak to all of them.

    **1. The model is the markup.** BUGS/JAGS, Stan, PyMC and NumPyro
    already describe exactly what a plate diagram draws — loops are plates,
    sampling statements are nodes, and the right-hand sides are the edges.
    That is why the syntax in this notebook is shaped like a model rather
    than like a drawing: it is close to the shortest text that still
    contains everything the picture needs. If you have a *live* model,
    `pm.model_to_graphviz(model)` (PyMC) and `pgmpy`'s `DAG`/`to_daft`
    helpers will draw one for you without any markup at all.

    **2. Figure libraries — positional, and the usual choice for papers.**
    [daft](https://docs.daft-pgm.org) is the Python/matplotlib standard;
    [`tikz-bayesnet`](https://github.com/jluttine/tikz-bayesnet) is the
    LaTeX one (`\node[latent]`, `\node[obs]`, `\plate`). Both make you
    place every node by hand, which is the tedious part — so the **daft**
    and **TikZ** tabs hand you a laid-out starting point to nudge.

    **3. General diagram markup.** Graphviz DOT gets plates from
    `subgraph cluster_*`, Mermaid and D2 from subgraphs/containers. They
    are everywhere and marimo renders Mermaid natively (`mo.mermaid`, the
    Mermaid tab above), but none of them know what an *observed* variable
    is, and you inherit whatever the layout engine feels like doing with
    clusters.

    ## The code tabs, and what they can't know

    The **PyMC** and **Stan** tabs write code, not a compiled model. The
    structure — who depends on whom, what repeats over what, which nodes
    are data — is exactly what you drew, and the distribution arguments are
    passed through as you typed them.

    PyMC comes two ways. As a **scaffold**, everything the diagram never
    knew is left as `...` for you to fill in. As **runnable**, those gaps
    get concrete stand-ins and the program simulates one dataset from the
    model itself — so every parameter has a known true value — then fits it
    back; **▶ Simulate and fit** runs exactly that and plots what came out
    against what went in. It needs `pymc` in the environment, which the
    notebook does not otherwise require.

    What a plate diagram cannot tell either target:

    - **sizes and data.** How far a plate runs, and the arrays an observed
      node is fitted to.
    - **the support of a multivariate node.** A Dirichlet inside a plate
      has a length the diagram never mentions. Where a `Categorical` draw
      over it indexes something, that length is recoverable and is used;
      otherwise it is a flagged guess.
    - **discrete latent variables.** Stan cannot sample them at all; the
      `z` in a mixture has to be marginalised out, and the program says
      which variables need it rather than quietly emitting something that
      won't fit.
    - **ragged plates.** `n = 1..N_d` becomes a rectangular array.

    Where the plates *do* settle the answer, they are used: Stan indexes by
    the plate path rather than by whatever subscript you typed, and PyMC
    gets `dims` from the plates, `a[:, None]` when an outer-plate variable
    feeds an inner one, and `phi[z]` for mixture-style indexing.

    ## What marimo brings

    marimo has no plate-model widget — its diagram support is `mo.mermaid`
    plus anything that renders to HTML, SVG or matplotlib, and
    [anywidget](https://docs.marimo.io/api/inputs/anywidget/) for custom
    interactive ones. The renderer here is deliberately plain: a few
    hundred lines of pure Python emitting SVG, so there is no Graphviz
    binary to install and it works in the WASM preview.

    Natural next steps, if this earns its keep:

    - drag-to-position on top of the automatic layout (anywidget), with the
      nudges saved back into the text;
    - going the other way: reading a *live* PyMC model back into the DSL,
      so the diagram cannot drift from the sampler;
    - `mo.ui.altair_chart`-style selection: click a node, see its Markov
      blanket highlighted.
    """)
    return


if __name__ == "__main__":
    app.run()
