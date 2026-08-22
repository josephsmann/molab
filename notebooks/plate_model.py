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
    import re
    from dataclasses import dataclass, field

    return dataclass, field, mo, re


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
    as you type, and the same model exports to TikZ, daft, Graphviz DOT and
    Mermaid.

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
def _(mo, model, svg, to_daft, to_dot, to_mermaid, to_tikz):
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
       the TikZ, daft, DOT and Mermaid exporters.
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

    return (parse,)


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
def _(NODE_R, layout, node_label, split_name):
    SERIF = "Georgia, 'Times New Roman', 'DejaVu Serif', serif"


    INK = "#1f2328"


    SHADE = "#c8ced7"


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


    def to_svg(model, direction="TB", scale=1.0, title=""):
        """Render the model as a standalone SVG document."""
        if not model.nodes:
            return ('<svg xmlns="http://www.w3.org/2000/svg" width="340" height="70">'
                    f'<text x="14" y="40" font-family="{SERIF}" font-size="15" '
                    'fill="#8b9096">nothing to draw yet</text></svg>')

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
            label = f"{split_name(plate.var)[0]} = {plate.span}"
            out.append(f'<rect x="{x1:.1f}" y="{y1:.1f}" width="{x2 - x1:.1f}" '
                       f'height="{y2 - y1:.1f}" rx="7" fill="none" stroke="#7c828d" '
                       f'stroke-width="1.2"/>')
            out.append(f'<text x="{x2 - 9:.1f}" y="{y2 - 9:.1f}" font-size="13" '
                       f'font-style="italic" text-anchor="end" fill="#5b616b">'
                       f'{xml_escape(label)}</text>')

        for src, dst in model.edges:
            if src not in pos or dst not in pos:
                continue
            points = [at(pos[src])] + [at(p) for p in bends.get((src, dst), [])] + [at(pos[dst])]
            r1 = 6 if model.nodes[src].kind == "fixed" else NODE_R
            r2 = (6 if model.nodes[dst].kind == "fixed" else NODE_R) + 8
            points = trim_ends(points, r1, r2)
            path = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
            out.append(f'<polyline points="{path}" fill="none" stroke="{INK}" '
                       f'stroke-width="1.35" stroke-linejoin="round" '
                       f'marker-end="url(#arrow)"/>')

        for name, node in model.nodes.items():
            x, y = at(pos[name])
            tip = f'<title>{xml_escape(name)} {"~" if node.kind != "det" else "="} {xml_escape(node.dist)}</title>' if node.dist else ""
            if node.kind == "fixed":
                out.append(f'<g>{tip}<circle cx="{x:.1f}" cy="{y:.1f}" r="4.5" fill="{INK}"/>'
                           + svg_label(node, x, y - 15, size=14) + "</g>")
                continue
            fill = SHADE if node.kind == "obs" else "#ffffff"
            dash = ' stroke-dasharray="5 3"' if node.kind == "det" else ""
            out.append(f'<g>{tip}<circle cx="{x:.1f}" cy="{y:.1f}" r="{NODE_R}" fill="{fill}" '
                       f'stroke="{INK}" stroke-width="1.4"{dash}/>' + svg_label(node, x, y) + "</g>")

        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w:.0f} {h:.0f}" '
            f'width="{w * scale:.0f}" height="{h * scale:.0f}" font-family="{SERIF}">'
            '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" '
            'markerWidth="6.5" markerHeight="6.5" orient="auto-start-reverse">'
            f'<path d="M 0 0 L 10 5 L 0 10 z" fill="{INK}"/></marker></defs>'
            '<rect width="100%" height="100%" fill="#ffffff" rx="8"/>'
            + "".join(out) + "</svg>"
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
    const K
    pi ~ Dirichlet(K)

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
    # HMM, unrolled two steps (plates cannot show the chain itself)
    const T
    A ~ Dirichlet(T)
    B ~ Dirichlet(T)

    z_1 ~ Categorical(A)
    z_2 ~ Categorical(z_1, A)
    z_3 ~ Categorical(z_2, A)
    y_1 ~ Categorical(z_1, B) obs
    y_2 ~ Categorical(z_2, B) obs
    y_3 ~ Categorical(z_3, B) obs
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
    layout,
    mo,
    nested_plates,
    overlap,
    parse,
    to_daft,
    to_dot,
    to_mermaid,
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
                for export in (to_svg, to_dot, to_tikz, to_daft, to_mermaid):
                    if model.nodes or export is to_svg:
                        assert export(model).strip()

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
    - reading the DSL straight into PyMC/NumPyro, so the diagram and the
      sampler cannot drift apart;
    - `mo.ui.altair_chart`-style selection: click a node, see its Markov
      blanket highlighted.
    """)
    return


if __name__ == "__main__":
    app.run()
