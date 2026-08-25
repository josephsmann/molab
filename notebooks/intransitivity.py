# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "altair",
#     "marimo>=0.24.0",
#     "numpy",
#     "polars",
#     "pytest",
#     "scipy",
# ]
# ///

import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import polars as pl
    import altair as alt
    import pytest
    from scipy.linalg import schur

    rng = np.random.default_rng(0)
    return alt, mo, np, pl, pytest, rng, schur


@app.cell
def _(mo):
    mo.md(r"""
    # Ratings, Style Circles, and Intransitivity

    Paired-comparison models beyond Elo: what a scalar rating can and cannot express,
    and the linear algebra that separates the two.

    **Setup.** Players $Pl_1 \dots Pl_N$, matches $M_t(Pl_i, Pl_j) \to \{W, L\}$.
    We want $P(i \text{ beats } j) = f(T_i, T_j)$ for latent talent $T$.

    The whole notebook lives on one object — the **log-odds matrix**

    $$L_{ij} = \operatorname{logit} P(i \text{ beats } j) = \log \frac{P(i \text{ beats } j)}{1 - P(i \text{ beats } j)}$$

    which is necessarily **skew-symmetric**, $L^\top = -L$, since
    $P(i \text{ beats } j) = 1 - P(j \text{ beats } i)$.

    A consequence worth stating: $L_{ii} = -L_{ii} = 0$. A player is 50/50 against
    themselves. This is *structural* zero, and must be distinguished in code from a
    *missing* entry (a pair who have never played) — both look like `0.0`.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. The classical model, and its ceiling

    **Bradley–Terry / Elo / Glicko** are all the scalar case:

    $$L_{ij} = u_i - u_j$$

    with $u$ a single number per player. Glicko adds a hierarchical Gaussian prior
    (essential — without shrinkage a 3–0 player has an infinite MLE); Glicko-2 adds
    time dynamics. But the *functional form* is fixed.

    This forces **transitivity**. Go around any cycle and the terms telescope:

    $$L_{AB} + L_{BC} + L_{CA} = (u_A - u_B) + (u_B - u_C) + (u_C - u_A) = 0$$

    So if $A$ beats $B$ and $B$ beats $C$, the model *must* favour $A$ over $C$.
    Real racquet-sport data has intransitive triangles — style matchups. A scalar
    rating cannot represent them at all; it can only average over them.

    **Identifiability caution.** If $f$ is fully nonparametric and $T$ unconstrained,
    the model is badly unidentified — any monotone reparametrisation of $T$ can be
    absorbed into $f$. Fix $f$ (logistic) or constrain it hard.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. The fix: add a cross product

    Write

    $$L_{ij} = \underbrace{(u_i - u_j)}_{\text{rating}} \;+\; \underbrace{m(i,j)}_{\text{matchup}}$$

    The only hard requirement on $m$ is anti-symmetry, $m(i,j) = -m(j,i)$.
    The simplest such thing is a **2D cross product**. Give each player a *style
    vector* $s_i = (x_i, y_i)$ and set

    $$m(i,j) = x_i y_j - y_i x_j = s_i \times s_j = \|s_i\|\,\|s_j\|\,\sin(\theta_j - \theta_i)$$

    The polar form is where the intuition is:

    | Quantity | Meaning |
    |---|---|
    | angle $\theta_i$ | **what kind** of player you are |
    | radius $\|s_i\|$ | **how pronounced** your style is — at the origin you are style-neutral, pure Elo |
    | $\sin(\Delta\theta)$ | only the *difference* in style matters |

    Two consequences that are not obvious:

    - **Same style ($\Delta\theta = 0$) → no effect.** Ratings alone decide.
    - **Opposite style ($\Delta\theta = 180°$) → also no effect!** $\sin 180° = 0$.
      Polar-opposite styles play *dead even*. It is the **oblique** matchup
      ($\Delta\theta = 90°$) that is decisive.

    Rock–paper–scissors is three players at $0°, 120°, 240°$ with equal $u$ — each
    beats the next going round, a perfect 3-cycle no scalar model can produce.
    **The circle is the intransitivity.**
    """)
    return


@app.cell
def _(np):
    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-z))

    def style_matrix(angles, radii, lam=1.0):
        """Skew matrix from one style circle: m_ij = lam * (s_i x s_j)."""
        _x = np.asarray(radii) * np.cos(np.asarray(angles))
        _y = np.asarray(radii) * np.sin(np.asarray(angles))
        return lam * (np.outer(_x, _y) - np.outer(_y, _x))

    def gradient_matrix(u):
        """Rating (transitive) part: G_ij = u_i - u_j."""
        _u = np.asarray(u, dtype=float)
        _ones = np.ones_like(_u)
        return np.outer(_u, _ones) - np.outer(_ones, _u)

    return gradient_matrix, sigmoid, style_matrix


@app.cell
def _(mo):
    _n = mo.ui.slider(3, 8, value=4, label="players $N$")
    _lam = mo.ui.slider(0.0, 3.0, step=0.1, value=1.0, label="style strength $\\lambda$")
    _spread = mo.ui.slider(0.0, 2.0, step=0.1, value=0.0, label="rating spread")
    controls = mo.ui.dictionary({"n": _n, "lam": _lam, "spread": _spread})
    controls.vstack()
    return (controls,)


@app.cell
def _(controls, gradient_matrix, np, style_matrix):
    n_players = int(controls["n"].value)
    names = [chr(65 + _k) for _k in range(n_players)]

    # players spaced evenly around the style circle, ratings descending
    angles_demo = np.linspace(0.0, 2 * np.pi, n_players, endpoint=False)
    ratings_demo = np.linspace(1.0, -1.0, n_players) * controls["spread"].value
    ratings_demo = ratings_demo - ratings_demo.mean()

    C_demo = style_matrix(angles_demo, np.ones(n_players), lam=controls["lam"].value)
    G_demo = gradient_matrix(ratings_demo)
    L_demo = G_demo + C_demo
    return L_demo, angles_demo, names, ratings_demo


@app.cell
def _(L_demo, mo, names, pl, sigmoid):
    _P = sigmoid(L_demo)
    _rows = []
    for _i, _nm in enumerate(names):
        _row = {"": _nm}
        for _j, _nm2 in enumerate(names):
            _row[f"vs {_nm2}"] = "—" if _i == _j else f"{100 * _P[_i, _j]:.0f}%"
        _rows.append(_row)

    mo.vstack(
        [
            mo.md("**$P(\\text{row beats column})$**"),
            pl.DataFrame(_rows),
        ]
    )
    return


@app.cell
def _(alt, angles_demo, mo, names, np, pl, ratings_demo):
    _df = pl.DataFrame(
        {
            "player": names,
            "x": np.cos(angles_demo),
            "y": np.sin(angles_demo),
            "rating": ratings_demo,
        }
    )
    _pts = (
        alt.Chart(_df)
        .mark_point(size=200, filled=True)
        .encode(
            x=alt.X("x", scale=alt.Scale(domain=[-1.4, 1.4])),
            y=alt.Y("y", scale=alt.Scale(domain=[-1.4, 1.4])),
            color=alt.Color("rating", scale=alt.Scale(scheme="blueorange")),
            tooltip=["player", "rating"],
        )
    )
    _lab = alt.Chart(_df).mark_text(dy=-16, fontSize=14).encode(x="x", y="y", text="player")

    mo.vstack(
        [
            mo.md("**Positions on the style circle** (colour = rating)"),
            (_pts + _lab).properties(width=380, height=380),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Bilinear forms are just stacked cross products

    The general matchup term is written $m(i,j) = a_i^\top C\, a_j$ with $C$
    **skew-symmetric** ($C^\top = -C$). That looks forbidding. It isn't.

    **Fact.** Every real skew-symmetric matrix can be rotated by an orthogonal $Q$
    into block-diagonal form with $2\times2$ blocks

    $$Q^\top C Q = \begin{bmatrix} 0 & \lambda_1 \\ -\lambda_1 & 0 \end{bmatrix}
    \oplus \begin{bmatrix} 0 & \lambda_2 \\ -\lambda_2 & 0 \end{bmatrix} \oplus \cdots \oplus 0$$

    and within one block, $a^\top C b = \lambda (x_a y_b - y_a x_b)$ — **a cross
    product again**. So a bilinear form is nothing but a stack of independent style
    planes:

    $$m(i,j) = \sum_k \lambda_k \left( s_i^{(k)} \times s_j^{(k)} \right)$$

    Rank 2 = one style circle. Rank 4 = two independent circles (say pace-vs-patience
    and power-vs-touch). The matrix $C$ is just this before you rotate into the
    natural coordinates.

    **Skew rank is always even**, so in odd dimension there is always a leftover
    zero direction. A $5\times5$ skew matrix has rank $\le 4$ — never 5.
    """)
    return


@app.cell
def _(np, schur):
    def skew_canonical(C, tol=1e-9):
        """Youla / real-Schur block form of a skew matrix.

        Returns (T, Q, lambdas) with Q orthogonal, T = Q.T @ C @ Q block diagonal.
        """
        _T, _Q = schur(np.asarray(C, dtype=float), output="real")
        _lams = []
        _k = 0
        while _k < _T.shape[0] - 1:
            if abs(_T[_k, _k + 1]) > tol:
                _lams.append(abs(_T[_k, _k + 1]))
                _k += 2
            else:
                _k += 1
        return _T, _Q, np.array(sorted(_lams, reverse=True))

    return (skew_canonical,)


@app.cell
def _(mo, np, pl, skew_canonical):
    # A 5x5 skew matrix that looks like noise but is two clean style circles.
    A5 = (
        np.array(
            [
                [0, 30, 0, -52, 72],
                [-30, 0, -40, 0, 0],
                [0, 40, 0, 39, 96],
                [52, 0, -39, 0, 0],
                [-72, 0, -96, 0, 0],
            ],
            dtype=float,
        )
        / 65.0
    )

    _T5, _Q5, _lam5 = skew_canonical(A5)

    def _fmt(M, dec=3):
        return pl.DataFrame(
            {f"c{_j}": [round(float(M[_i, _j]), dec) for _i in range(M.shape[0])]
             for _j in range(M.shape[1])}
        )

    mo.vstack(
        [
            mo.md("**$A$ — entries over 65, structure invisible by eye**"),
            _fmt(A5),
            mo.md(f"**$Q^\\top A Q$ — two blocks, $\\lambda = {_lam5[0]:.2f}, {_lam5[1]:.2f}$, then zero**"),
            _fmt(_T5),
            mo.md(
                f"Singular values: {np.round(np.linalg.svd(A5, compute_uv=False), 3).tolist()} "
                "— **note the forced equal pairs.**"
            ),
        ]
    )
    return (A5,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. The Hodge split: hills and whirlpools

    Treat $L$ as a **flow** on the complete graph of players. Two extreme kinds exist:

    - **Gradient (potential) flow** — $L_{ij} = u_i - u_j$, everything running downhill
      from a single height field. Loops telescope to zero. *Transitive.*
    - **Circulation (divergence-free) flow** — whatever flows into a player flows out.
      No heights explain it. Loops gain. *Intransitive.*

    Every flow splits uniquely into the two — the discrete **Hodge (Helmholtz)**
    decomposition. The separating operator is **divergence**, i.e. the row sum:

    $$u_i = \frac{1}{N} \sum_j L_{ij}, \qquad G = u\mathbf{1}^\top - \mathbf{1}u^\top,
    \qquad C = L - G$$

    and $C\mathbf{1} = 0$ — every row of the circulation part sums to zero.

    Note the $1/N$ (not $1/(N-1)$): the $j = i$ term contributes a legitimate
    $L_{ii} = 0$, so you really are averaging $N$ numbers.

    Fitting the model = decomposing observed win patterns into **hill-climbing**
    (ratings) plus **whirlpools** (style circles).
    """)
    return


@app.cell
def _(np):
    def hodge_split(L):
        """L (skew) -> (u, G, C) with G transitive, C divergence-free."""
        _L = np.asarray(L, dtype=float)
        _N = _L.shape[0]
        _u = _L.sum(axis=1) / _N
        _u = _u - _u.mean()
        _ones = np.ones(_N)
        _G = np.outer(_u, _ones) - np.outer(_ones, _u)
        return _u, _G, _L - _G

    def intransitivity_ratio(L):
        """Fraction of squared Frobenius mass that is circulation."""
        _u, _G, _C = hodge_split(L)
        _g, _c = np.sum(_G**2), np.sum(_C**2)
        return _c / (_g + _c) if (_g + _c) > 0 else 0.0

    return hodge_split, intransitivity_ratio


@app.cell
def _(mo):
    mo.md(r"""
    ### Why SVD finds the planes

    Take $C$ skew. Then $C^\top C = -C^2$ is symmetric PSD. Inside one invariant
    plane $C$ acts as a $90°$ rotation scaled by $\lambda$, so $C^2$ acts as
    $-\lambda^2 I$ on that **whole plane**. Therefore:

    > **Every eigenvalue of $C^\top C$ has multiplicity two** — one per plane — and
    > the $2$D eigenspace *is* the whirlpool plane. Singular values of a skew matrix
    > come in **equal pairs**.

    That degeneracy is the fingerprint. A spectrum reading $2.8, 2.8, 0.9, 0.9, \epsilon\dots$
    is two style circles of strength $2.8$ and $0.9$. Dynamically: $\exp(tC)$ is
    orthogonal — literally a rotation, spinning each plane at rate $\lambda$. The
    whirlpool is not a metaphor; it is the flow the matrix generates.

    ### ⚠️ Subtract $G$ *first*

    $G$ is **also** skew and **also** rank 2, so it contributes its own $2\times2$
    block, indistinguishable from a style circle by block structure or spectrum alone.
    SVD'ing raw $L$ makes your largest "whirlpool" just the rating spread.

    What tells them apart is not the block but **the plane** — see §6.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. The four-player example

    $N=4$ is the smallest genuinely interesting case, and it is rigid: since
    $C\mathbf{1}=0$, rank $\le 3$; since skew rank is even, rank $\le 2$.
    **Four players admit exactly one style circle, never two.** ($N=5$ is the first
    case allowing two.)

    Below: the pure 4-cycle. Style vectors sit at the four compass points.
    Neighbours ($90°$ apart) are maximally lopsided; **opposites ($180°$ apart) are
    dead even.** All four row sums are zero, so every scalar model converges to four
    equal ratings and predicts 50/50 across the board — right on two pairs, off by
    23 points on the other four, forever.
    """)
    return


@app.cell
def _(gradient_matrix, hodge_split, mo, np, pl, sigmoid):
    C4 = np.array(
        [[0, 1, 0, -1], [-1, 0, 1, 0], [0, -1, 0, 1], [1, 0, -1, 0]], dtype=float
    )
    u4 = np.array([0.5, 0.5, -0.5, -0.5])
    L4 = gradient_matrix(u4) + C4
    names4 = ["A", "B", "C", "D"]

    def _tab(M, pct=False):
        _rows = []
        for _i, _nm in enumerate(names4):
            _row = {"": _nm}
            for _j, _nm2 in enumerate(names4):
                if _i == _j:
                    _row[_nm2] = "—"
                elif pct:
                    _row[_nm2] = f"{100 * sigmoid(M[_i, _j]):.0f}%"
                else:
                    _row[_nm2] = f"{M[_i, _j]:+.1f}"
            _rows.append(_row)
        return pl.DataFrame(_rows)

    _uu, _GG, _CC = hodge_split(L4)

    mo.vstack(
        [
            mo.md("**Pure circulation $C$ (log-odds), equal ratings**"),
            _tab(C4),
            mo.md("**As win probabilities** — a 73% ring, with A–C and B–D exactly even"),
            _tab(C4, pct=True),
            mo.md("**Now add ratings $u = (.5,.5,-.5,-.5)$: $L = G + C$**"),
            _tab(L4),
            mo.md(
                "A is a full logit above D in rating and *still only splits with them* — "
                "the style term exactly eats the talent gap. B, rated identically to A, "
                "is at 88% over C. Recovered ratings: "
                f"`{np.round(_uu, 3).tolist()}` ✓"
            ),
        ]
    )
    return C4, L4


@app.cell
def _(mo):
    mo.md(r"""
    ## 6. Kernel, rank, and the plane test

    **Kernel** = everything a map sends to zero; what it *forgets*.
    rank + nullity = input dimension.

    | Kernel | Meaning here |
    |---|---|
    | $\ker(B)$, $B u = u_i - u_j$ | constants $\mathbf{1}$ — Elo's gauge freedom (add 100 to everyone) |
    | $\dim \ker(\Delta)$ | **number of connected components** of the match graph |
    | $\ker(B^\top)$ (divergence) | the **cycle space** — where all intransitivity lives |
    | $\ker(C) \ni \mathbf{1}$ | why $N=4$ caps at one style circle |

    ### Why $G$ has rank exactly 2

    $G = u\mathbf{1}^\top - \mathbf{1}u^\top$ is two rank-1 outer products, so
    rank $\le 2$. For any $v$:

    $$Gv = \left(\textstyle\sum_j v_j\right) u - (u \cdot v)\,\mathbf{1}$$

    Every output is a combination of $u$ and $\mathbf{1}$ — so the column space is
    $\operatorname{span}\{u, \mathbf{1}\}$, dimension 2 unless all ratings are equal
    (then $u \propto \mathbf{1}$ and $G = 0$). Anything orthogonal to both is killed:
    nullity $N-2$.

    > **Transitive = rank 2.** Elo's entire expressive power is *one plane*, and that
    > plane contains $\mathbf{1}$. Everything intransitive lives in the other $N-2$
    > dimensions.

    ### What "plane" means

    A **plane** = a 2D subspace of $\mathbb{R}^N$: the span of two independent vectors,
    a flat sheet through the origin. Vectors here are **functions on players** —
    one number each. $u$ is one; $\mathbf{1} = (1,1,\dots,1)$ ("same number for
    everyone") is another.

    A skew matrix *acts by rotation*: feed it a vector from one of its planes and the
    output stays in that plane, turned $90°$ and scaled by $\lambda$. Feed it something
    orthogonal to all of them and you get zero. That is what "the matrix's planes" means.

    **The test that matters** — does the plane contain $\mathbf{1}$?

    - **Rating block:** plane is $\operatorname{span}\{u, \mathbf{1}\}$ — **contains** $\mathbf{1}$.
    - **Style circle:** plane is **orthogonal** to $\mathbf{1}$; its vectors sum to zero
      across players. No overall level, pure contrast.

    In the 4-player example the style plane is spanned by $(1,0,-1,0)$ and $(0,1,0,-1)$
    — the A-vs-C and B-vs-D contrasts. **A player's position on the style circle is
    simply their coordinates in the plane the SVD found.**
    """)
    return


@app.cell
def _(L4, hodge_split, mo, np, pl):
    def plane_test(M, tol=1e-8):
        """For each 2D invariant plane of skew M, how much of 1 lies in it."""
        _U, _S, _ = np.linalg.svd(np.asarray(M, dtype=float))
        _N = M.shape[0]
        _ones = np.ones(_N) / np.sqrt(_N)
        _out = []
        for _k in range(0, _N - 1, 2):
            if _S[_k] < tol:
                break
            _basis = _U[:, _k : _k + 2]
            _frac = float(np.sum((_basis.T @ _ones) ** 2))
            _out.append(
                {
                    "plane": _k // 2 + 1,
                    "lambda": round(float(_S[_k]), 3),
                    "share of 1 in plane": round(_frac, 3),
                    "verdict": (
                        "RATING"
                        if _frac > 0.9
                        else "style circle"
                        if _frac < 0.1
                        else "MIXED — subtract G first"
                    ),
                }
            )
        return pl.DataFrame(_out)

    _u4, _G4, _C4 = hodge_split(L4)

    mo.vstack(
        [
            mo.md("**Planes of $G$ alone** — contains $\\mathbf{1}$ entirely:"),
            plane_test(_G4),
            mo.md("**Planes of $C$ alone** — orthogonal to $\\mathbf{1}$:"),
            plane_test(_C4),
            mo.md(
                "**Planes of raw $L = G + C$** — and here is the trap in action. "
                "In this example $u = (.5,.5,-.5,-.5)$ happens to *lie inside the style "
                "plane*, so $G$ and $C$ share a direction and $L$ collapses to "
                "**rank 2**, not 4. Its single plane is half rating, half style — "
                "share of $\\mathbf{1}$ is exactly $0.5$, verdict ambiguous. "
                "You cannot read structure off raw $L$:"
            ),
            plane_test(L4),
        ]
    )
    return (plane_test,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 7. Connection to spectral clustering

    Let $B$ be the incidence operator, $(Bu)(i,j) = u_i - u_j$. Then the graph
    Laplacian is $\Delta = B^\top B$. Spectral clustering studies $B^\top B$ on the
    **vertex** side; intransitivity studies $\ker(B^\top)$ on the **edge** side.
    Same operator, opposite ends. The Hodge split $L = G + C$ *is*
    $\mathbb{R}^E = \operatorname{im}(B) \oplus \ker(B^\top)$.

    **Bridges carry no style information.** An edge is a bridge $\iff$ it lies in no
    cycle $\iff$ it is orthogonal to the cycle space $\iff$ it is pure gradient.
    So a lone crossover series linking two otherwise-separate groups of players tells
    you about *ratings only* — no amount of play on it can estimate matchup structure.

    **Same eigenvector, two questions.** $\operatorname{Var}(\hat u_i - \hat u_j)
    \propto$ **effective resistance** between $i$ and $j$, which is large exactly
    across sparse cuts and maximal across a bridge. So "best place to cut the graph"
    (the Fiedler vector) and "the rating contrast my data determines worst" are the
    same near-null direction of the same matrix. This is the chess phenomenon of
    isolated rating pools drifting — structurally invisible from inside either pool.

    **Where the analogy stops.** $\Delta$ is symmetric PSD: real spectrum, generically
    simple eigenvalues, eigenvectors are cluster indicators. $C$ is skew: singular
    values in **forced equal pairs**, and the object is a *plane*, not a vector.
    Also "low-rank matchup model" $\subsetneq$ "circulation": for $N \ge 6$ the cycle
    space is far bigger than any low-rank $C$ can reach. ($N = 4, 5$ they coincide,
    which is why the small examples felt so tight.)
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 8. Diagnostics for real data

    Before fitting anything fancy, three cheap checks.
    """)
    return


@app.cell
def _(hodge_split, np, rng, sigmoid):
    def empirical_logodds(wins, games, prior=1.0):
        """Shrunk empirical log-odds matrix; returns (L, mask of observed pairs)."""
        _w = np.asarray(wins, dtype=float)
        _g = np.asarray(games, dtype=float)
        _p = (_w + prior) / (_g + 2 * prior)
        _L = np.log(_p / (1 - _p))
        _L = 0.5 * (_L - _L.T)  # enforce skewness
        np.fill_diagonal(_L, 0.0)
        return _L, _g > 0

    def simulate(L_true, games_per_pair=40, seed=None):
        """Simulate a round robin from a true log-odds matrix."""
        _r = np.random.default_rng(seed) if seed is not None else rng
        _N = L_true.shape[0]
        _P = sigmoid(L_true)
        _wins = np.zeros((_N, _N))
        _games = np.zeros((_N, _N))
        for _i in range(_N):
            for _j in range(_i + 1, _N):
                _w = _r.binomial(games_per_pair, _P[_i, _j])
                _wins[_i, _j], _wins[_j, _i] = _w, games_per_pair - _w
                _games[_i, _j] = _games[_j, _i] = games_per_pair
        return _wins, _games

    def null_spectrum(wins, games, n_rep=200, seed=1):
        """Noise floor: reshuffle outcomes preserving schedule + marginal rates."""
        _r = np.random.default_rng(seed)
        _N = wins.shape[0]
        _u, _G, _ = hodge_split(empirical_logodds(wins, games)[0])
        _out = []
        for _ in range(n_rep):
            _Lr = _G.copy()
            _wn = np.zeros((_N, _N))
            _P = sigmoid(_G)
            for _i in range(_N):
                for _j in range(_i + 1, _N):
                    if games[_i, _j] > 0:
                        _w = _r.binomial(int(games[_i, _j]), _P[_i, _j])
                        _wn[_i, _j], _wn[_j, _i] = _w, games[_i, _j] - _w
            _Ln, _ = empirical_logodds(_wn, games)
            _, _, _Cn = hodge_split(_Ln)
            _out.append(np.linalg.svd(_Cn, compute_uv=False))
        return np.array(_out)

    return empirical_logodds, null_spectrum, simulate


@app.cell
def _(
    alt,
    empirical_logodds,
    hodge_split,
    intransitivity_ratio,
    mo,
    np,
    null_spectrum,
    pl,
    simulate,
    style_matrix,
):
    # Ground truth: 8 players, real ratings + ONE genuine style circle
    _N8 = 8
    _u8 = np.linspace(1.2, -1.2, _N8)
    _ang8 = np.linspace(0, 2 * np.pi, _N8, endpoint=False)
    _G8 = np.subtract.outer(_u8, _u8)
    L_true8 = _G8 + style_matrix(_ang8, np.ones(_N8), lam=0.8)

    wins8, games8 = simulate(L_true8, games_per_pair=40, seed=7)
    L_hat8, _ = empirical_logodds(wins8, games8)
    _u_hat, _G_hat, C_hat8 = hodge_split(L_hat8)

    _obs = np.linalg.svd(C_hat8, compute_uv=False)
    _null = null_spectrum(wins8, games8, n_rep=150)
    _floor = np.percentile(_null, 95, axis=0)

    _df = pl.DataFrame(
        {
            "index": list(range(1, _N8 + 1)) * 2,
            "sigma": np.concatenate([_obs, _floor]).tolist(),
            "series": ["observed"] * _N8 + ["null 95th pct"] * _N8,
        }
    )
    _chart = (
        alt.Chart(_df)
        .mark_line(point=True)
        .encode(
            x=alt.X("index:O", title="singular value index"),
            y=alt.Y("sigma:Q", title="σ of circulation part C"),
            color=alt.Color("series:N", scale=alt.Scale(scheme="dark2")),
            tooltip=["series", "index", alt.Tooltip("sigma", format=".3f")],
        )
        .properties(width=520, height=300)
    )

    mo.vstack(
        [
            mo.md(
                f"""
    **Truth:** 8 players, real rating spread, **one** style circle at $\\lambda = 0.8$.
    40 games per pair, simulated.

    - Intransitivity share $\\|C\\|_F^2 / (\\|G\\|_F^2 + \\|C\\|_F^2)$ = **{intransitivity_ratio(L_hat8):.1%}**
    - Observed $\\sigma(C)$: `{np.round(_obs, 2).tolist()}`

    The first **pair** clears the null floor; everything after is noise — correctly
    recovering *one* circle. Estimation noise inflates small singular values, so a
    decaying spectrum looks like structure when it isn't. **Always build the floor.**
                """
            ),
            _chart,
        ]
    )
    return (L_true8,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 9. Practical checklist for ClubLocker data

    1. **Connectivity first.** Build the match graph, compute $\Delta$'s spectrum.
       A near-zero second eigenvalue means weakly-linked pools — fix with pool-level
       offsets or a shared prior *before* anything else, or both ratings and style
       estimates will be dominated by the bridge.
    2. **Mask properly.** Structural $L_{ii}=0$ vs. missing (unplayed) pairs are
       opposite meanings wearing the same `0.0`.
    3. **Cheap wins before extra dimensions.** Time dynamics (talent drifts — Glicko-2's
       whole point) and **game-score margins as ordinal outcomes** rather than binary
       W/L. Margin carries a lot of information and effectively multiplies sample size.
    4. **Measure the whirlpool.** Compute the intransitivity share. If $\|C\|_F^2$ is
       a few percent of $\|G\|_F^2$, **skip all of this and use Glicko-2.**
    5. **Count 3-cycles** among well-sampled triples vs. what the scalar model predicts —
       a model-free check that intransitivity exists at all.
    6. **Null-model the spectrum.** Shuffle outcomes within each pair's observed game
       count, preserving schedule and marginal win rates. Only planes clearing that
       floor are worth modelling.
    7. **Then fit** scalar-only vs. scalar + rank-2, and compare **held-out log-loss**.
       Expect a modest gain overall — but concentrated exactly where you care:
       closely-matched pairs where the scalar model shrugs and says 50/50.

    With ~86k matches you likely have enough for rank 2–3 among frequently-observed
    players. The top singular pair gives each player an interpretable
    "what kind of player are you" coordinate.
    """)
    return


@app.cell
def _(
    A5,
    C4,
    L4,
    L_true8,
    gradient_matrix,
    hodge_split,
    np,
    plane_test,
    pytest,
    skew_canonical,
    style_matrix,
):
    def test_skew_and_zero_diagonal():
        for _M in (A5, C4, L4, L_true8):
            assert np.allclose(_M, -_M.T)
            assert float(np.max(np.abs(np.diag(_M)))) == pytest.approx(0.0, abs=1e-12)

    def test_singular_values_come_in_pairs():
        _s = np.linalg.svd(A5, compute_uv=False)
        assert _s[0] == pytest.approx(_s[1], abs=1e-9)
        assert _s[2] == pytest.approx(_s[3], abs=1e-9)
        assert _s[4] == pytest.approx(0.0, abs=1e-9)

    def test_canonical_lambdas():
        _T, _Q, _lams = skew_canonical(A5)
        assert np.allclose(_Q @ _Q.T, np.eye(5))
        assert _lams[0] == pytest.approx(2.0, abs=1e-6)
        assert _lams[1] == pytest.approx(1.0, abs=1e-6)

    def test_gradient_is_rank_two():
        _G = gradient_matrix([1.0, 0.3, -0.5, -0.8])
        assert np.linalg.matrix_rank(_G) == 2
        assert np.linalg.matrix_rank(gradient_matrix([2.0] * 4)) == 0

    def test_hodge_recovers_ratings_exactly():
        _u = np.array([0.7, 0.1, -0.2, -0.6])
        _u_hat, _G, _C = hodge_split(gradient_matrix(_u) + C4)
        assert _u_hat == pytest.approx(_u, abs=1e-12)
        assert np.allclose(_C, C4)
        assert np.allclose(_C @ np.ones(4), 0.0)  # divergence-free

    def test_opposite_styles_are_even():
        _M = style_matrix([0.0, np.pi], [1.0, 1.0], lam=2.0)
        assert _M[0, 1] == pytest.approx(0.0, abs=1e-12)

    def test_perpendicular_styles_are_maximal():
        _M = style_matrix([0.0, np.pi / 2], [1.0, 1.0], lam=1.0)
        assert _M[0, 1] == pytest.approx(1.0, abs=1e-12)

    def test_four_players_admit_only_one_circle():
        _C = style_matrix(np.linspace(0, 2 * np.pi, 4, endpoint=False), np.ones(4))
        assert np.linalg.matrix_rank(_C) == 2
        assert pytest.approx(0.0, abs=1e-9) == float(np.max(np.abs(_C @ np.ones(4))))

    def test_plane_test_separates_rating_from_style():
        _u, _G, _C = hodge_split(L4)
        assert plane_test(_G)["verdict"].to_list() == ["RATING"]
        assert plane_test(_C)["verdict"].to_list() == ["style circle"]

    def test_raw_L_planes_are_ambiguous():
        # u lies inside the style plane here, so L collapses to rank 2 and its
        # single plane is half rating / half style -- the reason to subtract G first.
        assert np.linalg.matrix_rank(L4) == 2
        _row = plane_test(L4).row(0, named=True)
        assert _row["share of 1 in plane"] == pytest.approx(0.5, abs=1e-9)
        assert "MIXED" in _row["verdict"]

    return


if __name__ == "__main__":
    app.run()
