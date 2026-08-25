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

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import polars as pl
    import altair as alt
    import pytest
    from scipy.linalg import schur
    from scipy.optimize import minimize

    rng = np.random.default_rng(0)

    return alt, minimize, mo, np, pl, pytest, rng, schur


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


@app.cell(hide_code=True)
def _(mo):

    mo.accordion(
        {
            r"**Aside:** what a skew-symmetric matrix *means*": mo.md(
                r"""
                Skew-symmetry is easy to accept as bookkeeping and easy to miss as
                geometry. One fact generates the rest: **a skew matrix is an
                infinitesimal rotation.**

                | | symmetric $S$ | skew $C$ |
                |---|---|---|
                | $x^\top M x$ | meaningful (energy, variance) | **identically zero** |
                | eigenvalues | real | purely imaginary, $\pm i\lambda$ |
                | eigenvectors | orthogonal axes to stand on | none real, only the kernel |
                | $\exp(M)$ | a stretch | a **rotation** |
                | it measures | alignment, length | **area, orientation** |

                The first row is the tell. $x^\top C x = 0$ for *every* $x$ — a skew matrix
                carries no quadratic form at all, so it structurally cannot answer "how much
                of $x$ is there?", the question every symmetric matrix exists to answer. It
                must be reporting on *pairs* of distinct vectors instead.

                The fourth row says what: $\exp(tC)$ is orthogonal with $\det = 1$ for all
                $t$, so $C$ is the **velocity of a rotation** — the derivative at the
                identity of a one-parameter family of rotations. Imaginary eigenvalues
                $\pm i\lambda$ mean no stretching in any direction, only angular speed
                $\lambda$ within a plane.

                **This is why $0°$ and $180°$ both vanish.** Since
                $a^\top J b = \|a\|\|b\| \sin \Delta\theta$ is the *signed area of the
                parallelogram* spanned by $a$ and $b$, the two "surprises" above are one
                sentence: you cannot build a parallelogram out of two parallel sticks, and
                antiparallel is still parallel. Perpendicular spans the most area. The
                rock-paper-scissors 3-cycle is an **orientation** — three vectors where each
                consecutive pair sweeps positive area.

                **And why skew rank is even.** Rotation happens *in a plane*, and a plane
                costs two dimensions, so rank arrives in pairs. In odd dimension something is
                always left over — which is Euler's rotation theorem ("every rotation of
                3-space has an axis") in disguise. Sharpest case: every $3\times3$ skew matrix
                is $v \mapsto w \times v$ for a fixed $w$, with $\ker = w$ the axis. The 2D
                cross product used here is not an *analogy* to rotation; it is rotation, one
                dimension down.

                ---

                **Every matrix splits into symmetric and skew parts.** For any square $M$,

                $$M = \underbrace{\frac{M + M^\top}{2}}_{\text{symmetric}}
                \;+\; \underbrace{\frac{M - M^\top}{2}}_{\text{skew}}$$

                uniquely, and the two parts are **orthogonal** under the trace inner product
                $\langle X, Y \rangle = \operatorname{tr}(X^\top Y)$. The dimensions partition
                cleanly: $\frac{n(n+1)}{2} + \frac{n(n-1)}{2} = n^2$, or $10 + 6 = 16$ at
                $n = 4$. So "symmetric" and "skew" are not two adjectives a matrix might
                happen to have — they are **complementary halves of all of matrix space**,
                the stretch part and the turn part of any linear map.

                This is the prototype for §4. There the same move is made one level in: $L$ is
                *already* skew, and the Hodge split cuts the skew half itself into gradient
                plus circulation.

                **Worth being explicit:** skew-symmetry is not what makes a matchup
                intransitive. $L$ is skew, $C$ is skew, and the pure rating matrix
                $G_{ij} = u_i - u_j$ is skew too — and perfectly transitive. Skew-symmetry is
                the **arena**, forced the moment you say a match has a winner and a loser
                ($P_{ij} + P_{ji} = 1$); it models nothing by itself.

                That arena has a shape. A skew $n \times n$ matrix has $\binom{n}{2}$ free
                entries, one per *unordered pair* with a sign for direction — precisely a
                **flow on the edges of the complete graph** $K_n$:

                $$\{\text{skew } n \times n\} \;\cong\; \mathbb{R}^E,
                \qquad \dim = \binom{n}{2}$$

                which is why §7 can start talking about incidence operators without changing
                the subject. At $n = 4$: $6 = 3 + 3$, three degrees of rating freedom and
                three of genuine circulation.
                """
            )
        }
    )
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

    **Fact** (Youla's real canonical form). Every real skew-symmetric matrix can be
    brought by an **orthogonal** change of basis $Q$ into block-diagonal form: $2\times2$
    blocks, then a zero block filling whatever dimension is left over,

    $$Q^\top C Q = \begin{bmatrix} 0 & \lambda_1 \\ -\lambda_1 & 0 \end{bmatrix}
    \oplus \begin{bmatrix} 0 & \lambda_2 \\ -\lambda_2 & 0 \end{bmatrix} \oplus \cdots \oplus 0$$

    and within one block, $a^\top C b = \lambda (x_a y_b - y_a x_b)$ — **a cross
    product again**. So a bilinear form is nothing but a stack of independent style
    planes:

    $$m(i,j) = \sum_k \lambda_k \left( s_i^{(k)} \times s_j^{(k)} \right)$$

    Rank 2 = one style circle. Rank 4 = two independent circles (say pace-vs-patience
    and power-vs-touch). The matrix $C$ is just this before you change to the
    natural coordinates.

    **Skew rank is always even**, so in odd dimension there is always a leftover
    zero direction. A $5\times5$ skew matrix has rank $\le 4$ — never 5. That is also why
    the trailing $0$ above is **not optional**: whenever $C$ is singular the form needs it.

    **One caution on "orthogonal".** $Q$ need not be a *rotation*. Pinning every
    $\lambda_k > 0$ can force a reflection — rotations commute with $J$, so none of them can
    flip that sign. Only $|\lambda_k|$ is basis-independent, which is why `skew_canonical`
    returns absolute values.
    """)
    return


@app.cell(hide_code=True)
def _(mo):

    mo.accordion(
        {
            r"**Notation:** what is $\oplus$?": mo.md(
                r"""
                $\oplus$ is the **direct sum** — *put these side by side in independent
                coordinates, with nothing crossing between them*. It is not addition:
                the pieces do not overlap, they occupy separate slots.

                **On matrices.** For $A$ of size $m$ and $B$ of size $n$,

                $$A \oplus B = \begin{bmatrix} A & 0 \\ 0 & B \end{bmatrix}$$

                of size $m + n$ — block diagonal, zeros off the blocks. In code this is
                `scipy.linalg.block_diag(A, B)`.

                The zeros carry the meaning. Because the off-diagonal blocks vanish, the
                matrix never mixes the first two coordinates with the second two: a vector
                living in block 1 maps into block 1. Each block is a sealed subsystem.

                **On spaces.** $V = U \oplus W$ means every $v$ splits as $u + w$ with
                $u \in U$, $w \in W$, **uniquely**, and $U \cap W = \{0\}$. That is the
                sense used in §4 and §7: $L = G + C$ is not one decomposition among many,
                it is *the* decomposition, which is what makes `hodge_split` well defined.

                The two senses are the same fact. A block-diagonal matrix is exactly a map
                that respects a direct-sum split of the space it acts on.

                **Here.** $Q^\top C Q = \lambda_1 J \oplus \lambda_2 J \oplus 0$ says the
                style planes are mutually oblivious — pace-vs-patience and power-vs-touch
                add independently, and $\lambda_k$ says how hard each one bites. The
                trailing $0$ is a zero *block* of whatever dimension is left over, not a
                single entry: it is the leftover direction with no style content, the
                reason skew rank is even.

                ---

                **Is the $\sum_k$ below doing $\oplus$ operations?** No — it is ordinary
                addition of real numbers. The type signatures differ: $\oplus$ combines
                *matrices into a larger matrix* (or subspaces into a larger space), while
                $\sum_k \lambda_k (s_i^{(k)} \times s_j^{(k)})$ adds *scalars*. By then the
                vectors have been contracted against each block and you have landed in
                $\mathbb{R}$.

                But the $\oplus$ is precisely *why* that sum takes this form. Write
                $T = Q^\top C Q$ for the canonical form and $b = Q^\top a$ for coordinates in the
                basis that exposes it. Partition the indices $\{1, \dots, n\}$ into the consecutive
                groups those blocks occupy — $\{1,2\}, \{3,4\}, \dots$, followed by the leftover
                kernel indices — and let $b^{(k)}$ be the slice of $b$ lying in group $k$.

                **$T_{k\ell}$ is a sub*matrix*, not an entry** — the rectangular block of $T$ sitting
                in block-row $k$ and block-column $\ell$:

                $$T_{k\ell} \;=\; \big[\, T_{pq} \,\big]_{\,p \,\in\, \text{group } k,\;\,
                q \,\in\, \text{group } \ell} \;\in\; \mathbb{R}^{\,n_k \times n_\ell}$$

                For the $5 \times 5$ example below the groups are $\{1,2\}, \{3,4\}, \{5\}$, so
                $T_{11}$ and $T_{12}$ are $2 \times 2$, while $T_{13}$ is $2 \times 1$ and $T_{33}$ is
                the $1 \times 1$ leftover. Multiplying blockwise,

                $$m(i,j) = a_i^\top C a_j = b_i^\top T\, b_j
                = \sum_{k,\ell} \left( b_i^{(k)} \right)^\top T_{k\ell}\, b_j^{(\ell)}$$

                a **double** sum over every pair of groups — each style plane potentially interacting
                with every other. So far this is only block matrix multiplication; no structure has
                been used. The direct sum is exactly the claim that

                $$T_{kk} = \lambda_k J \qquad\text{and}\qquad T_{k\ell} = 0
                \;\;\text{ for } k \neq \ell$$

                so every cross term dies and the double sum collapses to a single $\sum_k$, whose
                $k$-th term is $\lambda_k \big( s_i^{(k)} \times s_j^{(k)} \big)$ once you write out
                $b^\top J b'$. A general skew $C$ *would* carry those cross terms; rotating into
                canonical coordinates is what removes them.

                *Sign convention.* $\lambda_k$ is determined only up to a choice — flipping one basis
                vector inside a plane reverses that plane's circulation and the sign of $\lambda_k$
                with it. Only $|\lambda_k|$ is invariant, which is why `skew_canonical` returns
                absolute values.

                So: $\oplus$ upstream, $\sum$ downstream. The direct sum is a structural
                claim about the operator; the plain sum is what that structure buys you
                when you evaluate it. Staying at matrix level you may write the same fact
                with $\oplus$ throughout:

                $$Q^\top C Q = \bigoplus_k \lambda_k J,
                \qquad J = \begin{bmatrix} 0 & 1 \\ -1 & 0 \end{bmatrix}$$
                """
            )
        }
    )
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Running the checklist

    The functions below implement items 1, 2, 4, 5, 6 and 7 as things you can actually
    run. Item 3 is a modelling decision, not a diagnostic, so it has no code here.

    **Pointing this at real data.** Every ClubLocker export reduces to a list of
    `(winner, loser)` pairs, so that is the only input `pairs_to_matrices` asks for. It
    returns the `(wins, games)` pair that every function downstream consumes. The demo
    below runs on a *simulated* league with ClubLocker's awkward shape — two weakly-linked
    clubs, wildly uneven schedule, most pairs never meeting — because this repo is public
    and real match records with real names do not belong in it.
    """)
    return


@app.cell
def _(np, sigmoid):

    def pairs_to_matrices(results, players=None):
        """[(winner, loser), ...] -> (wins, games, names).

        wins[i, j] = times i beat j; games[i, j] = times they met (symmetric).
        Unplayed pairs stay 0 in *games* -- that is the mask, and it is what
        distinguishes a missing pair from a structural L_ii = 0.
        """
        _res = list(results)
        if players is None:
            _seen = {}
            for _w, _l in _res:
                _seen.setdefault(_w, None)
                _seen.setdefault(_l, None)
            players = sorted(_seen)
        _idx = {_p: _k for _k, _p in enumerate(players)}
        _n = len(players)
        _W = np.zeros((_n, _n))
        _G = np.zeros((_n, _n))
        for _w, _l in _res:
            _i, _j = _idx[_w], _idx[_l]
            _W[_i, _j] += 1
            _G[_i, _j] += 1
            _G[_j, _i] += 1
        return _W, _G, list(players)


    def split_games(wins, games, frac=0.3, seed=0):
        """Hypergeometric train/test split within each pair's own game count.

        Splitting *within* a pair, not across pairs, keeps every player in both
        folds -- a held-out pair whose players never appear in training has no
        rating to predict from.
        """
        _r = np.random.default_rng(seed)
        _W, _G = np.asarray(wins, float), np.asarray(games, float)
        _n = _W.shape[0]
        _Wte, _Gte = np.zeros((_n, _n)), np.zeros((_n, _n))
        for _i in range(_n):
            for _j in range(_i + 1, _n):
                _tot = int(_G[_i, _j])
                _nte = int(round(frac * _tot))
                if _tot == 0 or _nte == 0:
                    continue
                _wte = int(_r.hypergeometric(int(_W[_i, _j]), _tot - int(_W[_i, _j]), _nte))
                _Wte[_i, _j], _Wte[_j, _i] = _wte, _nte - _wte
                _Gte[_i, _j] = _Gte[_j, _i] = _nte
        return _W - _Wte, _G - _Gte, _Wte, _Gte


    def logloss(L, wins, games):
        """Mean per-game binary log-loss, over observed pairs only."""
        _iu = np.triu_indices(np.asarray(L).shape[0], 1)
        _p = np.clip(sigmoid(np.asarray(L, float)[_iu]), 1e-12, 1 - 1e-12)
        _w, _n = np.asarray(wins, float)[_iu], np.asarray(games, float)[_iu]
        _m = _n > 0
        if not _m.any():
            return float("nan")
        return float(
            -(_w[_m] * np.log(_p[_m]) + (_n[_m] - _w[_m]) * np.log(1 - _p[_m])).sum() / _n[_m].sum()
        )


    return logloss, pairs_to_matrices, split_games


@app.cell
def _(np):

    def match_graph_report(games, names=None, tol=1e-9):
        """Checklist 1: connectivity of the match graph, before any model.

        Returns a dict; `laplacian_spectrum` is the thing to look at first. A
        near-zero second eigenvalue (the Fiedler value) means weakly-linked pools.
        """
        _A = (np.asarray(games, float) > 0).astype(float)
        np.fill_diagonal(_A, 0.0)
        _n = _A.shape[0]
        _names = list(names) if names is not None else [str(_k) for _k in range(_n)]
        _lap = np.diag(_A.sum(axis=1)) - _A
        _ev = np.linalg.eigvalsh(_lap)
        _ncomp = int(np.sum(_ev < tol))
        # Fiedler value = second-smallest eigenvalue. It is 0 exactly when the graph
        # is disconnected, which is the signal we want -- not the first nonzero one.
        _fiedler = float(_ev[1]) if _n > 1 else 0.0
        # within-component gap: useful once you know it is connected
        _gap = float(_ev[_ncomp]) if _ncomp < _n else 0.0

        # effective resistance: Var(u_i - u_j) is proportional to this
        _pinv = np.linalg.pinv(_lap)
        _res = np.add.outer(np.diag(_pinv), np.diag(_pinv)) - 2 * _pinv

        # bridges: an edge whose removal disconnects. Pure gradient, no style content.
        _bridges = []
        for _i in range(_n):
            for _j in range(_i + 1, _n):
                if _A[_i, _j] == 0:
                    continue
                _A2 = _A.copy()
                _A2[_i, _j] = _A2[_j, _i] = 0.0
                _lap2 = np.diag(_A2.sum(axis=1)) - _A2
                if int(np.sum(np.linalg.eigvalsh(_lap2) < tol)) > _ncomp:
                    _bridges.append((_names[_i], _names[_j]))

        # the rating contrast the data determines worst
        _res_masked = _res.copy()
        np.fill_diagonal(_res_masked, -np.inf)
        _wi, _wj = np.unravel_index(np.argmax(_res_masked), _res.shape)

        _played = int((_A.sum()) // 2)
        return {
            "players": _n,
            "pairs_played": _played,
            "pairs_possible": _n * (_n - 1) // 2,
            "density": _played / (_n * (_n - 1) / 2) if _n > 1 else 0.0,
            "components": _ncomp,
            "fiedler": _fiedler,
            "spectral_gap": _gap,
            "laplacian_spectrum": _ev,
            "bridges": _bridges,
            "resistance": _res,
            "worst_contrast": (_names[int(_wi)], _names[int(_wj)], float(_res[_wi, _wj])),
        }


    def unplayed_mask(games):
        """Checklist 2: True where a pair never met. Structural L_ii = 0 is NOT missing."""
        _G = np.asarray(games, float)
        _m = _G == 0
        np.fill_diagonal(_m, False)
        return _m


    return match_graph_report, unplayed_mask


@app.cell
def _(empirical_logodds, hodge_split, np, sigmoid):

    def count_intransitive_triples(wins, games, min_games=3):
        """Checklist 5: model-free evidence that intransitivity exists at all.

        Counts 3-cycles (A>B>C>A) among triples where all three pairs are
        sufficiently sampled. Ties are broken toward the higher win count and
        excluded when exactly even.
        """
        _W, _G = np.asarray(wins, float), np.asarray(games, float)
        _n = _W.shape[0]
        _beats = np.zeros((_n, _n), dtype=bool)
        _decided = np.zeros((_n, _n), dtype=bool)
        for _i in range(_n):
            for _j in range(_n):
                if _i != _j and _G[_i, _j] >= min_games and _W[_i, _j] != _G[_i, _j] / 2:
                    _beats[_i, _j] = _W[_i, _j] > _G[_i, _j] / 2
                    _decided[_i, _j] = True
        _cyc = _tot = 0
        for _i in range(_n):
            for _j in range(_i + 1, _n):
                for _k in range(_j + 1, _n):
                    if not (_decided[_i, _j] and _decided[_j, _k] and _decided[_i, _k]):
                        continue
                    _tot += 1
                    _fwd = _beats[_i, _j] and _beats[_j, _k] and _beats[_k, _i]
                    _bwd = _beats[_j, _i] and _beats[_k, _j] and _beats[_i, _k]
                    if _fwd or _bwd:
                        _cyc += 1
        return _cyc, _tot


    def expected_triples_under_scalar(wins, games, min_games=3, n_rep=200, seed=0):
        """Null for checklist 5: how many 3-cycles a *transitive* model produces anyway.

        Fits the scalar part by Hodge projection, then resimulates the observed
        schedule. Sampling noise alone creates cycles, so a raw count means nothing
        without this floor.
        """
        _r = np.random.default_rng(seed)
        _G = np.asarray(games, float)
        _L, _ = empirical_logodds(wins, games)
        _u, _Gm, _ = hodge_split(_L)
        _P = sigmoid(_Gm)
        _n = _G.shape[0]
        _out = []
        for _ in range(n_rep):
            _Wn = np.zeros((_n, _n))
            for _i in range(_n):
                for _j in range(_i + 1, _n):
                    if _G[_i, _j] > 0:
                        _w = _r.binomial(int(_G[_i, _j]), _P[_i, _j])
                        _Wn[_i, _j], _Wn[_j, _i] = _w, _G[_i, _j] - _w
            _out.append(count_intransitive_triples(_Wn, _G, min_games)[0])
        return np.array(_out)


    return count_intransitive_triples, expected_triples_under_scalar


@app.cell
def _(logloss, minimize, np, pl, sigmoid, split_games):

    def fit_paired_model(wins, games, rank=0, ridge=1e-2, seed=0):
        """Checklist 7: MLE for L = (u_i - u_j) + sum_k (x_i y_j - y_i x_j).

        rank=0 is Bradley-Terry / Elo. rank=2 adds one style circle, rank=4 two.
        Unplayed pairs contribute nothing to the likelihood -- the mask is honoured
        by construction, since games == 0 zeroes that term.
        """
        _W, _G = np.asarray(wins, float), np.asarray(games, float)
        _n = _W.shape[0]
        _iu = np.triu_indices(_n, 1)
        _w, _g = _W[_iu], _G[_iu]
        _npar = _n * rank

        def _build(theta):
            _u = theta[:_n]
            _u = _u - _u.mean()
            _L = np.subtract.outer(_u, _u)
            _S = theta[_n:].reshape(_n, rank) if rank else np.zeros((_n, 0))
            for _k in range(0, rank, 2):
                _x, _y = _S[:, _k], _S[:, _k + 1]
                _L = _L + (np.outer(_x, _y) - np.outer(_y, _x))
            return _u, _S, _L

        def _nll(theta):
            _, _, _L = _build(theta)
            _p = np.clip(sigmoid(_L[_iu]), 1e-12, 1 - 1e-12)
            return -np.sum(_w * np.log(_p) + (_g - _w) * np.log(1 - _p)) + ridge * np.sum(theta**2)

        _r = np.random.default_rng(seed)
        _res = minimize(_nll, _r.normal(scale=0.1, size=_n + _npar), method="L-BFGS-B")
        _u, _S, _L = _build(_res.x)
        return {"u": _u, "S": _S, "L": _L, "converged": bool(_res.success), "nll": float(_res.fun)}


    def holdout_comparison(wins, games, ranks=(0, 2, 4), frac=0.3, seed=0, ridge=1e-2):
        """Fit each rank on a training split, score held-out log-loss.

        The number that decides whether any of this is worth it. Expect a modest
        overall gain concentrated on closely-matched pairs.
        """
        _Wtr, _Gtr, _Wte, _Gte = split_games(wins, games, frac=frac, seed=seed)
        _rows = []
        for _rk in ranks:
            _fit = fit_paired_model(_Wtr, _Gtr, rank=_rk, ridge=ridge, seed=seed)
            _rows.append(
                {
                    "model": "scalar (Elo)" if _rk == 0 else f"scalar + rank {_rk}",
                    "rank": _rk,
                    "train logloss": round(logloss(_fit["L"], _Wtr, _Gtr), 4),
                    "held-out logloss": round(logloss(_fit["L"], _Wte, _Gte), 4),
                    "converged": _fit["converged"],
                }
            )
        return pl.DataFrame(_rows)


    return fit_paired_model, holdout_comparison


@app.cell
def _(gradient_matrix, mo, np, sigmoid, style_matrix):

    def make_league(games_per_pair, lam=1.2, n_club=7, seed=7):
        """A league shaped like ClubLocker data rather than a clean round robin.

        Two clubs that mostly play internally, a single thin crossover, per-player
        activity varying several-fold, and most pairs never meeting. Style angles are
        shuffled so they do NOT track club or rating -- otherwise every
        well-sampled triple sits inside one half of the style circle, which is
        provably transitive, and no 3-cycle can ever appear.
        """
        _r = np.random.default_rng(seed)
        _names = [f"{_c}{_k}" for _c in "AB" for _k in range(1, n_club + 1)]
        _n = len(_names)
        _u = np.concatenate(
            [np.linspace(0.9, -0.36, n_club), np.linspace(0.54, -0.9, n_club)]
        )
        _ang = _r.permutation(np.linspace(0, 2 * np.pi, _n, endpoint=False))
        _L = gradient_matrix(_u) + style_matrix(_ang, np.ones(_n), lam=lam)

        _G = np.zeros((_n, _n))
        for _i in range(_n):
            for _j in range(_i + 1, _n):
                if (_i < n_club) == (_j < n_club):
                    _G[_i, _j] = _G[_j, _i] = int(
                        _r.poisson(games_per_pair * _r.uniform(0.3, 1.7))
                    )
        _cross = [(_i, _j) for _i in range(n_club) for _j in range(n_club, _n)]
        _ci, _cj = _cross[int(_r.choice(len(_cross), 1)[0])]
        _G[_ci, _cj] = _G[_cj, _ci] = max(4, int(games_per_pair * 0.4))

        _W = np.zeros((_n, _n))
        _P = sigmoid(_L)
        for _i in range(_n):
            for _j in range(_i + 1, _n):
                _m = int(_G[_i, _j])
                if _m:
                    _wn = _r.binomial(_m, _P[_i, _j])
                    _W[_i, _j], _W[_j, _i] = _wn, _m - _wn
        return {"wins": _W, "games": _G, "names": _names, "L_true": _L, "lam": lam}


    # Same generating process, two data volumes: the question is not whether style
    # exists (it does, by construction) but whether this much data can see it.
    league_thin = make_league(9)
    league_deep = make_league(30)

    mo.md(
        f"""
    Two leagues from the **same** ground truth — 14 players, two clubs, one style
    circle at $\\lambda = 1.2$ — differing only in how much was played:

    | | matches | pairs that met | median games/pair |
    |---|---|---|---|
    | **thin** | {int(league_thin["games"].sum() // 2)} | {int((np.triu(league_thin["games"], 1) > 0).sum())} of 91 | {int(np.median(league_thin["games"][league_thin["games"] > 0]))} |
    | **deep** | {int(league_deep["games"].sum() // 2)} | {int((np.triu(league_deep["games"], 1) > 0).sum())} of 91 | {int(np.median(league_deep["games"][league_deep["games"] > 0]))} |
    """
    )

    return league_deep, league_thin, make_league


@app.cell
def _(
    alt,
    count_intransitive_triples,
    empirical_logodds,
    expected_triples_under_scalar,
    hodge_split,
    holdout_comparison,
    intransitivity_ratio,
    league_deep,
    league_thin,
    match_graph_report,
    mo,
    np,
    null_spectrum,
    pl,
    unplayed_mask,
):

    def run_checklist(league, label, min_games=3, n_rep=120, seed=3):
        """Items 1, 2, 4, 5, 6, 7 on one league. Returns a marimo view."""
        _W, _G, _nm = league["wins"], league["games"], league["names"]
        _L, _ = empirical_logodds(_W, _G)
        _rep = match_graph_report(_G, _nm)

        # 1 + 2: connectivity and masking
        _miss = unplayed_mask(_G)
        _conn = pl.DataFrame(
            [
                {"check": "players", "value": str(_rep["players"])},
                {"check": "matches", "value": str(int(_G.sum() // 2))},
                {"check": "pairs played", "value": f'{_rep["pairs_played"]} of {_rep["pairs_possible"]}'},
                {"check": "unplayed pairs (masked)", "value": str(int(_miss.sum() // 2))},
                {"check": "components", "value": str(_rep["components"])},
                {"check": "Fiedler value", "value": f'{_rep["fiedler"]:.3f}'},
                {"check": "bridges", "value": str(_rep["bridges"]) if _rep["bridges"] else "none"},
                {
                    "check": "worst-determined contrast",
                    "value": f'{_rep["worst_contrast"][0]}-{_rep["worst_contrast"][1]}'
                    f' (R={_rep["worst_contrast"][2]:.2f})',
                },
            ]
        )

        # 4: how much of the flow is circulation
        _share = intransitivity_ratio(_L)

        # 5: 3-cycles against a transitive null
        _obs_c, _tot_c = count_intransitive_triples(_W, _G, min_games)
        _null_c = expected_triples_under_scalar(_W, _G, min_games, n_rep=n_rep, seed=seed)
        _c95 = float(np.percentile(_null_c, 95))
        _cyc_verdict = "clears the null" if _obs_c > _c95 else "inside the null — no evidence"

        # 6: circulation spectrum against its own null
        _u, _Gm, _C = hodge_split(_L)
        _sv = np.linalg.svd(_C, compute_uv=False)
        _null_s = null_spectrum(_W, _G, n_rep=min(n_rep, 120), seed=seed)
        _floor = np.percentile(_null_s, 95, axis=0)
        _planes = int(np.sum(_sv > _floor) // 2)

        # 7: does the extra structure pay on held-out games
        _ho = holdout_comparison(_W, _G, ranks=(0, 2), frac=0.3, seed=seed, ridge=1e-1)
        _hl = _ho["held-out logloss"].to_list()
        _pays = _hl[1] < _hl[0]

        _spec_df = pl.DataFrame(
            {
                "index": list(range(1, len(_sv) + 1)) * 2,
                "sigma": np.concatenate([_sv, _floor]).tolist(),
                "series": ["observed σ(C)"] * len(_sv) + ["null 95th pct"] * len(_sv),
            }
        )
        _chart = (
            alt.Chart(_spec_df)
            .mark_line(point=True)
            .encode(
                x=alt.X("index:O", title="singular value index"),
                y=alt.Y("sigma:Q", title="σ of circulation part"),
                color=alt.Color("series:N", scale=alt.Scale(scheme="dark2")),
                tooltip=["series", "index", alt.Tooltip("sigma", format=".3f")],
            )
            .properties(width=430, height=230)
        )

        return mo.vstack(
            [
                mo.md(f"### {label}"),
                mo.md("**1–2. Connectivity and masking**"),
                _conn,
                mo.md(
                    f"**4. Intransitivity share** = **{_share:.1%}** of squared Frobenius mass "
                    "— but read it next to the nulls below, since estimation noise inflates it."
                ),
                mo.md(
                    f"**5. 3-cycles**: observed **{_obs_c}** of {_tot_c} well-sampled triples; "
                    f"transitive null mean {_null_c.mean():.1f}, 95th pct {_c95:.0f} → **{_cyc_verdict}**."
                ),
                mo.md(
                    f"**6. Spectrum**: {_planes} plane(s) clear the null floor "
                    f"(truth is 1, at $\\lambda$ = {league['lam']})."
                ),
                _chart,
                mo.md("**7. Held-out log-loss** — the number that decides it:"),
                _ho,
                mo.md(
                    f"→ **{'rank 2 pays for itself' if _pays else 'rank 2 does NOT pay — use Glicko-2'}** "
                    f"({_hl[1]:.4f} vs {_hl[0]:.4f} for scalar)."
                ),
            ]
        )


    def _label(league, word):
        return f"{word} league — {int(league['games'].sum() // 2)} matches"


    mo.hstack(
        [
            run_checklist(league_thin, _label(league_thin, "Thin")),
            run_checklist(league_deep, _label(league_deep, "Deep")),
        ],
        widths="equal",
        gap=2,
    )

    return


@app.cell
def _(
    count_intransitive_triples,
    fit_paired_model,
    gradient_matrix,
    holdout_comparison,
    logloss,
    make_league,
    match_graph_report,
    np,
    pairs_to_matrices,
    pytest,
    sigmoid,
    split_games,
    unplayed_mask,
):

    def test_pairs_to_matrices_counts_and_symmetry():
        _W, _G, _nm = pairs_to_matrices(
            [("ann", "bob"), ("bob", "cy"), ("cy", "ann"), ("ann", "bob")]
        )
        assert _nm == ["ann", "bob", "cy"]
        assert np.allclose(_G, _G.T)                       # games symmetric
        assert np.allclose(_W + _W.T, _G)                  # every game has one winner
        assert _W[0, 1] == 2 and _W[1, 0] == 0             # ann beat bob twice
        assert float(np.max(np.abs(np.diag(_G)))) == 0.0   # nobody plays themselves


    def test_unplayed_mask_excludes_the_diagonal():
        _G = np.array([[0.0, 3.0, 0.0], [3.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        _m = unplayed_mask(_G)
        assert not _m.diagonal().any()                     # structural 0 is not "missing"
        assert _m[0, 2] and _m[2, 0] and not _m[0, 1]


    def test_split_games_conserves_every_game():
        _W, _G, _ = pairs_to_matrices(
            [("a", "b")] * 7 + [("b", "a")] * 5 + [("a", "c")] * 3 + [("c", "b")] * 4
        )
        _Wtr, _Gtr, _Wte, _Gte = split_games(_W, _G, frac=0.4, seed=0)
        assert np.allclose(_Gtr + _Gte, _G)
        assert np.allclose(_Wtr + _Wte, _W)
        assert (_Gtr >= 0).all() and (_Gte >= 0).all()
        assert np.allclose(_Wte + _Wte.T, _Gte)


    def test_match_graph_report_finds_components_and_bridges():
        # two triangles joined by a single edge
        _G = np.zeros((6, 6))
        for _i, _j in [(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5)]:
            _G[_i, _j] = _G[_j, _i] = 5.0
        _split = match_graph_report(_G)
        assert _split["components"] == 2
        assert _split["fiedler"] == pytest.approx(0.0, abs=1e-9)
        _G[2, 3] = _G[3, 2] = 5.0
        _joined = match_graph_report(_G)
        assert _joined["components"] == 1
        assert _joined["fiedler"] > 0
        assert ("2", "3") in _joined["bridges"]            # the lone link is a bridge


    def test_count_intransitive_triples_detects_a_rock_paper_scissors():
        _W, _G, _ = pairs_to_matrices(
            [("a", "b")] * 4 + [("b", "c")] * 4 + [("c", "a")] * 4
        )
        assert count_intransitive_triples(_W, _G, min_games=3) == (1, 1)
        # a transitive triple is not a cycle
        _W2, _G2, _ = pairs_to_matrices(
            [("a", "b")] * 4 + [("b", "c")] * 4 + [("a", "c")] * 4
        )
        assert count_intransitive_triples(_W2, _G2, min_games=3) == (0, 1)


    def test_logloss_is_minimised_at_the_truth():
        _u = np.array([0.8, 0.1, -0.9])
        _L = gradient_matrix(_u)
        _P = sigmoid(_L)
        _G = np.full((3, 3), 400.0)
        np.fill_diagonal(_G, 0.0)
        _W = _P * _G                                       # noiseless expected wins
        _truth = logloss(_L, _W, _G)
        for _pert in (0.4, -0.4):
            assert logloss(gradient_matrix(_u + np.array([_pert, 0.0, 0.0])), _W, _G) > _truth


    def test_fit_recovers_a_scalar_truth():
        _u = np.array([1.0, 0.2, -0.4, -0.8])
        _L = gradient_matrix(_u)
        _G = np.full((4, 4), 4000.0)
        np.fill_diagonal(_G, 0.0)
        _W = sigmoid(_L) * _G
        _fit = fit_paired_model(_W, _G, rank=0, ridge=1e-9, seed=0)
        assert _fit["converged"]
        assert _fit["u"] == pytest.approx(_u - _u.mean(), abs=2e-2)


    def test_rank_two_beats_scalar_when_style_is_real_and_well_sampled():
        _dense = make_league(60, lam=1.2, seed=11)
        _ho = holdout_comparison(_dense["wins"], _dense["games"], ranks=(0, 2), seed=1, ridge=1e-1)
        _hl = _ho["held-out logloss"].to_list()
        assert _hl[1] < _hl[0]                             # the circle pays for itself


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
