# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo>=0.24.0",
#     "numpy",
#     "plotly",
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
    import plotly.graph_objects as go
    import pytest
    from scipy.linalg import expm

    return expm, go, mo, np, pytest


@app.cell
def _(mo):
    mo.md(r"""
    # Euler's Rotation Theorem, as a Statement About Kernels

    **The theorem.** Every rotation of 3-space fixes an axis. Compose as many
    rotations as you like about as many different axes as you like — the result is
    still a single rotation about *some* axis, and every point on that axis stays
    exactly where it was.

    Stated that way it sounds like a fact about rigid bodies. It is really a fact
    about a **null space**, and that is the version worth carrying:

    $$\text{the axis} \;=\; \ker W \quad \text{for the skew matrix } W \text{ generating the rotation}$$

    This notebook builds that identification and then shows what it costs you in
    odd dimensions. It is the geometric companion to
    [`intransitivity.py`](https://github.com/josephsmann/molab/blob/main/notebooks/intransitivity.py),
    where the same leftover direction appears as the skill blend that generates no
    matchup — there it looks like lost information, here it looks like an axis.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. Every $3\times3$ skew matrix is a cross product

    A $3\times3$ skew-symmetric matrix has $\binom{3}{2} = 3$ free entries, and
    $\mathbb{R}^3$ has 3 coordinates. That numerical coincidence is not a coincidence:
    the two are the same object. The **hat map** carries a vector to a matrix,

    $$\hat{w} = \begin{bmatrix} 0 & -w_3 & w_2 \\ w_3 & 0 & -w_1 \\ -w_2 & w_1 & 0 \end{bmatrix},
    \qquad \hat{w}\,v = w \times v$$

    so *every* $3\times3$ skew matrix acts by taking the cross product with one fixed
    vector. Nothing else is available to it.
    The map runs both ways. Going back — matrix to vector — is the **vee map**, written
    with a check instead of a hat, and it just reads the three entries off again:

    $$W^\vee = \big(W_{32},\; W_{13},\; W_{21}\big), \qquad
    \big(\hat{w}\big)^{\vee} = w, \qquad \widehat{\big(W^{\vee}\big)} = W$$

    The two are **mutually inverse**: hat then vee returns the vector you started with, and
    vee then hat returns the matrix you started with. Neither direction loses anything,
    which is the real content — a $3$-vector and a $3\times3$ skew matrix are the same three
    numbers in different clothes.

    That is a coincidence of dimension, not a general fact. A $d \times d$ skew matrix has
    $\binom{d}{2}$ independent entries while a vector has $d$, and $\binom{d}{2} = d$ only
    at $d = 3$. Two dimensions have too few skew matrices to go round; four have far too
    many. This is why the cross product exists in three dimensions and nowhere else.

    The axis falls out immediately. Since $w \times w = 0$,

    $$\hat{w}\,w = 0 \qquad\Longrightarrow\qquad w \in \ker \hat{w}$$

    and because $\hat{w}$ has rank 2 — skew rank is always even, and it is not zero
    unless $w = 0$ — that kernel is exactly one-dimensional. **The generator's null
    space is a line, and that line is the axis.**
    """)
    return


@app.cell
def _(np):
    def hat(w):
        """R^3 -> so(3). The unique skew matrix with hat(w) @ v == cross(w, v)."""
        _w = np.asarray(w, dtype=float)
        return np.array(
            [
                [0.0, -_w[2], _w[1]],
                [_w[2], 0.0, -_w[0]],
                [-_w[1], _w[0], 0.0],
            ]
        )

    def vee(W):
        """so(3) -> R^3, the inverse of hat."""
        _W = np.asarray(W, dtype=float)
        return np.array([_W[2, 1], _W[0, 2], _W[1, 0]])

    def unit(v):
        """Normalise, leaving the zero vector alone."""
        _v = np.asarray(v, dtype=float)
        _n = float(np.linalg.norm(_v))
        return _v / _n if _n > 0 else _v

    def axis_of(W, tol=1e-9):
        """The kernel direction of a 3x3 skew matrix -- i.e. its rotation axis."""
        _U, _S, _Vt = np.linalg.svd(np.asarray(W, dtype=float))
        if _S[-1] > tol:
            return None
        return _Vt[-1]

    return axis_of, hat, unit, vee


@app.cell
def _(hat, mo, np, vee):
    _rng = np.random.default_rng(0)
    _w, _v = _rng.normal(size=3), _rng.normal(size=3)
    _W = hat(_w)
    _S = _rng.normal(size=(3, 3))
    _S = _S - _S.T          # an independent skew matrix, to test the other direction

    mo.md(
        f"""
    **Check, on a random $w$ and $v$:**

    | claim | result |
    |---|---|
    | $\\hat{{w}}$ is skew | `{np.allclose(_W, -_W.T)}` |
    | $\\hat{{w}}\\,v = w \\times v$ | `{np.allclose(_W @ _v, np.cross(_w, _v))}` |
    | $\\hat{{w}}\\,w = 0$ | `{np.allclose(_W @ _w, 0)}` |
    | $\\operatorname{{rank}} \\hat{{w}} = 2$ | `{np.linalg.matrix_rank(_W) == 2}` |
    | $\\dim \\ker \\hat{{w}} = 1$ | `{3 - np.linalg.matrix_rank(_W) == 1}` |
    | $(\\hat{{w}})^{{\\vee}} = w$ | `{np.allclose(vee(_W), _w)}` |
        | $\\widehat{{(W^{{\\vee}})}} = W$, starting from a random skew $W$ | `{np.allclose(hat(vee(_S)), _S)}` |

    Three numbers in, three numbers out, and the vector you put in is the direction
    the matrix annihilates.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. Exponentiating the generator gives the rotation

    A skew matrix is an *infinitesimal* rotation — the velocity of a turning motion.
    Integrating it produces the finite rotation, and for $3\times3$ the integral has a
    closed form, **Rodrigues' formula**:

    $$\exp(\theta \hat{n}) = I + \sin\theta\, \hat{n} + (1 - \cos\theta)\, \hat{n}^2,
    \qquad \|n\| = 1$$

    Two things to notice, both of which the sliders below let you confirm by eye:

    - $\exp(\theta\hat{n})$ is **orthogonal with determinant $+1$** — a genuine rotation,
      no stretching, no reflection.
    - It **fixes $n$**. That is inherited directly from $\hat n\,n = 0$: if the generator
      annihilates a direction, the flow it generates leaves that direction alone forever.

    So the kernel of the generator is the fixed axis of the rotation. Euler's theorem
    is the observation that in three dimensions there is always such a direction to be
    found — which is really the statement that **skew rank is even, so odd dimensions
    always leave one over.**
    """)
    return


@app.cell
def _(hat, np, unit):
    def rodrigues(axis, theta):
        """Rotation by theta radians about `axis`, via exp(theta * hat(n))."""
        _K = hat(unit(axis))
        return (
            np.eye(3)
            + np.sin(theta) * _K
            + (1.0 - np.cos(theta)) * (_K @ _K)
        )

    def spherical(azimuth_deg, elevation_deg):
        """Unit vector from azimuth/elevation in degrees."""
        _a, _e = np.radians(azimuth_deg), np.radians(elevation_deg)
        return np.array(
            [np.cos(_e) * np.cos(_a), np.cos(_e) * np.sin(_a), np.sin(_e)]
        )

    return rodrigues, spherical


@app.cell
def _(mo):
    axis_azimuth = mo.ui.slider(0, 360, step=1, value=35, label="axis azimuth (°)")
    axis_elevation = mo.ui.slider(-89, 89, step=1, value=32, label="axis elevation (°)")
    turn_angle = mo.ui.slider(0, 360, step=1, value=110, label="rotation angle $\\theta$ (°)")
    probe_azimuth = mo.ui.slider(0, 360, step=1, value=210, label="test vector azimuth (°)")
    probe_elevation = mo.ui.slider(-89, 89, step=1, value=-15, label="test vector elevation (°)")

    euler_controls = mo.ui.dictionary(
        {
            "axis_az": axis_azimuth,
            "axis_el": axis_elevation,
            "turn": turn_angle,
            "probe_az": probe_azimuth,
            "probe_el": probe_elevation,
        }
    )
    euler_controls.vstack()
    return (euler_controls,)


@app.cell
def _(euler_controls, np, rodrigues, spherical):
    n_axis = spherical(euler_controls["axis_az"].value, euler_controls["axis_el"].value)
    v_probe = spherical(euler_controls["probe_az"].value, euler_controls["probe_el"].value)
    theta_now = np.radians(euler_controls["turn"].value)

    R_now = rodrigues(n_axis, theta_now)
    v_rotated = R_now @ v_probe

    # split the probe into the part along the axis and the part orbiting it
    v_parallel = float(v_probe @ n_axis) * n_axis
    v_perp = v_probe - v_parallel

    # the full orbit the probe sweeps as theta runs all the way round
    orbit = np.array([rodrigues(n_axis, _t) @ v_probe for _t in np.linspace(0, 2 * np.pi, 160)])
    return R_now, n_axis, orbit, v_parallel, v_probe, v_rotated


@app.cell
def _(R_now, go, n_axis, np, orbit, v_parallel, v_probe, v_rotated):
    def _seg(p0, p1, color, name, width=6, dash=None):
        return go.Scatter3d(
            x=[p0[0], p1[0]],
            y=[p0[1], p1[1]],
            z=[p0[2], p1[2]],
            mode="lines",
            line=dict(color=color, width=width, dash=dash),
            name=name,
        )

    _origin = np.zeros(3)
    _fig = go.Figure(
        [
            # the axis: everything on this line is fixed
            _seg(-1.6 * n_axis, 1.6 * n_axis, "#111827", "axis  ker(W)", width=8),
            # the orbit swept by the probe vector
            go.Scatter3d(
                x=orbit[:, 0],
                y=orbit[:, 1],
                z=orbit[:, 2],
                mode="lines",
                line=dict(color="#9ca3af", width=3),
                name="orbit of v",
            ),
            _seg(_origin, v_probe, "#2563eb", "v"),
            _seg(_origin, v_rotated, "#dc2626", "R v"),
            # the component along the axis -- unmoved
            _seg(_origin, v_parallel, "#059669", "component along axis (fixed)", width=9),
            _seg(v_parallel, v_probe, "#93c5fd", "perpendicular part", width=4, dash="dot"),
            _seg(v_parallel, v_rotated, "#fca5a5", "…after rotating", width=4, dash="dot"),
            go.Scatter3d(
                x=[n_axis[0]],
                y=[n_axis[1]],
                z=[n_axis[2]],
                mode="markers",
                marker=dict(size=5, color="#111827"),
                name="n  (R n = n)",
            ),
        ]
    )
    _fig.update_layout(
        scene=dict(
            xaxis=dict(range=[-1.7, 1.7], title="x"),
            yaxis=dict(range=[-1.7, 1.7], title="y"),
            zaxis=dict(range=[-1.7, 1.7], title="z"),
            aspectmode="cube",
        ),
        margin=dict(l=0, r=0, t=30, b=0),
        height=520,
        legend=dict(orientation="h", y=-0.05),
        title=f"‖Rn − n‖ = {np.linalg.norm(R_now @ n_axis - n_axis):.2e}   —   the axis does not move",
    )
    _fig
    return


@app.cell
def _(R_now, mo, n_axis, np, v_probe, v_rotated):
    mo.md(f"""
    | quantity | value |
    |---|---|
    | $R^\\top R = I$ (orthogonal) | `{np.allclose(R_now.T @ R_now, np.eye(3))}` |
    | $\\det R$ | `{np.linalg.det(R_now):+.6f}` |
    | $\\|R n - n\\|$ — the axis is fixed | `{np.linalg.norm(R_now @ n_axis - n_axis):.2e}` |
    | $\\|Rv\\| - \\|v\\|$ — lengths preserved | `{np.linalg.norm(v_rotated) - np.linalg.norm(v_probe):+.2e}` |
    | component of $v$ along the axis, before | `{float(v_probe @ n_axis):+.4f}` |
    | component of $Rv$ along the axis, after | `{float(v_rotated @ n_axis):+.4f}` |

    Drag the scene to look along the axis. The blue and red vectors swing; the green
    stub along the axis never budges no matter what $\\theta$ does — and the two dotted
    segments, the perpendicular parts, have equal length and differ only by the turn.
    The rotation does nothing whatsoever to $\\ker W$, and acts as a plain $2$D rotation
    on the plane perpendicular to it.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Reading it backwards: find the axis from the rotation

    Above we built $R$ from an axis we chose. In practice you meet $R$ first — a
    measured orientation, or a product of many rotations — and want its axis. Euler's
    theorem promises one exists, and the proof is a two-line eigenvalue argument.

    $R$ is orthogonal, so its eigenvalues have modulus 1; it is real, so complex
    eigenvalues arrive in conjugate pairs; and $\det R = +1$ is their product. In three
    dimensions the only way to satisfy all three at once is a spectrum
    $\{1,\, e^{i\theta},\, e^{-i\theta}\}$. **The eigenvalue $1$ is forced**, and its
    eigenvector is the axis.

    This is where the odd dimension does the work. Pair up the complex eigenvalues and
    one real eigenvalue is always left standing — the same parity argument as "skew rank
    is even". In $SO(4)$ the eigenvalues can pair off completely as
    $\{e^{i\alpha}, e^{-i\alpha}, e^{i\beta}, e^{-i\beta}\}$, no eigenvalue $1$ is
    forced, and a generic 4D rotation **has no fixed axis at all** — it turns two
    independent planes at once.

    Euler's theorem is not a fact about rotations. It is a fact about *three*.
    """)
    return


@app.cell
def _(axis_of, np):
    def axis_from_rotation(R, tol=1e-9):
        """Recover the fixed axis of R in SO(3) as its eigenvalue-1 eigenvector."""
        _R = np.asarray(R, dtype=float)
        _vals, _vecs = np.linalg.eig(_R)
        _k = int(np.argmin(np.abs(_vals - 1.0)))
        if abs(_vals[_k] - 1.0) > 1e-6:
            return None
        return np.real(_vecs[:, _k]) / np.linalg.norm(np.real(_vecs[:, _k]))

    def axis_via_generator(R, expm_fn, tol=1e-9):
        """Same axis, reached the other way: take the log, then read its kernel."""
        from scipy.linalg import logm

        _W = np.real(logm(np.asarray(R, dtype=float)))
        return axis_of(0.5 * (_W - _W.T), tol=tol)

    return axis_from_rotation, axis_via_generator


@app.cell
def _(axis_from_rotation, axis_via_generator, expm, mo, np, rodrigues, unit):
    # Compose three unrelated rotations; Euler says the product still has an axis.
    _r = np.random.default_rng(7)
    _R1 = rodrigues(_r.normal(size=3), 0.7)
    _R2 = rodrigues(_r.normal(size=3), -1.9)
    _R3 = rodrigues(_r.normal(size=3), 2.4)
    _R = _R3 @ _R2 @ _R1

    _n_eig = axis_from_rotation(_R)
    _n_log = axis_via_generator(_R, expm)
    _agree = min(
        float(np.linalg.norm(unit(_n_eig) - unit(_n_log))),
        float(np.linalg.norm(unit(_n_eig) + unit(_n_log))),
    )

    mo.md(
        f"""
    **Three rotations about three unrelated axes, composed.** The product is still a
    rotation, and Euler promises it fixes something:

    | | |
    |---|---|
    | eigenvalues of $R_3R_2R_1$ | `{np.round(np.linalg.eigvals(_R), 3).tolist()}` |
    | exactly one is $+1$ | `{int(np.sum(np.abs(np.linalg.eigvals(_R) - 1.0) < 1e-9))}` |
    | axis, from the eigenvector | `{np.round(_n_eig, 4).tolist()}` |
    | axis, from $\\ker \\log R$ | `{np.round(_n_log, 4).tolist()}` |
    | the two routes agree (up to sign) | `{_agree < 1e-7}` |
    | $\\|R n - n\\|$ | `{np.linalg.norm(_R @ _n_eig - _n_eig):.2e}` |

    Note the spectrum: one real $+1$ and a conjugate pair on the unit circle, exactly as
    the parity argument requires.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. Four dimensions, where the axis disappears

    The claim that the parity argument is doing all the work is easy to check: repeat the
    construction one dimension up and watch the fixed axis fail to exist.
    """)
    return


@app.cell
def _(expm, mo, np):
    def random_rotation(d, seed=0, scale=1.0):
        """A rotation in SO(d), built as exp of a random skew matrix."""
        _r = np.random.default_rng(seed)
        _M = _r.normal(size=(d, d)) * scale
        return expm(_M - _M.T)

    _rows = []
    for _d in (2, 3, 4, 5, 6, 7):
        _R = random_rotation(_d, seed=_d)
        _vals = np.linalg.eigvals(_R)
        _n_fixed = int(np.sum(np.abs(_vals - 1.0) < 1e-8))
        _rows.append(
            f"| {_d} | `{np.allclose(_R.T @ _R, np.eye(_d))}` | "
            f"`{np.linalg.det(_R):+.3f}` | **{_n_fixed}** | "
            f"{'axis guaranteed' if _d % 2 else 'no fixed axis'} |"
        )

    mo.md(
        "**Generic rotations in $SO(d)$, and how many directions they fix**\n\n"
        "| $d$ | orthogonal | $\\det$ | eigenvalues equal to $1$ | |\n|---|---|---|---|---|\n"
        + "\n".join(_rows)
        + """

    Odd $d$ always fixes at least one direction; even $d$ generically fixes none. A
    $4$D rotation spins two perpendicular planes simultaneously at unrelated rates, and
    there is nowhere left over to stand still.

    So "every rotation has an axis" is a three-dimensional accident, and the accident is
    parity. Same parity, same argument, as **skew rank is always even**: pair the
    dimensions off two at a time and an odd dimension leaves exactly one unpaired.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. The same leftover direction, in a different costume

    In [`intransitivity.py`](https://github.com/josephsmann/molab/blob/main/notebooks/intransitivity.py)
    players carry $d$ skills and the matchup term is $m(i,j) = a_i^\top C\, a_j$ with $C$
    skew and $d \times d$. At $d = 3$ that $C$ is *exactly* the object studied here: it is
    $\hat{w}$ for some $w$, its rank is 2, and it annihilates a line.

    The two readings of that line:

    | | rotations | ratings |
    |---|---|---|
    | the object | $\hat{w}$, generator of a turn | $C$, the matchup form |
    | $\ker$ | the **axis** — what stays put | a skill blend generating **no matchup** |
    | the rank-2 part | the plane that spins | the **style circle** |
    | why 1 dimension is left | $3$ is odd | $3$ is odd |

    It is the identical theorem. Whether the leftover direction reads as a feature or a
    loss depends only on what you wanted:

    - Wanting to *turn* something, the axis is the useful part — it is the handle, the
      thing you can hold while everything else moves.
    - Wanting to *measure* matchups, that direction is a blind spot: no head-to-head
      result can ever reveal a player's coordinate along it.

    Neither is more correct. A kernel is a statement about what an operator cannot see,
    and whether "cannot see" is good news depends entirely on whether you wanted to look.
    """)
    return


@app.cell
def _(
    axis_from_rotation,
    axis_of,
    expm,
    hat,
    np,
    pytest,
    rodrigues,
    spherical,
    unit,
    vee,
):
    def test_hat_is_the_cross_product():
        _r = np.random.default_rng(0)
        for _ in range(5):
            _w, _v = _r.normal(size=3), _r.normal(size=3)
            assert np.allclose(hat(_w) @ _v, np.cross(_w, _v))
            assert np.allclose(hat(_w), -hat(_w).T)
            assert np.allclose(vee(hat(_w)), _w)          # hat then vee
            _M = _r.normal(size=(3, 3))
            _S = _M - _M.T
            assert np.allclose(hat(vee(_S)), _S)          # vee then hat -- the other side

    def test_generator_annihilates_its_own_vector():
        _r = np.random.default_rng(1)
        for _ in range(5):
            _w = _r.normal(size=3)
            assert np.allclose(hat(_w) @ _w, 0.0)
            assert np.linalg.matrix_rank(hat(_w)) == 2
            assert np.allclose(np.abs(unit(axis_of(hat(_w)))), np.abs(unit(_w)))

    def test_rodrigues_matches_the_matrix_exponential():
        _r = np.random.default_rng(2)
        for _ in range(5):
            _n = unit(_r.normal(size=3))
            _theta = float(_r.uniform(-np.pi, np.pi))
            assert np.allclose(rodrigues(_n, _theta), expm(_theta * hat(_n)))

    def test_rotations_are_orthogonal_and_fix_their_axis():
        _r = np.random.default_rng(3)
        for _ in range(5):
            _n = unit(_r.normal(size=3))
            _R = rodrigues(_n, float(_r.uniform(0, 2 * np.pi)))
            assert np.allclose(_R.T @ _R, np.eye(3))
            assert np.linalg.det(_R) == pytest.approx(1.0, abs=1e-9)
            assert np.allclose(_R @ _n, _n)

    def test_composition_of_rotations_still_has_an_axis():
        """Euler's theorem proper: the product is a rotation, and it fixes something."""
        _r = np.random.default_rng(4)
        _R = np.eye(3)
        for _ in range(4):
            _R = rodrigues(_r.normal(size=3), float(_r.uniform(0, 2 * np.pi))) @ _R
        _n = axis_from_rotation(_R)
        assert _n is not None
        assert np.allclose(_R @ _n, _n, atol=1e-9)
        assert int(np.sum(np.abs(np.linalg.eigvals(_R) - 1.0) < 1e-8)) == 1

    def test_odd_dimensions_fix_an_axis_and_even_ones_need_not():
        _r = np.random.default_rng(5)
        for _d in (3, 5, 7):
            _M = _r.normal(size=(_d, _d))
            _R = expm(_M - _M.T)
            assert int(np.sum(np.abs(np.linalg.eigvals(_R) - 1.0) < 1e-8)) >= 1
        for _d in (4, 6):
            _M = _r.normal(size=(_d, _d))
            _R = expm(_M - _M.T)
            assert int(np.sum(np.abs(np.linalg.eigvals(_R) - 1.0) < 1e-8)) == 0

    def test_spherical_returns_unit_vectors():
        for _az, _el in [(0, 0), (35, 32), (210, -15), (359, 89)]:
            assert np.linalg.norm(spherical(_az, _el)) == pytest.approx(1.0, abs=1e-12)

    return


if __name__ == "__main__":
    app.run()
