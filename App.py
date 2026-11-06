"""
app.py  -  Streamlit front-end for root_methods.py

Run:   streamlit run app.py
"""
import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import root_methods as rm

st.set_page_config(page_title="Root Finder", page_icon="📐", layout="wide")

BLUE, RED, GREEN, ORANGE, GRAY = "#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#7f7f7f"

METHODS = {
    "bisection": "Bisection",
    "false_position": "False-Position",
    "newton": "Newton-Raphson",
    "secant": "Secant",
    "fixed_point": "Fixed-Point Iteration",
}

EXAMPLES = {
    "Custom": None,
    "x³ − 7x + 1   (the slides)": dict(f="x^3 - 7*x + 1", a="0", b="1", x0="0", x1="1", dec=3),
    "cos(x) − x": dict(f="cos(x) - x", a="0", b="1", x0="1", x1="0", dec=8),
    "eˣ − 3x": dict(f="exp(x) - 3*x", a="0", b="1", x0="0.5", x1="1", dec=6),
    "ln(x) + x − 2": dict(f="log(x) + x - 2", a="1", b="2", x0="1.5", x1="2", dec=8),
    "⚠ Pole demo: 1/x on [−1, 2]": dict(f="1/x", a="-1", b="2", x0="0.5", x1="1", dec=4),
    "⚠ Domain demo: ln(x) on [−1, 2]": dict(f="log(x)", a="-1", b="2", x0="3", x1="2", dec=4),
}

DEFAULTS = dict(example="x³ − 7x + 1   (the slides)", method="bisection", ftext="x^3 - 7*x + 1",
                a_txt="0", b_txt="1", x0_txt="0", x1_txt="1", dec=3, dtext="", gtext="(x^3 + 1)/7",
                fp_mode="auto", max_iter=500)
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)


def load_example():
    ex = EXAMPLES[st.session_state["example"]]
    if ex:
        st.session_state.update(ftext=ex["f"], a_txt=ex["a"], b_txt=ex["b"],
                                x0_txt=ex["x0"], x1_txt=ex["x1"], dec=ex["dec"])


# =====================================================================
#  helpers
# =====================================================================
def show_plot(fig):
    """st.plotly_chart that works on old and new Streamlit versions."""
    try:
        st.plotly_chart(fig, width="stretch")
    except TypeError:
        st.plotly_chart(fig, use_container_width=True)


def show_df(df, **kw):
    """st.dataframe that works on old and new Streamlit versions."""
    try:
        st.dataframe(df, hide_index=True, width="stretch", **kw)
    except TypeError:
        st.dataframe(df, hide_index=True, use_container_width=True, **kw)


def problem(title, text, hint=None):
    body = f"**{title}**\n\n{text}"
    if hint:
        body += f"\n\n💡 {hint}"
    st.error(body)


def safe(fn, x):
    try:
        return fn(x)
    except (rm.MathError, ArithmeticError):
        return float("nan")


def x_range(points, pad=0.25):
    pts = [p for p in points if p is not None and math.isfinite(p) and abs(p) < 1e6]
    if not pts:
        return -5.0, 5.0
    lo, hi = min(pts), max(pts)
    if hi - lo < 1e-9:
        lo, hi = lo - 1, hi + 1
    p = (hi - lo) * pad
    return lo - p, hi + p


def y_range(curve, extra):
    vals = np.array([v for v in curve if math.isfinite(v)] + [0.0])
    lo, hi = np.percentile(vals, 1), np.percentile(vals, 99)
    ex = [v for v in extra if math.isfinite(v)] + [0.0]
    lo, hi = min(lo, min(ex)), max(hi, max(ex))
    pad = 0.12 * (hi - lo or 1.0)
    return lo - pad, hi + pad


def base_layout(fig, title, xlab="x", ylab="f(x)", height=500):
    fig.update_layout(template="plotly_white", height=height, title=title,
                      xaxis_title=xlab, yaxis_title=ylab, hovermode="closest",
                      legend=dict(orientation="h", y=-0.18), margin=dict(l=40, r=20, t=60, b=40))
    return fig


# =====================================================================
#  figures
# =====================================================================
def fig_function(res, f, key, start_pts, max_lines=12):
    lo, hi = x_range(start_pts + res.xs)
    X = np.linspace(lo, hi, 600)
    Y = np.array([safe(f, x) for x in X])
    pts_x = [x for x in res.xs if math.isfinite(x) and lo <= x <= hi]
    pts_y = [safe(f, x) for x in pts_x]
    ylo, yhi = y_range(Y, pts_y)

    fig = go.Figure()
    fig.add_hline(y=0, line_color=GRAY, line_width=1)
    fig.add_trace(go.Scatter(x=X, y=np.where(np.isfinite(Y), Y, np.nan), mode="lines",
                             name="f(x)", line=dict(color=BLUE, width=3), connectgaps=False))

    rows = res.rows[:max_lines]
    sx, sy = [], []                                     # step lines (one trace)
    if key in ("bisection",):
        a0, b0 = res.rows[0][1], res.rows[0][2]
        fig.add_vrect(x0=min(a0, b0), x1=max(a0, b0), fillcolor=ORANGE, opacity=0.08,
                      line_width=0, annotation_text="initial interval", annotation_position="top left")
        for r in rows:
            sx += [r[3], r[3], None]
            sy += [0, safe(f, r[3]), None]
    elif key == "false_position":
        for r in rows:
            sx += [r[1], r[2], None]
            sy += [safe(f, r[1]), safe(f, r[2]), None]
    elif key == "newton":
        for r in rows:
            fxp = safe(f, r[1])
            sx += [r[1], r[1], r[2], None]
            sy += [0, fxp, 0, None]
    elif key == "secant":
        for r in rows:
            sx += [r[1], r[2], r[3], None]
            sy += [safe(f, r[1]), safe(f, r[2]), 0, None]
    if sx:
        label = {"bisection": "midpoints", "false_position": "chords",
                 "newton": "tangent steps", "secant": "secant lines"}[key]
        fig.add_trace(go.Scatter(x=sx, y=sy, mode="lines", name=label,
                                 line=dict(color=ORANGE, width=1.5, dash="dot")))

    if pts_x:
        n = len(pts_x)
        fig.add_trace(go.Scatter(
            x=pts_x, y=pts_y, mode="markers+text" if n <= 14 else "markers", name="iterates xi",
            text=[str(i) for i in range(len(res.xs)) if lo <= res.xs[i] <= hi] if n <= 14 else None,
            textposition="top center",
            marker=dict(size=9, color=list(range(n)), colorscale="Viridis",
                        line=dict(color="white", width=1)),
            hovertemplate="x = %{x:.10g}<br>f(x) = %{y:.4g}<extra>iterate</extra>"))
    if res.status == "ok" and res.root is not None:
        fig.add_trace(go.Scatter(x=[res.root], y=[0], mode="markers", name="root xr",
                                 marker=dict(size=16, color=RED, symbol="star",
                                             line=dict(color="white", width=1)),
                                 hovertemplate=f"xr = {res.root:.10g}<extra>root</extra>"))
    fig.update_xaxes(range=[lo, hi])
    fig.update_yaxes(range=[ylo, yhi])
    return base_layout(fig, "Function and iterations")


def fig_cobweb(res, g, x0, max_pts=40):
    limit = 50 * (1 + abs(x0))
    rows = []
    for r in res.rows[:max_pts]:
        if abs(r[2]) > limit:                      # diverging: do not stretch the axes to 10^15
            break
        rows.append(r)
    lo, hi = x_range([x0] + [r[2] for r in rows[:25]])
    X = np.linspace(lo, hi, 600)
    Y = np.array([safe(g, x) for x in X])
    path = [(x0, x0)]                              # a cobweb starts ON the line y = x
    for r in rows:
        path += [(r[1], r[2]), (r[2], r[2])]
    ylo, yhi = y_range(Y, [p[1] for p in path])
    ylo, yhi = min(ylo, lo), max(yhi, hi)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="y = x",
                             line=dict(color=GRAY, dash="dash")))
    fig.add_trace(go.Scatter(x=X, y=Y, mode="lines", name="y = g(x)", line=dict(color=BLUE, width=3)))
    fig.add_trace(go.Scatter(x=[p[0] for p in path], y=[p[1] for p in path], mode="lines+markers",
                             name="iteration path", line=dict(color=ORANGE, width=1.6),
                             marker=dict(size=5)))
    fig.add_trace(go.Scatter(x=[x0], y=[x0], mode="markers", name="x0",
                             marker=dict(size=11, color=GREEN, symbol="diamond")))
    if res.status == "ok" and res.root is not None:
        fig.add_trace(go.Scatter(x=[res.root], y=[res.root], mode="markers", name="fixed point",
                                 marker=dict(size=16, color=RED, symbol="star")))
    fig.update_xaxes(range=[lo, hi])
    fig.update_yaxes(range=[ylo, yhi])
    return base_layout(fig, "Cobweb diagram - the staircase must lead INTO the intersection",
                       "x", "g(x)")


def fig_convergence(res):
    if not res.diffs:
        return None
    it = [d[0] for d in res.diffs]
    val = [max(d[1], 1e-17) for d in res.diffs]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=it, y=val, mode="lines+markers", name="|xi − x(i−1)|",
                             line=dict(color=BLUE, width=3), marker=dict(size=8)))
    fig.add_hline(y=res.tol, line_color=RED, line_dash="dash",
                  annotation_text=f"tolerance 0.5·10^-{res.decimals}", annotation_position="top right")
    fig.update_yaxes(type="log", exponentformat="power")
    return base_layout(fig, "Convergence (log scale) - the curve must fall below the red line",
                       "iteration i", "|xi − x(i−1)|", height=420)


def fig_intervals(res):
    ia, ib, im = [], [], []
    for r in res.rows:
        ia += [r[1], r[2], None]
        ib += [r[0], r[0], None]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ia, y=ib, mode="lines+markers", name="[a, b]",
                             line=dict(color=BLUE, width=3), marker=dict(size=5)))
    fig.add_trace(go.Scatter(x=[r[3] for r in res.rows], y=[r[0] for r in res.rows],
                             mode="markers", name="xi", marker=dict(size=8, color=ORANGE)))
    if res.status == "ok" and res.root is not None:
        fig.add_vline(x=res.root, line_color=RED, line_dash="dot")
    fig.update_yaxes(autorange="reversed")
    return base_layout(fig, "How the bracket [a, b] shrinks", "x", "iteration i", height=450)


# =====================================================================
#  table
# =====================================================================
def build_tables(res):
    dg = max(9, res.decimals + 3)
    shown, raw = [], []
    for r in res.rows:
        shown.append([str(r[0])] + [rm.format_number(v, dg, 1e-5 if j + 1 in res.sci_cols else None)
                                    for j, v in enumerate(r[1:])])
        raw.append(list(r))
    return pd.DataFrame(shown, columns=res.columns), pd.DataFrame(raw, columns=res.columns)


def style_table(df, res):
    last, xc = len(df) - 1, df.columns[res.x_col]

    def row_style(row):
        if row.name == last and res.status == "ok":
            return ["color:#d62728;font-weight:700"] * len(row)
        return ["color:#1f77b4" if c == xc else "" for c in row.index]

    return df.style.apply(row_style, axis=1).set_properties(**{"text-align": "center"})


# =====================================================================
#  results
# =====================================================================
EXPLAIN = {
    "bisection": (r"x_i = \dfrac{a+b}{2}",
                  "Keep the half-interval where f changes sign. Always converges when f is continuous "
                  "and f(a), f(b) have opposite signs."),
    "false_position": (r"x_i = \dfrac{a\,f(b) - b\,f(a)}{f(b) - f(a)}",
                       "Same bracketing idea, but the new point is where the chord crosses the x-axis."),
    "newton": (r"x_i = x_{i-1} - \dfrac{f(x_{i-1})}{f'(x_{i-1})}",
               "Very fast near a simple root, but fails if f'(x) = 0 or the guess is poor."),
    "secant": (r"x_i = x_{i-1} - \dfrac{(x_{i-1}-x_{i-2})\,f(x_{i-1})}{f(x_{i-1}) - f(x_{i-2})}",
               "Newton without a derivative; fails if f(x_{i-1}) = f(x_{i-2})."),
    "fixed_point": (r"x_i = g(x_{i-1}), \qquad \text{converges if } |g'(x)| < 1",
                    "Rewrite f(x)=0 as x=g(x). The choice of g decides whether it converges."),
}


def show_result(res, f, key, start_pts, g=None, x0=None, dec=3):
    # ---- status ----
    if res.status == "ok":
        st.success(f"**The root will be xr ≈ {res.root:.{dec}f} correct to {dec} decimal places.**"
                   + (f"  \n{res.message}" if res.message else ""))
    elif res.status == "not_root":
        problem("Not a root", res.message,
                "Plot the function (tab Graph), then choose an interval/guess around a real crossing of the x-axis.")
    else:
        titles = {"error": "The calculation failed", "diverged": "The iteration does not converge",
                  "max_iter": "Maximum iterations reached"}
        hints = {"error": "Pick initial values where f (and f') are defined, away from poles and domain limits.",
                 "diverged": "Try a starting value closer to the root (see the graph), use Bisection "
                             "(always converges when f changes sign), or - for fixed-point - another g(x).",
                 "max_iter": "Increase 'Maximum iterations' in the sidebar, use fewer decimals, or try another method."}
        problem(titles[res.status], res.message, hints[res.status])
        if res.rows:
            st.caption(f"Showing the {len(res.rows)} iteration(s) computed before it stopped.")
    for w in res.warnings:
        st.warning(w)
    if res.status == "ok" and res.root is not None and abs(res.root) >= 1e4 and dec >= 11:
        st.warning("The answer is large, so 11-12 decimals may be beyond double-precision accuracy.")

    # ---- metrics ----
    if res.rows:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Root xr", f"{res.root:.{dec}f}" if res.status == "ok" else "—")
        c2.metric("Iterations", len(res.rows))
        if res.fx_root is not None:
            c3.metric("|f(xr)|", f"{abs(res.fx_root):.3e}")
        else:
            c3.metric("|f(xr)|", "—")
        last = res.diffs[-1][1] if res.diffs else None
        c4.metric("Last |xi − x(i−1)|", f"{last:.3e}" if last is not None else "—",
                  help=f"Stops when ≤ 0.5·10^-{dec} = {res.tol:.1e}")
    if not res.rows:
        return

    # ---- tabs ----
    tabs = ["📋 Table", "📈 Graph", "📉 Convergence"]
    if key in ("bisection", "false_position"):
        tabs.append("📏 Interval")
    tabs.append("ℹ️ Method")
    T = st.tabs(tabs)
    shown, raw = build_tables(res)
    with T[0]:
        show_df(style_table(shown, res), height=min(36 * len(shown) + 40, 620))
        st.download_button("⬇ Download table (CSV)", raw.to_csv(index=False).encode(),
                           f"{key}_iterations.csv", "text/csv")
    with T[1]:
        if key == "fixed_point":
            show_plot(fig_cobweb(res, g, x0))
            if f is not None:
                show_plot(fig_function(res, f, key, start_pts))
        else:
            show_plot(fig_function(res, f, key, start_pts))
    with T[2]:
        fc = fig_convergence(res)
        if fc is not None:
            show_plot(fc)
        else:
            st.info("Not enough iterations to plot.")
    k = 3
    if key in ("bisection", "false_position"):
        with T[3]:
            show_plot(fig_intervals(res))
        k = 4
    with T[k]:
        tex, text = EXPLAIN[key]
        st.latex(tex)
        st.write(text)
        st.write(f"**Stopping rule:** keep iterating while  |xᵢ − xᵢ₋₁| > 0.5 × 10⁻{dec} = {res.tol:.1e}.")
        if key == "bisection" and len(res.rows) > 1:
            n = rm.bisection_iterations(res.rows[0][1], res.rows[0][2], dec)
            st.write(f"The interval halves each step, so about **{n}** iterations are needed for this precision.")


# =====================================================================
#  sidebar
# =====================================================================
with st.sidebar:
    st.header("⚙️ Setup")
    st.selectbox("Load an example", list(EXAMPLES), key="example", on_change=load_example)
    key = st.radio("Method", list(METHODS), format_func=METHODS.get, key="method")
    if key != "fixed_point":
        st.text_input("Equation  f(x) = 0", key="ftext",
                      help="Use x. Examples: x^3 - 7x + 1 · cos(x) - x · exp(x) = 3x · log(x) + x - 2")
    else:
        st.radio("I will give…", ["auto", "g"], key="fp_mode", horizontal=True,
                 format_func=lambda v: "f(x) = 0  (find g)" if v == "auto" else "x = g(x)")
        if st.session_state["fp_mode"] == "auto":
            st.text_input("Equation  f(x) = 0", key="ftext")
        else:
            st.text_input("Iteration function  g(x)", key="gtext")

    if key in ("bisection", "false_position"):
        c1, c2 = st.columns(2)
        c1.text_input("a", key="a_txt")
        c2.text_input("b", key="b_txt")
    elif key in ("newton", "fixed_point"):
        st.text_input("Initial guess x₀", key="x0_txt")
    else:
        c1, c2 = st.columns(2)
        c1.text_input("x₀", key="x0_txt")
        c2.text_input("x₁", key="x1_txt")
    if key == "newton":
        st.text_input("f′(x)  (leave empty = automatic)", key="dtext")

    st.slider("Required decimal places", 1, rm.MAX_DECIMALS, key="dec",
              help="Precision up to 10⁻¹²")
    with st.expander("Advanced"):
        st.number_input("Maximum iterations", 10, 5000, step=50, key="max_iter")
    st.caption("Numbers may be expressions: `pi/2`, `-sqrt(2)`, `1e-3`.")

# =====================================================================
#  main
# =====================================================================
st.title("📐 Root Finder")
st.caption("Bisection · False-Position · Newton-Raphson · Secant · Fixed-Point  —  with validation, tables and graphs")

S = st.session_state
dec, max_iter = int(S["dec"]), int(S["max_iter"])


def number(key_, label):
    return rm.parse_number(S[key_], label)


try:
    # ------------------------------------------------ parse the equation
    f = None
    g_typed = None
    if key == "fixed_point" and S["fp_mode"] == "g":
        g_typed = rm.Func(S["gtext"], name="g")
        tex = rm.latex_of(g_typed)
        st.latex("x = " + tex if tex else "x = g(x)")
    else:
        f = rm.Func(S["ftext"], name="f", allow_equals=True)
        tex = rm.latex_of(f)
        st.latex(("f(x) = " + tex + " = 0") if tex else "f(x) = 0")

    # ------------------------------------------------ bracketing methods
    if key in ("bisection", "false_position"):
        a, b = number("a_txt", "a"), number("b_txt", "b")
        info = rm.analyze_interval(f, a, b)
        if info["undefined"] and len(info["undefined"]) > 0:
            u = info["undefined"]
            st.warning(f"f is undefined at {len(u)} sample point(s) inside [{min(a, b):g}, {max(a, b):g}] "
                       f"(division by zero or domain error, e.g. near x = {u[0]:.4g}). "
                       f"The method may fail or return a wrong answer.")
        if info["poles"]:
            p = info["poles"][0]
            st.warning(f"f seems to jump through infinity (a pole) near x ∈ [{p[0]:.4g}, {p[1]:.4g}]. "
                       f"A sign change there is not a root.")
        if len(info["sign_changes"]) > 1:
            st.info(f"f changes sign {len(info['sign_changes'])} times in this interval - "
                    f"the method will find one of the roots. Narrow the interval to choose which.")
        run = rm.bisection if key == "bisection" else rm.false_position
        res = run(f, a, b, dec, max_iter)
        show_result(res, f, key, [a, b], dec=dec)

    # ------------------------------------------------ Newton
    elif key == "newton":
        x0 = number("x0_txt", "x₀")
        df, kind, dtex = rm.make_derivative(f, S["dtext"], probe=x0)
        st.caption(f"Derivative: **{kind}**" + (f"   f′(x) = ${dtex}$" if dtex else ""))
        res = rm.newton(f, df, x0, dec, max_iter)
        show_result(res, f, key, [x0], dec=dec)

    # ------------------------------------------------ Secant
    elif key == "secant":
        x0, x1 = number("x0_txt", "x₀"), number("x1_txt", "x₁")
        res = rm.secant(f, x0, x1, dec, max_iter)
        show_result(res, f, key, [x0, x1], dec=dec)

    # ------------------------------------------------ Fixed point
    else:
        x0 = number("x0_txt", "x₀")
        if g_typed is not None:                                   # user typed g(x)
            g, chosen = g_typed, None
            sl = rm.slope_at(g, x0)
            st.markdown(f"**Convergence test:**  |g′(x₀)| = "
                        f"{'undefined' if sl == math.inf else f'{sl:.6g}'}  →  "
                        + ("✅ converges near x₀" if sl < 1 else "❌ |g′(x₀)| ≥ 1: probably diverges"))
            if sl >= 1:
                st.warning("The test is local but a value ≥ 1 means the iteration will most likely "
                           "diverge from this x₀. It is run anyway below so you can see what happens.")
            f_for_check = None
        else:                                                     # program finds g(x)
            cands = rm.fixed_point_candidates(f, x0)
            table = pd.DataFrame({
                "#": range(1, len(cands) + 1),
                "x = g(x)": [c.label for c in cands],
                "|g′(x₀)|": [("undefined" if c.slope == math.inf else f"{c.slope:.6g}") for c in cands],
                "Verdict": ["✅ converges" if c.slope < 1 else "❌ diverges" for c in cands],
            })
            st.subheader("Convergence test of every rearrangement")
            st.caption("x = g(x) converges near x₀ when |g′(x₀)| < 1. The best one is used by default.")
            show_df(table)
            if cands[0].slope >= 1:
                problem("No convergent rearrangement from this x₀",
                        "Every form x = g(x) has |g′(x₀)| ≥ 1 here, so the iteration would diverge.",
                        "Try an initial guess closer to a root (look at the graph of f), "
                        "or type your own g(x) in the sidebar.")
                xs_ = [x0 - 2, x0 + 2]
                fig = go.Figure(go.Scatter(x=np.linspace(*xs_, 400),
                                           y=[safe(f, v) for v in np.linspace(*xs_, 400)], mode="lines",
                                           line=dict(color=BLUE, width=3), name="f(x)"))
                fig.add_hline(y=0, line_color=GRAY)
                show_plot(base_layout(fig, "f(x) around x₀ - pick x₀ near a crossing", height=380))
                st.stop()
            idx = st.selectbox("Use rearrangement #", range(1, len(cands) + 1), index=0,
                               format_func=lambda i: f"{i}.  {cands[i - 1].label}") - 1
            chosen = cands[idx]
            g, f_for_check = chosen.g, f
            if chosen.latex:
                st.latex(chosen.latex)
            if chosen.slope >= 1:
                st.warning("This rearrangement does not satisfy |g′(x₀)| < 1 and will probably diverge.")
        res = rm.fixed_point(g, x0, dec, max_iter, f=f_for_check)
        if res.status == "ok":
            st.info(f"Check at the answer: |g′(xr)| = {res.extra.get('g_slope', float('nan')):.4g} "
                    f"{'< 1 ✅ stable fixed point' if res.extra.get('g_slope', 2) < 1 else '≥ 1 ❌'}")
        show_result(res, f_for_check, key, [x0], g=g, x0=x0, dec=dec)

except rm.InputError as e:
    problem("Please check your input", str(e))
except rm.MathError as e:
    problem("Math error", str(e), "Choose values where the function is defined.")
except Exception as e:                                  # last safety net - never show a raw traceback
    problem("Unexpected problem", f"{type(e).__name__}: {e}")