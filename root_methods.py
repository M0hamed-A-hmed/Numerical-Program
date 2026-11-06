"""
root_methods.py  -  numerical core (no printing, no Streamlit).

Methods : Bisection, False-Position, Newton-Raphson, Secant, Fixed-Point.
Safety  : the equation is parsed through a whitelist (no arbitrary code), and
          every evaluation is guarded against
              - division by zero              - log / sqrt of negative numbers
              - overflow                      - complex results
              - NaN / infinity                - poles (discontinuities)
          Failures raise InputError (bad input) or MathError (bad maths) with a
          readable message, and the iteration methods return whatever was
          computed before the failure.
"""
from __future__ import annotations

import ast
import math
import re
from dataclasses import dataclass, field

MAX_DECIMALS = 12
DEFAULT_MAX_ITER = 500
BLOW_UP = 1e10


class InputError(Exception):
    """The user's input is invalid (syntax, unknown symbol, no sign change ...)."""


class MathError(Exception):
    """Mathematical failure while evaluating (division by zero, domain error ...)."""


# =====================================================================
#  1.  Safe equation parser
# =====================================================================
def _cbrt(v):
    return math.copysign(abs(v) ** (1.0 / 3.0), v)


FUNCS = {n: getattr(math, n) for n in (
    "sin", "cos", "tan", "asin", "acos", "atan", "sinh", "cosh", "tanh",
    "asinh", "acosh", "atanh", "exp", "log", "log10", "log2", "sqrt")}
FUNCS.update({
    "ln": math.log, "abs": abs, "cbrt": _cbrt, "pow": math.pow,
    "sec": lambda v: 1.0 / math.cos(v),
    "csc": lambda v: 1.0 / math.sin(v),
    "cot": lambda v: 1.0 / math.tan(v),
})
ARITY = {"log": (1, 2), "pow": (2, 2)}          # default is exactly 1 argument
CONSTS = {"pi": math.pi, "e": math.e}
_NAMESPACE = {"__builtins__": {}, **FUNCS, **CONSTS}
FUNC_LIST = ", ".join(sorted(FUNCS))


def _preprocess(text: str) -> str:
    """Friendly syntax:  x^3  7x  2(x+1)  3sin(x)  (x+1)(x-1)  x(x+1)."""
    s = (text.strip().replace("^", "**").replace("×", "*")
         .replace("−", "-").replace("÷", "/"))
    s = re.sub(r"(?<![A-Za-z_0-9.])(\d+\.?\d*|\.\d+)\s*(?=(?:x\b|\(|pi\b|[A-Za-z_]\w*\())",
               r"\1*", s)
    s = re.sub(r"\)\s*(?=[A-Za-z_(\d.])", ")*", s)
    s = re.sub(r"(?<![A-Za-z_0-9])x\s*(?=\()", "x*", s)
    return s


def split_equation(text: str) -> str:
    """'lhs = rhs'  ->  '(lhs) - (rhs)'   (so the user may type either form)."""
    if "=" not in text:
        return text
    if re.search(r"[<>!=]=|=[<>=]", text) or text.count("=") != 1:
        raise InputError("Use a single '=' sign, e.g.  x^3 = 7*x - 1")
    lhs, rhs = (p.strip() for p in text.split("="))
    if not lhs or not rhs:
        raise InputError("Both sides of '=' must contain something.")
    return f"({lhs}) - ({rhs})"


class _ToFloat(ast.NodeTransformer):
    """All numbers become floats, so 9**9**9 overflows instead of freezing."""
    def visit_Constant(self, node):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise InputError("Only numbers are allowed in the equation.")
        try:
            return ast.copy_location(ast.Constant(float(node.value)), node)
        except OverflowError:
            raise InputError("A number in the equation is too large.") from None


def _check(node, allow_x):
    if isinstance(node, ast.Expression):
        _check(node.body, allow_x)
    elif isinstance(node, ast.BinOp):
        if not isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)):
            raise InputError("Only + - * / and ^ are allowed.")
        _check(node.left, allow_x)
        _check(node.right, allow_x)
    elif isinstance(node, ast.UnaryOp):
        if not isinstance(node.op, (ast.UAdd, ast.USub)):
            raise InputError("Only unary + and - are allowed.")
        _check(node.operand, allow_x)
    elif isinstance(node, ast.Constant):
        pass
    elif isinstance(node, ast.Name):
        if node.id == "x":
            if not allow_x:
                raise InputError("x cannot be used here - enter a number such as 0.5 or pi/2.")
        elif node.id in FUNCS:
            raise InputError(f"'{node.id}' is a function - write it as {node.id}(...).")
        elif node.id not in CONSTS:
            raise InputError(
                f"Unknown symbol '{node.id}'. Use x as the only variable. "
                f"Constants: pi, e.  Functions: {FUNC_LIST}.")
    elif isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in FUNCS:
            raise InputError(f"Unknown function. Available functions: {FUNC_LIST}.")
        if node.keywords:
            raise InputError("Keyword arguments are not allowed.")
        lo, hi = ARITY.get(node.func.id, (1, 1))
        if not lo <= len(node.args) <= hi:
            raise InputError(f"{node.func.id}() takes {lo if lo == hi else f'{lo} or {hi}'} argument(s).")
        for a in node.args:
            _check(a, allow_x)
    else:
        raise InputError("This expression contains something that is not allowed.")


def _g(x) -> str:
    return f"{x:.10g}"


def _run(fn, x, name):
    """Call fn(x); turn every numerical failure into a readable MathError."""
    try:
        v = fn(x)
    except ZeroDivisionError:
        raise MathError(f"Division by zero: {name}(x) is undefined at x = {_g(x)}.") from None
    except OverflowError:
        raise MathError(f"Overflow: {name}(x) is too large to compute at x = {_g(x)}.") from None
    except ValueError:
        raise MathError(
            f"Math domain error: {name}(x) is undefined at x = {_g(x)} "
            f"(e.g. log or sqrt of a negative number, log(0), asin/acos outside [-1, 1]).") from None
    except (TypeError, ArithmeticError) as e:
        raise MathError(f"Cannot evaluate {name}(x) at x = {_g(x)}: {e}") from None
    if isinstance(v, complex):
        raise MathError(
            f"Complex result: {name}(x) is not a real number at x = {_g(x)} "
            f"(even root or fractional power of a negative number).")
    v = float(v)
    if not math.isfinite(v):
        raise MathError(f"{name}(x) is not finite at x = {_g(x)}.")
    return v


class Func:
    """A validated, guarded function of x built from user text."""

    def __init__(self, text: str, name: str = "f", allow_x: bool = True,
                 allow_equals: bool = False):
        text = (text or "").strip()
        if not text:
            raise InputError("The expression is empty.")
        if len(text) > 500:
            raise InputError("The expression is too long.")
        if allow_equals:
            text = split_equation(text)
        elif "=" in text:
            raise InputError("Remove the '=' sign - type only the expression.")
        self.text, self.name = text, name
        self.py = _preprocess(text)
        try:
            tree = ast.parse(self.py, mode="eval")
        except SyntaxError:
            raise InputError(
                "Syntax error - check the parentheses and operators "
                "(example: x^3 - 7*x + 1).") from None
        _check(tree, allow_x)
        tree = _ToFloat().visit(tree)
        ast.fix_missing_locations(tree)
        self._code = compile(tree, "<expression>", "eval")

    def __call__(self, x: float) -> float:
        x = float(x)
        return _run(lambda v: eval(self._code, _NAMESPACE, {"x": v}), x, self.name)


class Wrapped:
    """Guarded wrapper around any python callable (used for generated g(x))."""

    def __init__(self, fn, name="g"):
        self.fn, self.name = fn, name

    def __call__(self, x):
        return _run(self.fn, float(x), self.name)


def parse_number(text: str, label: str = "value") -> float:
    """A number or constant expression such as 0.5, -2, pi/2, sqrt(2)."""
    try:
        f = Func(text, name=label, allow_x=False)
    except InputError as e:
        raise InputError(f"{label}: {e}") from None
    try:
        return f(0.0)
    except MathError:
        raise InputError(f"{label}: this expression cannot be evaluated "
                         f"(division by zero, domain error or overflow).") from None


# ---- derivatives -----------------------------------------------------
def num_deriv(fn, x, h=None):
    h = h or 1e-6 * max(1.0, abs(x))
    return (fn(x + h) - fn(x - h)) / (2 * h)


def _sympy_locals(sp):
    return {"ln": sp.log, "e": sp.E, "pi": sp.pi, "cbrt": sp.cbrt, "abs": sp.Abs,
            "sec": sp.sec, "csc": sp.csc, "cot": sp.cot, "pow": sp.Pow,
            "log2": lambda v: sp.log(v, 2), "log10": lambda v: sp.log(v, 10)}


def latex_of(f: Func):
    """LaTeX of the expression (needs sympy), or None."""
    try:
        import sympy as sp
        return sp.latex(sp.sympify(f.py, locals=_sympy_locals(sp)))
    except Exception:
        return None


def make_derivative(f: Func, user_text: str = "", probe: float | None = None):
    """Returns (df, kind, latex).  df(x) raises MathError where f' is undefined."""
    if user_text.strip():
        return Func(user_text, name="f'"), "typed by you", None
    try:
        import sympy as sp
        X = sp.Symbol("x")
        d = sp.diff(sp.sympify(f.py, locals=_sympy_locals(sp)), X)
        raw = sp.lambdify(X, d, "math")
        if probe is not None:
            try:
                raw(float(probe))
            except (ValueError, ZeroDivisionError, OverflowError):
                pass                                   # domain issue, not a code issue
        return Wrapped(raw, "f'"), "symbolic", sp.latex(d)
    except Exception:
        return (Wrapped(lambda v: num_deriv(f, v), "f'"),
                "numerical (central difference)", None)


# =====================================================================
#  2.  Results and helpers
# =====================================================================
@dataclass
class Result:
    method: str
    columns: list
    decimals: int
    tol: float
    rows: list = field(default_factory=list)       # raw numbers, first column = i
    xs: list = field(default_factory=list)         # x_i after every iteration
    diffs: list = field(default_factory=list)      # (i, |x_i - x_(i-1)|)
    root: float | None = None
    status: str = "ok"      # ok | error | diverged | max_iter | not_root
    message: str = ""
    warnings: list = field(default_factory=list)
    fx_root: float | None = None
    verified: bool | None = None
    x_col: int = 3          # index of the x_i column (for table styling)
    sci_cols: tuple = ()    # columns shown in scientific notation when tiny
    extra: dict = field(default_factory=dict)


def tol_of(decimals: int) -> float:
    return 0.5 * 10.0 ** (-decimals)


def _check_decimals(d):
    if not isinstance(d, int) or not 1 <= d <= MAX_DECIMALS:
        raise InputError(f"Decimal places must be an integer from 1 to {MAX_DECIMALS}.")


def format_number(v, digits, sci_below=None):
    """Fixed decimals without trailing zeros; tiny values like -6.08357E-06."""
    if v == 0:
        return "0"
    if abs(v) < 10 ** -(digits - 3) or (sci_below and abs(v) < sci_below):
        return f"{v:.6g}".upper()
    s = format(round(v, digits), f".{digits}f").rstrip("0").rstrip(".")
    return "0" if s in ("", "-0") else s


def _fail(res, i, exc):
    res.status = "error"
    res.root = None
    msg = str(exc) if isinstance(exc, MathError) else f"Arithmetic error: {exc}"
    res.message = f"Iteration {i}: {msg}"
    return res


class _Monitor:
    """Detects divergence, runaway growth and cycles."""

    def __init__(self):
        self.prev, self.inc, self.hist = None, 0, []

    def check(self, x, diff):
        msg = None
        if abs(x) > BLOW_UP:
            msg = "The values are growing without bound (x > 10^10)."
        else:
            self.inc = self.inc + 1 if (self.prev is not None and diff > self.prev) else 0
            if self.inc >= 10:
                msg = "|xi - x(i-1)| keeps getting bigger - the iteration is diverging."
            else:
                for h in self.hist[-8:-1]:
                    if abs(x - h) <= 4 * 2.220446049250313e-16 * max(1.0, abs(x)):
                        msg = "The iteration is stuck in a cycle (it keeps returning to the same values)."
                        if diff < 1e-6 * max(1.0, abs(x)):
                            msg += (" The requested precision is probably finer than double "
                                    "precision allows for numbers of this size - use fewer decimals.")
                        break
        self.prev = diff
        self.hist.append(x)
        return msg


def _verify(res: Result, f, scale: float):
    """After convergence: is the answer really a root (not a pole)?"""
    if res.root is None or f is None:
        return
    try:
        fr = f(res.root)
    except MathError as e:
        res.status, res.message = "not_root", f"f is undefined at the final value. {e}"
        return
    res.fx_root = fr
    if abs(fr) > max(1.0, scale):
        res.status = "not_root"
        res.message = (f"The iteration settled at x = {_g(res.root)}, but |f(x)| = {abs(fr):.3g} there. "
                       f"This is a discontinuity (pole), such as 1/x at 0 - it is NOT a root.")
        return
    try:
        slope = abs(num_deriv(f, res.root))
        thr = max(1e-9, 100 * slope * res.tol)
    except MathError:
        thr = 1e-6
    if abs(fr) > thr:
        res.verified = False
        res.warnings.append(
            f"|f(xr)| = {abs(fr):.3g} is not very small. The stopping rule may have been met "
            f"too early (common with False-Position) - try more decimals or another method.")
    else:
        res.verified = True


# =====================================================================
#  3.  Bracketing methods
# =====================================================================
def _bracket(f, a, b):
    if a == b:
        raise InputError("a and b must be different numbers.")
    vals = []
    for label, v in (("a", a), ("b", b)):
        try:
            vals.append(f(v))
        except MathError as e:
            raise InputError(f"The initial value {label} = {_g(v)} is not valid. {e}") from None
    fa, fb = vals
    if fa != 0 and fb != 0 and (fa > 0) == (fb > 0):
        raise InputError(
            f"f(a) = {fa:.6g} and f(b) = {fb:.6g} have the same sign, so a root is not guaranteed "
            f"in [{_g(min(a, b))}, {_g(max(a, b))}]. Choose an interval where f changes sign "
            f"(look at the graph).")
    return fa, fb


def analyze_interval(f, a, b, n=400):
    """Scan [a, b]: undefined points, sign changes, suspected poles."""
    lo, hi = min(a, b), max(a, b)
    xs = [lo + (hi - lo) * k / n for k in range(n + 1)]
    vals, undefined = [], []
    for x in xs:
        try:
            vals.append(f(x))
        except MathError:
            vals.append(None)
            undefined.append(x)
    ok = [abs(v) for v in vals if v is not None]
    scale = max([1.0] + ok[:1] + ok[-1:])
    changes, poles = [], []
    for k in range(1, n):
        if vals[k] is None and vals[k - 1] is not None and vals[k + 1] is not None \
                and (vals[k - 1] > 0) != (vals[k + 1] > 0):
            poles.append((xs[k - 1], xs[k + 1]))
    for k in range(n):
        v1, v2 = vals[k], vals[k + 1]
        if v1 is None or v2 is None:
            continue
        if (v1 > 0) != (v2 > 0) and v1 != 0 and v2 != 0:
            changes.append((xs[k], xs[k + 1]))
            if min(abs(v1), abs(v2)) > 10 * scale:
                poles.append((xs[k], xs[k + 1]))
    return {"undefined": undefined, "sign_changes": changes, "poles": poles}


def bisection_iterations(a, b, decimals):
    return max(1, math.ceil(math.log2(abs(b - a) / tol_of(decimals))))


def _bracket_method(name, formula, f, a, b, decimals, max_iter):
    _check_decimals(decimals)
    fa, fb = _bracket(f, a, b)
    third = ("xi=(a+b)/2" if formula == "bisection" else "xi=[af(b)-bf(a)]/[f(b)-f(a)]")
    res = Result(name, ["i", "a (f is +ve)", "b (f is -ve)", third, "f(xi)"],
                 decimals, tol_of(decimals), x_col=3, sci_cols=(4,))
    if fa == 0 or fb == 0:                               # endpoint already a root
        r = a if fa == 0 else b
        res.rows, res.xs, res.root = [(0, a, b, r, 0.0)], [r], r
        res.fx_root, res.verified = 0.0, True
        res.message = "One of the initial values is already an exact root."
        return res
    if fa < 0:
        a, b, fa, fb = b, a, fb, fa
    scale = max(abs(fa), abs(fb))
    prev = None
    for i in range(max_iter):
        try:
            x = (a + b) / 2 if formula == "bisection" else (a * fb - b * fa) / (fb - fa)
            fx = f(x)
        except (MathError, ArithmeticError) as e:
            return _fail(res, i, e)
        res.rows.append((i, a, b, x, fx))
        res.xs.append(x)
        if prev is not None:
            res.diffs.append((i, abs(x - prev)))
        if fx == 0 or (prev is not None and abs(x - prev) <= res.tol):
            res.root = x
            _verify(res, f, scale)
            return res
        if fx > 0:
            a, fa = x, fx
        else:
            b, fb = x, fx
        prev = x
    res.status = "max_iter"
    res.root = res.xs[-1]
    res.message = f"No convergence after {max_iter} iterations."
    return res


def bisection(f, a, b, decimals, max_iter=DEFAULT_MAX_ITER):
    return _bracket_method("Bisection Method", "bisection", f, a, b, decimals, max_iter)


def false_position(f, a, b, decimals, max_iter=DEFAULT_MAX_ITER):
    return _bracket_method("False-Position Method", "false", f, a, b, decimals, max_iter)


# =====================================================================
#  4.  Open methods
# =====================================================================
def newton(f, df, x0, decimals, max_iter=DEFAULT_MAX_ITER):
    _check_decimals(decimals)
    res = Result("Newton-Raphson Method",
                 ["i", "x(i-1)", "xi = x(i-1) - f(x(i-1))/f'(x(i-1))", "|xi - x(i-1)|"],
                 decimals, tol_of(decimals), x_col=2)
    try:
        scale = abs(f(x0))
    except MathError as e:
        raise InputError(f"The initial guess x0 = {_g(x0)} is not valid. {e}") from None
    mon, xp = _Monitor(), x0
    mon.hist.append(x0)
    for i in range(1, max_iter + 1):
        try:
            fx, slope = f(xp), df(xp)
            if slope == 0:
                raise MathError(
                    f"Division by zero: f'(x) = 0 at x = {_g(xp)} (horizontal tangent). "
                    f"Choose a different initial guess.")
            x = xp - fx / slope
        except (MathError, ArithmeticError) as e:
            return _fail(res, i, e)
        diff = abs(x - xp)
        res.rows.append((i, xp, x, diff))
        res.xs.append(x)
        res.diffs.append((i, diff))
        if diff <= res.tol:
            res.root = x
            _verify(res, f, scale)
            return res
        msg = mon.check(x, diff)
        if msg:
            res.status, res.message, res.root = "diverged", f"Iteration {i}: {msg}", None
            return res
        xp = x
    res.status, res.root = "max_iter", res.xs[-1]
    res.message = f"No convergence after {max_iter} iterations."
    return res


def secant(f, x0, x1, decimals, max_iter=DEFAULT_MAX_ITER):
    _check_decimals(decimals)
    if x0 == x1:
        raise InputError("x0 and x1 must be different numbers.")
    res = Result("Secant Method", ["i", "x(i-2)", "x(i-1)", "xi", "|xi - x(i-1)|"],
                 decimals, tol_of(decimals), x_col=3)
    scale = 0.0
    for label, v in (("x0", x0), ("x1", x1)):
        try:
            scale = max(scale, abs(f(v)))
        except MathError as e:
            raise InputError(f"The initial guess {label} = {_g(v)} is not valid. {e}") from None
    mon, xa, xb = _Monitor(), x0, x1
    mon.hist += [x0, x1]
    for i in range(2, max_iter + 2):
        try:
            fa, fb = f(xa), f(xb)
            if fb == fa:
                raise MathError(
                    f"Division by zero: f(x(i-1)) = f(x(i-2)) = {fb:.6g}, the secant line is horizontal.")
            x = xb - (xb - xa) * fb / (fb - fa)
        except (MathError, ArithmeticError) as e:
            return _fail(res, i, e)
        diff = abs(x - xb)
        res.rows.append((i, xa, xb, x, diff))
        res.xs.append(x)
        res.diffs.append((i, diff))
        if diff <= res.tol:
            res.root = x
            _verify(res, f, scale)
            return res
        msg = mon.check(x, diff)
        if msg:
            res.status, res.message, res.root = "diverged", f"Iteration {i}: {msg}", None
            return res
        xa, xb = xb, x
    res.status, res.root = "max_iter", res.xs[-1]
    res.message = f"No convergence after {max_iter} iterations."
    return res


# =====================================================================
#  5.  Fixed-point iteration
# =====================================================================
def real_root(v, k):
    """Real k-th root (odd roots of negative numbers are allowed)."""
    if k == 1:
        return v
    if v < 0:
        if float(k).is_integer() and int(k) % 2 != 0:
            return -((-v) ** (1.0 / k))
        raise ValueError("even root of a negative number")
    return v ** (1.0 / k)


def slope_at(g, x):
    """|g'(x)|, or infinity when g is not usable there."""
    try:
        v = abs(num_deriv(g, x))
        return v if math.isfinite(v) else math.inf
    except (MathError, ArithmeticError):
        return math.inf


@dataclass
class Candidate:
    label: str
    g: object
    slope: float
    latex: str = ""


def fixed_point_candidates(f: Func, x0: float):
    """Rearrangements of f(x)=0 into x=g(x), each tested with |g'(x0)|."""
    raw = []                                             # (label, latex, g)
    try:                                                  # (a) isolate one power of x
        import sympy as sp
        X = sp.Symbol("x")
        fe = sp.expand(sp.sympify(f.py, locals=_sympy_locals(sp)))
        for term in sp.Add.make_args(fe):
            c, part = term.as_independent(X)
            if not part.has(X):
                continue
            base, ex = part.as_base_exp()
            if base != X or not (ex.is_number and ex.is_real) or ex == 0:
                continue
            inner = sp.simplify(-(fe - term) / c)
            h, k = sp.lambdify(X, inner, "math"), float(ex)
            txt = str(inner).replace("**", "^")
            if k == 1:
                label, tex = f"x = {txt}", "x = " + sp.latex(inner)
            else:
                label = f"x = ({txt})^({sp.Integer(1) / ex})"
                tex = "x = \\left(" + sp.latex(inner) + "\\right)^{" + sp.latex(sp.Integer(1) / ex) + "}"
            raw.append((label, tex, Wrapped(lambda v, h=h, k=k: real_root(h(v), k))))
    except Exception:
        pass
    try:                                                  # (b) generic forms
        m = num_deriv(f, x0)
        if abs(m) > 1e-12:
            raw.append((f"x = x - f(x)/({m:.6g})   [slope frozen at x0]",
                        f"x = x - \\dfrac{{f(x)}}{{{m:.6g}}}",
                        Wrapped(lambda v, m=m: v - f(v) / m)))
    except (MathError, ArithmeticError):
        pass
    raw.append(("x = x + f(x)", "x = x + f(x)", Wrapped(lambda v: v + f(v))))
    raw.append(("x = x - f(x)", "x = x - f(x)", Wrapped(lambda v: v - f(v))))
    out = [Candidate(lab, g, slope_at(g, x0), tex) for lab, tex, g in raw]
    out.sort(key=lambda c: c.slope)
    return out


def fixed_point(g, x0, decimals, max_iter=DEFAULT_MAX_ITER, f=None):
    _check_decimals(decimals)
    res = Result("Fixed-Point Iteration Method",
                 ["i", "x(i-1)", "xi = g(x(i-1))", "|xi - x(i-1)|", "|g'(x(i-1))|"],
                 decimals, tol_of(decimals), x_col=2)
    scale = 0.0
    if f is not None:
        try:
            scale = abs(f(x0))
        except MathError as e:
            raise InputError(f"The initial guess x0 = {_g(x0)} is not valid. {e}") from None
    mon, xp = _Monitor(), x0
    mon.hist.append(x0)
    for i in range(1, max_iter + 1):
        try:
            x = g(xp)
        except (MathError, ArithmeticError) as e:
            return _fail(res, i, e)
        diff = abs(x - xp)
        res.rows.append((i, xp, x, diff, slope_at(g, xp)))
        res.xs.append(x)
        res.diffs.append((i, diff))
        if diff <= res.tol:
            res.root = x
            sl = slope_at(g, x)
            res.extra["g_slope"] = sl
            if sl >= 1:
                res.warnings.append(
                    f"|g'(xr)| = {sl:.3g} >= 1: this fixed point is not stable, "
                    f"so the value is not a reliable root.")
            _verify(res, f, scale)
            return res
        msg = mon.check(x, diff)
        if msg:
            res.status, res.message, res.root = "diverged", f"Iteration {i}: {msg}", None
            return res
        xp = x
    res.status, res.root = "max_iter", res.xs[-1]
    res.message = f"No convergence after {max_iter} iterations."
    return res