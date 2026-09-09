"""Membership function families used by the fuzzification engine.

The four families of the thesis (trapezoidal, triangular, Gaussian and
sigmoidal, Table "membership functions") are implemented both as plain
vectorised functions and as small callable objects that carry their
parameters, so that a configuration file can describe every fuzzy attribute.

Two extra shapes cover the working-hour variants of the retail case:

* :class:`WrapAroundTrapezoidal` -- a window that crosses midnight
  (night shift, ``(a,b,c,d) = (20,22,4,6)``);
* :class:`SplitShift` -- the union (maximum) of two trapezoids
  (working day interrupted by a midday pause).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np

ArrayLike = Any


# --------------------------------------------------------------------------- #
# Vectorised primitive functions (Equations 1, 4, gaussian, sigmoid)
# --------------------------------------------------------------------------- #
def trapezoidal(x: ArrayLike, a: float, b: float, c: float, d: float) -> np.ndarray:
    """Trapezoidal membership ``mu_trap(x; a, b, c, d)`` (Eq. 1).

    Rises linearly on ``[a, b]``, is 1 on ``[b, c]`` and falls on ``[c, d]``.
    ``a == b`` or ``c == d`` give vertical edges (a crisp interval).
    """
    x = np.asarray(x, dtype=float)
    if not (a <= b <= c <= d):
        raise ValueError(f"Trapezoid requires a <= b <= c <= d, got {(a, b, c, d)}")
    out = np.zeros_like(x)
    # rising edge
    if b > a:
        mask = (x > a) & (x < b)
        out[mask] = (x[mask] - a) / (b - a)
    # plateau
    out[(x >= b) & (x <= c)] = 1.0
    # falling edge
    if d > c:
        mask = (x > c) & (x < d)
        out[mask] = (d - x[mask]) / (d - c)
    return np.clip(out, 0.0, 1.0)


def triangular(x: ArrayLike, a: float, b: float, c: float) -> np.ndarray:
    """Triangular membership ``mu_tri(x; a, b, c)`` (Eq. 4) with peak at ``b``."""
    if not (a <= b <= c):
        raise ValueError(f"Triangle requires a <= b <= c, got {(a, b, c)}")
    return trapezoidal(x, a, b, b, c)


def gaussian(x: ArrayLike, c: float, sigma: float) -> np.ndarray:
    """Gaussian membership ``exp(-(x - c)^2 / (2 sigma^2))``."""
    if sigma <= 0:
        raise ValueError("sigma must be > 0")
    x = np.asarray(x, dtype=float)
    return np.exp(-((x - c) ** 2) / (2.0 * sigma ** 2))


def sigmoidal(x: ArrayLike, a: float, c: float) -> np.ndarray:
    """Sigmoidal membership ``1 / (1 + exp(-a (x - c)))``.

    ``a > 0`` gives an increasing curve (e.g. *high* chat activity),
    ``a < 0`` a decreasing one (e.g. *low* chat activity).
    """
    x = np.asarray(x, dtype=float)
    z = np.clip(-a * (x - c), -500, 500)
    return 1.0 / (1.0 + np.exp(z))


# --------------------------------------------------------------------------- #
# Parameterised membership-function objects
# --------------------------------------------------------------------------- #
@dataclass
class MembershipFunction:
    """Base class: a callable mapping crisp values to degrees in [0, 1]."""

    family: str = "base"
    label: str = ""

    def __call__(self, x: ArrayLike) -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError

    def params(self) -> Dict[str, float]:
        return {k: v for k, v in self.__dict__.items() if k not in ("family", "label")}

    def support(self) -> tuple[float, float]:
        """Approximate support, used for plotting."""
        raise NotImplementedError

    def to_dict(self) -> Dict[str, Any]:
        return {"family": self.family, "label": self.label, **self.params()}

    def alpha_cut_interval(self, alpha: float, grid: int = 2001) -> tuple[float, float] | None:
        """Interval ``{x | mu(x) >= alpha}`` evaluated on a grid over the support."""
        lo, hi = self.support()
        xs = np.linspace(lo, hi, grid)
        mask = self(xs) >= alpha
        if not mask.any():
            return None
        return float(xs[mask].min()), float(xs[mask].max())


@dataclass
class Trapezoidal(MembershipFunction):
    a: float = 0.0
    b: float = 0.0
    c: float = 1.0
    d: float = 1.0
    family: str = field(default="trapezoidal", init=False)

    def __call__(self, x: ArrayLike) -> np.ndarray:
        return trapezoidal(x, self.a, self.b, self.c, self.d)

    def support(self) -> tuple[float, float]:
        return (self.a, self.d)


@dataclass
class Triangular(MembershipFunction):
    a: float = 0.0
    b: float = 0.5
    c: float = 1.0
    family: str = field(default="triangular", init=False)

    def __call__(self, x: ArrayLike) -> np.ndarray:
        return triangular(x, self.a, self.b, self.c)

    def support(self) -> tuple[float, float]:
        return (self.a, self.c)


@dataclass
class Gaussian(MembershipFunction):
    c: float = 0.5
    sigma: float = 0.15
    family: str = field(default="gaussian", init=False)

    def __call__(self, x: ArrayLike) -> np.ndarray:
        return gaussian(x, self.c, self.sigma)

    def support(self) -> tuple[float, float]:
        return (self.c - 4 * self.sigma, self.c + 4 * self.sigma)


@dataclass
class Sigmoidal(MembershipFunction):
    a: float = 10.0
    c: float = 0.5
    family: str = field(default="sigmoidal", init=False)

    def __call__(self, x: ArrayLike) -> np.ndarray:
        return sigmoidal(x, self.a, self.c)

    def support(self) -> tuple[float, float]:
        width = 8.0 / max(abs(self.a), 1e-9)
        return (self.c - width, self.c + width)


@dataclass
class WrapAroundTrapezoidal(MembershipFunction):
    """Trapezoid on a circular domain (hour of day), e.g. night shift 22:00-06:00.

    Parameters follow the thesis notation ``(a, b, c, d) = (20, 22, 4, 6)``:
    the window rises from ``a`` to ``b`` before midnight and falls from ``c`` to
    ``d`` after midnight. ``period`` is the length of the circular domain.
    """

    a: float = 20.0
    b: float = 22.0
    c: float = 4.0
    d: float = 6.0
    period: float = 24.0
    family: str = field(default="wraparound_trapezoidal", init=False)

    def __call__(self, x: ArrayLike) -> np.ndarray:
        x = np.mod(np.asarray(x, dtype=float), self.period)
        # Unwrap: shift so that ``a`` becomes the origin, then it is an ordinary trapezoid.
        shift = self.a
        xs = np.mod(x - shift, self.period)
        a, b = 0.0, np.mod(self.b - shift, self.period)
        c, d = np.mod(self.c - shift, self.period), np.mod(self.d - shift, self.period)
        if d == 0.0:
            d = self.period
        return trapezoidal(xs, a, b, c, d)

    def support(self) -> tuple[float, float]:
        return (0.0, self.period)


@dataclass
class SplitShift(MembershipFunction):
    """Maximum of several trapezoids, e.g. a split shift with a midday pause."""

    windows: List[Sequence[float]] = field(default_factory=list)
    family: str = field(default="split_shift", init=False)

    def __call__(self, x: ArrayLike) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        out = np.zeros_like(x)
        for w in self.windows:
            out = np.maximum(out, trapezoidal(x, *w))
        return out

    def support(self) -> tuple[float, float]:
        lo = min(w[0] for w in self.windows)
        hi = max(w[3] for w in self.windows)
        return (lo, hi)

    def params(self) -> Dict[str, Any]:  # type: ignore[override]
        return {"windows": [list(map(float, w)) for w in self.windows]}


@dataclass
class Complement(MembershipFunction):
    """``1 - mu(x)`` of another function (e.g. non-working hours, Eq. 3)."""

    inner: MembershipFunction = field(default_factory=Trapezoidal)
    family: str = field(default="complement", init=False)

    def __call__(self, x: ArrayLike) -> np.ndarray:
        return 1.0 - self.inner(x)

    def support(self) -> tuple[float, float]:
        return self.inner.support()

    def params(self) -> Dict[str, Any]:  # type: ignore[override]
        return {"inner": self.inner.to_dict()}


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #
_FAMILIES = {
    "trapezoidal": Trapezoidal,
    "trapezoid": Trapezoidal,
    "triangular": Triangular,
    "triangle": Triangular,
    "gaussian": Gaussian,
    "sigmoidal": Sigmoidal,
    "sigmoid": Sigmoidal,
    "wraparound_trapezoidal": WrapAroundTrapezoidal,
    "wraparound": WrapAroundTrapezoidal,
    "split_shift": SplitShift,
    "complement": Complement,
}


def build_membership_function(spec: Dict[str, Any], label: str = "") -> MembershipFunction:
    """Build a membership function from a mapping such as
    ``{"family": "trapezoidal", "params": [6, 8, 16, 18]}`` or
    ``{"family": "gaussian", "c": 0.5, "sigma": 0.2}``."""
    spec = dict(spec)
    family = str(spec.pop("family", "trapezoidal")).lower()
    if family not in _FAMILIES:
        raise ValueError(f"Unknown membership family '{family}'. Known: {sorted(_FAMILIES)}")
    cls = _FAMILIES[family]
    label = spec.pop("label", label)
    params = spec.pop("params", None)
    if family == "complement":
        inner_spec = spec.pop("inner", params)
        return Complement(inner=build_membership_function(inner_spec), label=label)
    if family == "split_shift":
        windows = spec.pop("windows", params)
        return SplitShift(windows=[list(map(float, w)) for w in windows], label=label)
    if params is not None:
        names = [f for f in cls.__dataclass_fields__ if f not in ("family", "label", "period")]
        if len(params) != len(names):
            raise ValueError(f"{family} expects {len(names)} params {names}, got {params}")
        spec.update(dict(zip(names, map(float, params))))
    kwargs = {k: (float(v) if isinstance(v, (int, float)) else v) for k, v in spec.items()}
    return cls(label=label, **kwargs)


def linguistic_partition(
    family: str, terms: Iterable[str], lo: float = 0.0, hi: float = 1.0
) -> Dict[str, MembershipFunction]:
    """Evenly spaced *low/medium/high*-style partition on ``[lo, hi]``.

    Useful when a configuration only names the terms and lets the engine
    derive standard overlapping shapes on the normalised domain.
    """
    terms = list(terms)
    n = len(terms)
    if n == 0:
        raise ValueError("at least one term is required")
    centres = np.linspace(lo, hi, n) if n > 1 else np.array([(lo + hi) / 2])
    step = (hi - lo) / max(n - 1, 1)
    out: Dict[str, MembershipFunction] = {}
    for i, term in enumerate(terms):
        c = float(centres[i])
        if family in ("triangular", "triangle"):
            a, b = c - step, c + step
            if i == 0:
                out[term] = Trapezoidal(a=lo - 1e-9, b=lo, c=lo, d=b, label=term)
            elif i == n - 1:
                out[term] = Trapezoidal(a=a, b=hi, c=hi, d=hi + 1e-9, label=term)
            else:
                out[term] = Triangular(a=a, b=c, c=b, label=term)
        elif family in ("trapezoidal", "trapezoid"):
            half = step / 2.0
            out[term] = Trapezoidal(a=c - step, b=c - half, c=c + half, d=c + step, label=term)
            if i == 0:
                out[term] = Trapezoidal(a=lo - 1e-9, b=lo, c=c + half, d=c + step, label=term)
            elif i == n - 1:
                out[term] = Trapezoidal(a=c - step, b=c - half, c=hi, d=hi + 1e-9, label=term)
        elif family == "gaussian":
            out[term] = Gaussian(c=c, sigma=step / 2.0 if n > 1 else (hi - lo) / 4, label=term)
        elif family in ("sigmoidal", "sigmoid"):
            slope = 10.0 / max(hi - lo, 1e-9)
            if n == 1:
                out[term] = Sigmoidal(a=slope, c=c, label=term)
            elif i == 0:
                out[term] = Sigmoidal(a=-slope, c=(lo + hi) / 2, label=term)
            elif i == n - 1:
                out[term] = Sigmoidal(a=slope, c=(lo + hi) / 2, label=term)
            else:
                out[term] = Gaussian(c=c, sigma=step / 2.0, label=term)
        else:
            raise ValueError(f"cannot build a linguistic partition for family '{family}'")
    return out
