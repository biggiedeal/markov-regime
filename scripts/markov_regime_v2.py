# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "pandas", "yfinance", "scipy"]
# ///
"""Markov 2.0 — Hedge Fund Method (three documented flaws fixed).

FIX 1  Stride sampling  — builds the matrix from NON-overlapping windows so
        consecutive windows do not share 19 days and fake diagonal persistence.
        Both overlapping (legacy) and stride-sampled (true) matrices are always
        shown side-by-side. Only the stride-sampled matrix is statistically honest.

FIX 2  Label verification — after building any matrix the mapping is
        programmatically self-checked against three historical periods: the most
        extreme bear in the data, the most extreme bull, and the flattest stretch.
        If the rendered labels disagree with the underlying returns the script
        prints a clear ERROR and exits rather than showing a wrong table.

FIX 3  Two explicit modes — the user chooses once at startup:
        FILTER    Regime gates an existing strategy. The script emits a gating
                  signal (+1 / 0 / -1) the user's own logic can read. Position
                  decisions stay with the user's strategy.
        STANDALONE Trade the differential directly; position size is scaled to
                  |signal| with a user-supplied cap (default 1.0 = fully invested).

Framework: Roan (@RohOnChain). Upgraded to 2.0 by Lewis Jackson.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

STATES = ["Bear", "Sideways", "Bull"]   # indices 0, 1, 2 — never change this order
DEFAULT_WINDOW = 20
DEFAULT_THRESHOLD = 0.05                # ±5 % rolling return
DEFAULT_YEARS = 10
DEFAULT_MIN_TRAIN = 252


# ---------------------------------------------------------------------------
# Data loading (unchanged from v1 — already asset-agnostic)
# ---------------------------------------------------------------------------

def fetch_ticker(ticker: str, years: int = DEFAULT_YEARS) -> pd.Series:
    import yfinance as yf
    end = pd.Timestamp.now("UTC").tz_localize(None).normalize()
    start = end - pd.DateOffset(years=years)
    df = pd.DataFrame()
    for attempt in (1, 2):
        try:
            df = yf.download(
                ticker,
                start=start.strftime("%Y-%m-%d"),
                end=end.strftime("%Y-%m-%d"),
                progress=False,
                auto_adjust=True,
            )
        except Exception as exc:
            print(f"  ! yfinance error on attempt {attempt}: {exc}", file=sys.stderr)
            df = pd.DataFrame()
        if not df.empty:
            break
        if attempt == 1:
            print("  ! yfinance empty — retrying in 30 s.", file=sys.stderr)
            time.sleep(30)
    if df.empty:
        raise RuntimeError(
            f"yfinance returned empty data for {ticker} after retry. "
            "Yahoo may be rate-limiting. Try again in a few minutes."
        )
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    close = df["Close"].dropna()
    close.name = ticker
    return close


def load_csv(path: str) -> pd.Series:
    df = pd.read_csv(path)
    if df.empty:
        raise RuntimeError(f"{path} is empty.")
    cols = {c.lower().strip(): c for c in df.columns}
    date_col = next(
        (cols[k] for k in ("date", "time", "timestamp", "datetime") if k in cols),
        df.columns[0],
    )
    close_col = next(
        (cols[k] for k in ("close", "adj close", "adj_close", "adjclose", "price", "last") if k in cols),
        None,
    )
    if close_col is None:
        numeric = [c for c in df.select_dtypes("number").columns if c != date_col]
        if len(numeric) == 1:
            close_col = numeric[0]
        else:
            raise RuntimeError(
                f"Could not find a close column in {path}. "
                f"Add a column named: close, adj close, price, or last. "
                f"Columns seen: {list(df.columns)}"
            )
    out = df[[date_col, close_col]].copy()
    out[date_col] = pd.to_datetime(out[date_col], utc=False, errors="coerce")
    out = out.dropna(subset=[date_col]).sort_values(date_col)
    return pd.Series(
        pd.to_numeric(out[close_col], errors="coerce").to_numpy(),
        index=pd.DatetimeIndex(out[date_col]),
        name=Path(path).stem,
    ).dropna()


# ---------------------------------------------------------------------------
# Core labelling
# ---------------------------------------------------------------------------

def label_regimes(
    close: pd.Series,
    window: int = DEFAULT_WINDOW,
    threshold: float = DEFAULT_THRESHOLD,
) -> pd.Series:
    """Label each bar from the trailing `window`-day return.

    Bear (0) : rolling return < -threshold
    Sideways (1): |rolling return| <= threshold
    Bull (2) : rolling return > +threshold
    """
    rolling_return = close.pct_change(window)
    labels = pd.Series(1, index=close.index, dtype=int)
    labels[rolling_return > threshold] = 2
    labels[rolling_return < -threshold] = 0
    return labels.loc[rolling_return.notna()]


# ---------------------------------------------------------------------------
# FIX 1 — Stride sampling
# ---------------------------------------------------------------------------

def build_transition_matrix_overlapping(labels: pd.Series) -> np.ndarray:
    """Legacy: counts day-to-day transitions.

    WARNING — consecutive 20-day rolling windows share 19 days. This inflates
    the diagonal and is NOT statistically honest. Shown for comparison only.
    """
    counts = np.zeros((3, 3), dtype=float)
    arr = np.asarray(labels, dtype=int)
    for i in range(len(arr) - 1):
        counts[arr[i], arr[i + 1]] += 1.0
    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return counts / row_sums


def build_transition_matrix_stride(labels: pd.Series, stride: int = DEFAULT_WINDOW) -> np.ndarray:
    """FIX 1: stride-sampled transition matrix.

    Takes every `stride`-th label so that consecutive observations come from
    NON-overlapping windows. This eliminates the artificial autocorrelation
    that the legacy overlapping version produces on the diagonal.

    This is the ONLY matrix that is statistically honest for a rolling-window label.
    """
    arr = np.asarray(labels, dtype=int)[::stride]
    counts = np.zeros((3, 3), dtype=float)
    for i in range(len(arr) - 1):
        counts[arr[i], arr[i + 1]] += 1.0
    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return counts / row_sums


# ---------------------------------------------------------------------------
# FIX 2 — Label verification
# ---------------------------------------------------------------------------

def _find_verification_periods(
    close: pd.Series,
    labels: pd.Series,
    window: int,
) -> list[dict]:
    """Automatically identify three reference periods from the data.

    Bear  : the window ending at the date with the most negative rolling return
    Bull  : the window ending at the date with the most positive rolling return
    Sideways: the window ending at the date whose rolling return is closest to 0
    """
    rolling_return = close.pct_change(window).dropna()
    periods = []

    # BEAR period: most negative rolling return
    bear_date = rolling_return.idxmin()
    bear_ret = float(rolling_return.loc[bear_date])
    bear_label = int(labels.loc[bear_date]) if bear_date in labels.index else None
    periods.append({
        "name": "worst drawdown",
        "date": str(bear_date.date()),
        "rolling_return": bear_ret,
        "expected_state": 0,
        "expected_name": "Bear",
        "actual_state": bear_label,
        "actual_name": STATES[bear_label] if bear_label is not None else "N/A",
    })

    # BULL period: most positive rolling return
    bull_date = rolling_return.idxmax()
    bull_ret = float(rolling_return.loc[bull_date])
    bull_label = int(labels.loc[bull_date]) if bull_date in labels.index else None
    periods.append({
        "name": "strongest rally",
        "date": str(bull_date.date()),
        "rolling_return": bull_ret,
        "expected_state": 2,
        "expected_name": "Bull",
        "actual_state": bull_label,
        "actual_name": STATES[bull_label] if bull_label is not None else "N/A",
    })

    # SIDEWAYS period: rolling return closest to zero
    side_date = rolling_return.abs().idxmin()
    side_ret = float(rolling_return.loc[side_date])
    side_label = int(labels.loc[side_date]) if side_date in labels.index else None
    periods.append({
        "name": "flattest stretch",
        "date": str(side_date.date()),
        "rolling_return": side_ret,
        "expected_state": 1,
        "expected_name": "Sideways",
        "actual_state": side_label,
        "actual_name": STATES[side_label] if side_label is not None else "N/A",
    })

    return periods


def verify_labels(
    close: pd.Series,
    labels: pd.Series,
    window: int = DEFAULT_WINDOW,
) -> tuple[bool, list[dict]]:
    """FIX 2: self-check label mapping against three historical reference points.

    Returns (all_passed, list_of_checks). Prints a clear pass/fail per period.
    If any check fails the matrix should NOT be shown — this indicates a label
    inversion or mapping bug, which is exactly what the v1 shipped with.
    """
    periods = _find_verification_periods(close, labels, window)
    all_passed = True

    print("\nFIX 2 — Label verification (self-check before displaying matrix):")
    for p in periods:
        passed = p["actual_state"] == p["expected_state"]
        if not passed:
            all_passed = False
        icon = "PASS" if passed else "FAIL"
        print(
            f"  [{icon}] {p['name']:20s} {p['date']}  "
            f"rolling_ret={p['rolling_return']:+.3f}  "
            f"expected={p['expected_name']:8s}  "
            f"got={p['actual_name']}"
        )

    if all_passed:
        print("  All three reference periods verified. Matrix is safe to display.")
    else:
        print("\n  ERROR: label mapping mismatch detected.")
        print("  The matrix below WOULD contain a bull/bear label inversion.")
        print("  Fix: check STATES order and threshold sign before displaying.")

    return all_passed, periods


# ---------------------------------------------------------------------------
# Downstream math (unchanged — correct in v1)
# ---------------------------------------------------------------------------

def stationary_distribution(matrix: np.ndarray) -> np.ndarray:
    eigvals, eigvecs = np.linalg.eig(matrix.T)
    idx = np.argmin(np.abs(eigvals - 1.0))
    vec = np.abs(np.real(eigvecs[:, idx]))
    return vec / vec.sum()


def signal_from_matrix(matrix: np.ndarray, current_state: int) -> float:
    """P(next=Bull | current) - P(next=Bear | current)."""
    return float(matrix[current_state, 2] - matrix[current_state, 0])


def nstep_forecast(matrix: np.ndarray, n: int) -> np.ndarray:
    return np.linalg.matrix_power(matrix, n)


# ---------------------------------------------------------------------------
# Walk-forward — run on BOTH matrices for the before/after comparison
# ---------------------------------------------------------------------------

def walk_forward_backtest(
    close: pd.Series,
    labels: pd.Series,
    use_stride: bool = True,
    stride: int = DEFAULT_WINDOW,
    min_train: int = DEFAULT_MIN_TRAIN,
) -> dict:
    """No-lookahead walk-forward. At each bar t:
      - fit the (overlapping or stride-sampled) matrix on labels[:t]
      - read signal from current state
      - hold for one bar

    use_stride=True  → stride-sampled (FIX 1 applied)
    use_stride=False → legacy overlapping (v1 behaviour)
    """
    daily_returns = close.pct_change().dropna()
    common = labels.index.intersection(daily_returns.index)
    lab = np.asarray(labels.loc[common], dtype=int)
    rets = daily_returns.loc[common].to_numpy(dtype=float)

    if len(lab) < min_train + 30:
        return {"sharpe": float("nan"), "max_drawdown": float("nan"), "n_trades": 0}

    strategy_returns = []

    for t in range(min_train, len(lab) - 1):
        lab_slice = pd.Series(lab[:t])
        if use_stride:
            P_t = build_transition_matrix_stride(lab_slice, stride=stride)
        else:
            P_t = build_transition_matrix_overlapping(lab_slice)

        current_state = int(lab[t])
        signal = signal_from_matrix(P_t, current_state)
        position = float(np.sign(signal))
        strategy_returns.append(position * rets[t + 1])

    sr = np.array(strategy_returns, dtype=float)
    std = sr.std(ddof=1) if len(sr) > 1 else 0.0
    sharpe = float(sr.mean() / std * np.sqrt(252)) if std > 0 and np.isfinite(std) else float("nan")
    equity = (1.0 + sr).cumprod()
    running_max = np.maximum.accumulate(equity)
    dd = (equity - running_max) / running_max
    max_dd = float(dd.min()) if len(dd) else float("nan")
    return {"sharpe": sharpe, "max_drawdown": max_dd, "n_trades": int(len(sr))}


# ---------------------------------------------------------------------------
# FIX 3 — Mode helpers
# ---------------------------------------------------------------------------

def describe_modes() -> None:
    print(
        "\nFIX 3 — Two explicit modes (choose one):\n"
        "\n"
        "  FILTER (default)\n"
        "    The regime gates YOUR existing strategy. Markov 2.0 outputs a\n"
        "    gating signal (+1 long-allowed / 0 flat / -1 short-allowed).\n"
        "    Your strategy decides entries; the regime decides whether it is\n"
        "    ALLOWED to act. Position sizing stays entirely with you.\n"
        "\n"
        "  STANDALONE\n"
        "    Trade the bull−bear differential directly, no existing strategy\n"
        "    needed. Position size = signal × size_cap (default 1.0 = fully\n"
        "    invested). The walk-forward Sharpe and max drawdown are what you\n"
        "    are actually signing up for when you choose this mode.\n"
    )


def gating_signal(signal: float, threshold: float = 0.0) -> int:
    """FILTER mode: +1 = longs allowed, -1 = shorts allowed, 0 = flat (chop)."""
    if signal > threshold:
        return 1
    if signal < -threshold:
        return -1
    return 0


def standalone_position(signal: float, size_cap: float = 1.0) -> float:
    """STANDALONE mode: position size proportional to conviction."""
    return float(np.clip(signal * size_cap, -size_cap, size_cap))


# ---------------------------------------------------------------------------
# Pretty display helpers
# ---------------------------------------------------------------------------

def _print_matrix(label: str, P: np.ndarray) -> None:
    print(f"\n{label}:")
    print(f"  {'':>9s}  {'Bear':>9s}  {'Sideways':>9s}  {'Bull':>9s}")
    for i, from_state in enumerate(STATES):
        row = "  ".join(f"{P[i, j] * 100:7.2f}%" for j in range(3))
        diag_note = f"  ← {P[i, i]*100:.1f}% sticky" if P[i, i] > 0.5 else ""
        print(f"  {from_state:>9s}  {row}{diag_note}")


def _print_comparison(P_legacy: np.ndarray, P_stride: np.ndarray) -> None:
    print("\n" + "=" * 70)
    print(" SIDE-BY-SIDE MATRIX COMPARISON (FIX 1)")
    print("=" * 70)
    print(
        "\n  WARNING: The overlapping matrix (left) is statistically dishonest.\n"
        "  Consecutive 20-day windows share 19 days — this artificially inflates\n"
        "  the diagonal, making regimes look stickier than they actually are.\n"
        "  The stride-sampled matrix (right) is the only one worth trading.\n"
    )
    _print_matrix("OVERLAPPING (legacy — DO NOT trade this)", P_legacy)
    _print_matrix("STRIDE-SAMPLED (FIX 1 — statistically honest)", P_stride)

    print("\n  Diagonal inflation (overlapping minus stride-sampled):")
    for i, s in enumerate(STATES):
        diff = (P_legacy[i, i] - P_stride[i, i]) * 100
        sign = "+" if diff >= 0 else ""
        print(f"    {s:>9s}: {sign}{diff:.2f} pp inflation from overlapping windows")


def _print_backtest_comparison(bt_legacy: dict, bt_stride: dict) -> None:
    print("\n" + "=" * 70)
    print(" BEFORE vs AFTER FIX 1 — WALK-FORWARD BACKTEST")
    print("=" * 70)
    print(
        "\n  'Before' = overlapping matrix (inflated diagonals, v1 behaviour)\n"
        "  'After'  = stride-sampled matrix (honest, FIX 1 applied)\n"
        "\n  Backtests flatter. The fixed matrix shows uglier, truer numbers —\n"
        "  those are the only ones worth trading.\n"
    )
    rows = [
        ("Sharpe (annualised)", bt_legacy["sharpe"], bt_stride["sharpe"], ".3f"),
        ("Max drawdown", bt_legacy["max_drawdown"], bt_stride["max_drawdown"], ".2%"),
        ("Trades evaluated", bt_legacy["n_trades"], bt_stride["n_trades"], "d"),
    ]
    for name, v_before, v_after, fmt in rows:
        if name == "Trades evaluated":
            b_str = str(v_before)
            a_str = str(v_after)
        elif np.isfinite(v_before) and np.isfinite(v_after):
            b_str = format(v_before, fmt)
            a_str = format(v_after, fmt)
        else:
            b_str = "NaN"
            a_str = "NaN"
        print(f"  {name:<25s}  before={b_str:>10s}   after={a_str:>10s}")


# ---------------------------------------------------------------------------
# Optional HMM layer
# ---------------------------------------------------------------------------

def _hmm_section(close: pd.Series, labels: pd.Series, enabled: bool) -> dict:
    if not enabled:
        return {"available": False, "reason": "disabled via --no-hmm"}
    try:
        from hmmlearn import hmm
    except Exception:
        return {"available": False, "reason": "hmmlearn not installed"}

    X = close.pct_change().dropna().to_numpy(dtype=float).reshape(-1, 1)
    model = hmm.GaussianHMM(n_components=3, covariance_type="diag",
                             n_iter=200, random_state=42)
    model.fit(X)
    hidden = model.predict(X)

    # Identify which latent state corresponds to Bear/Sideways/Bull by mean return
    means = np.array([model.means_[k][0] for k in range(3)])
    order = np.argsort(means)  # ascending → [bear_k, side_k, bull_k]

    # Agreement: compare HMM labelling with threshold labelling
    # We need to align the two label sequences to the same index
    returns_index = close.pct_change().dropna().index
    hmm_series = pd.Series(hidden, index=returns_index[-len(hidden):], dtype=int)
    common = hmm_series.index.intersection(labels.index)
    if len(common) == 0:
        agreement = float("nan")
    else:
        # Map HMM latent states to Bear=0/Side=1/Bull=2 by mean return rank
        rank_map = {int(k): rank for rank, k in enumerate(order)}
        hmm_mapped = hmm_series.loc[common].map(rank_map)
        thresh_mapped = labels.loc[common]
        agreement = float((hmm_mapped == thresh_mapped).mean())

    regimes = []
    rank_names = ["Bear", "Sideways", "Bull"]
    for rank, k in enumerate(order):
        regimes.append({
            "label": rank_names[rank],
            "latent_state": int(k),
            "mean_daily_return": float(means[k]),
        })

    return {
        "available": True,
        "regimes": regimes,
        "agreement_with_threshold_labels": agreement,
        "green_light": agreement > 0.65,
        "note": (
            "Agreement > 65% between HMM and threshold labels is the green light. "
            "Both methods agree on the same regime history."
        ),
    }


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def analyze_v2(
    close: pd.Series,
    source: str,
    window: int = DEFAULT_WINDOW,
    threshold: float = DEFAULT_THRESHOLD,
    min_train: int = DEFAULT_MIN_TRAIN,
    mode: str = "FILTER",
    filter_threshold: float = 0.0,
    size_cap: float = 1.0,
    hmm: bool = True,
    skip_verification: bool = False,
) -> dict:
    close = close.dropna()
    labels = label_regimes(close, window=window, threshold=threshold)

    if len(labels) < 2:
        raise RuntimeError(
            f"Not enough labelled bars. Need > {window} rows; got {len(close)}."
        )

    # FIX 2: verify before touching the matrix
    if not skip_verification:
        ok, verification_periods = verify_labels(close, labels, window=window)
        if not ok:
            raise RuntimeError(
                "Label verification failed — matrix would contain a bull/bear "
                "inversion. Aborting before displaying bad output."
            )
    else:
        ok, verification_periods = True, []

    # FIX 1: both matrices
    P_legacy = build_transition_matrix_overlapping(labels)
    P_stride = build_transition_matrix_stride(labels, stride=window)

    pi_stride = stationary_distribution(P_stride)
    current_state = int(labels.iloc[-1])
    next_probs = P_stride[current_state]

    signal = signal_from_matrix(P_stride, current_state)

    # FIX 3: mode-specific output
    if mode == "FILTER":
        gate = gating_signal(signal, filter_threshold)
        mode_output = {"mode": "FILTER", "gate": gate, "signal": signal}
    else:
        pos = standalone_position(signal, size_cap)
        mode_output = {
            "mode": "STANDALONE",
            "position_size": pos,
            "signal": signal,
            "size_cap": size_cap,
        }

    # Before/after walk-forward (done on stride-only for speed in --json mode)
    bt_stride = walk_forward_backtest(close, labels, use_stride=True, stride=window, min_train=min_train)
    bt_legacy = walk_forward_backtest(close, labels, use_stride=False, stride=window, min_train=min_train)

    hmm_result = _hmm_section(close, labels, hmm)

    return {
        "version": "2.0",
        "source": source,
        "rows": int(len(close)),
        "date_start": str(close.index.min().date()),
        "date_end": str(close.index.max().date()),
        "params": {"window": window, "threshold": threshold, "min_train": min_train},
        "states": STATES,
        "current_regime": STATES[current_state],
        "next_state_probabilities": {
            "bear": float(next_probs[0]),
            "sideways": float(next_probs[1]),
            "bull": float(next_probs[2]),
        },
        "signal": signal,
        "fix1": {
            "overlapping_matrix": [[float(x) for x in row] for row in P_legacy],
            "stride_matrix": [[float(x) for x in row] for row in P_stride],
            "persistence_diagonal_overlapping": {
                s: float(P_legacy[i, i]) for i, s in enumerate(["bear", "sideways", "bull"])
            },
            "persistence_diagonal_stride": {
                s: float(P_stride[i, i]) for i, s in enumerate(["bear", "sideways", "bull"])
            },
        },
        "fix2": {
            "verification_passed": ok,
            "periods_checked": verification_periods,
        },
        "fix3": mode_output,
        "stationary_distribution": {
            "bear": float(pi_stride[0]),
            "sideways": float(pi_stride[1]),
            "bull": float(pi_stride[2]),
        },
        "walk_forward_before_fix1": {
            "sharpe": bt_legacy["sharpe"],
            "max_drawdown": bt_legacy["max_drawdown"],
            "n_trades": bt_legacy["n_trades"],
        },
        "walk_forward_after_fix1": {
            "sharpe": bt_stride["sharpe"],
            "max_drawdown": bt_stride["max_drawdown"],
            "n_trades": bt_stride["n_trades"],
        },
        "hmm": hmm_result,
        "framework": "Roan (@RohOnChain) — Markov 2.0 by Lewis Jackson",
        "disclaimer": (
            "Backtests flatter. The fixed matrix shows uglier, truer numbers — "
            "those are the only ones worth trading."
        ),
    }


def _print_pretty(a: dict) -> None:
    print(
        f"\nMarkov 2.0 — source={a['source']} "
        f"window={a['params']['window']} threshold=±{a['params']['threshold']*100:.0f}%"
    )
    print(f"  {a['rows']} rows | {a['date_start']} -> {a['date_end']}")

    # FIX 1 comparison
    P_legacy = np.array(a["fix1"]["overlapping_matrix"])
    P_stride = np.array(a["fix1"]["stride_matrix"])
    _print_comparison(P_legacy, P_stride)

    # Stationary distribution (stride only — the honest one)
    sd = a["stationary_distribution"]
    print("\nStationary distribution (long-run regime mix, stride-sampled):")
    print(f"       Bear: {sd['bear'] * 100:.2f}%")
    print(f"   Sideways: {sd['sideways'] * 100:.2f}%")
    print(f"       Bull: {sd['bull'] * 100:.2f}%")

    # Current signal
    nsp = a["next_state_probabilities"]
    print(f"\nCurrent regime: {a['current_regime']}")
    print(
        f"  Next-day  →  Bull: {nsp['bull']*100:.2f}%   "
        f"Bear: {nsp['bear']*100:.2f}%   "
        f"Sideways: {nsp['sideways']*100:.2f}%"
    )
    print(f"  Signal (bull_prob − bear_prob): {a['signal']:+.4f}")

    # FIX 3 — mode output
    m = a["fix3"]
    if m["mode"] == "FILTER":
        gate_str = {1: "LONG ALLOWED (+1)", -1: "SHORT ALLOWED (-1)", 0: "FLAT — chop (0)"}
        print(f"\nMode: FILTER — gate = {gate_str.get(m['gate'], str(m['gate']))}")
        print(
            "  Your strategy acts; the gate decides whether it is ALLOWED to act.\n"
            "  Position sizing, entry logic, and stop placement stay with you."
        )
    else:
        print(f"\nMode: STANDALONE — position size = {m['position_size']:+.4f}  (cap={m['size_cap']:.2f})")
        print(
            "  No existing strategy needed. Size is |signal| × cap, signed by direction.\n"
            "  See walk-forward Sharpe and max drawdown below before risking capital."
        )

    # Before/after backtest comparison
    _print_backtest_comparison(
        bt_legacy={
            "sharpe": a["walk_forward_before_fix1"]["sharpe"],
            "max_drawdown": a["walk_forward_before_fix1"]["max_drawdown"],
            "n_trades": a["walk_forward_before_fix1"]["n_trades"],
        },
        bt_stride={
            "sharpe": a["walk_forward_after_fix1"]["sharpe"],
            "max_drawdown": a["walk_forward_after_fix1"]["max_drawdown"],
            "n_trades": a["walk_forward_after_fix1"]["n_trades"],
        },
    )

    # HMM
    hmm = a["hmm"]
    print("\n" + "=" * 70)
    print(" HIDDEN MARKOV MODEL (optional confirmation layer)")
    print("=" * 70)
    if hmm.get("available"):
        for r in hmm["regimes"]:
            print(
                f"  {r['label']:<9s} (latent {r['latent_state']}): "
                f"{r['mean_daily_return']*100:+.3f}% mean daily return"
            )
        ag = hmm["agreement_with_threshold_labels"]
        gl = "GREEN LIGHT" if hmm["green_light"] else "CAUTION"
        print(f"\n  Agreement with threshold labels: {ag*100:.1f}%  [{gl}]")
        print(f"  {hmm['note']}")
    else:
        print(f"\n  HMM skipped: {hmm.get('reason', 'unavailable')}")
        print("  (observable Markov model above is unaffected)")

    print("\n" + "=" * 70)
    print(f" {a['disclaimer']}")
    print(f" {a['framework']}")
    print("=" * 70 + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="markov-2-hedge-fund-method",
        description="Markov 2.0 Hedge Fund Method — three documented flaws fixed.",
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--ticker", help="Symbol via yfinance, e.g. SPY, BTC-USD")
    src.add_argument("--csv", help="Path to your own CSV (date + close columns)")
    parser.add_argument("--years", type=int, default=DEFAULT_YEARS)
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--min-train", type=int, default=DEFAULT_MIN_TRAIN)
    parser.add_argument(
        "--mode",
        choices=["FILTER", "STANDALONE"],
        default="FILTER",
        help=(
            "FILTER: regime gates your existing strategy (default). "
            "STANDALONE: trade the differential directly."
        ),
    )
    parser.add_argument(
        "--filter-threshold",
        type=float,
        default=0.0,
        help="FILTER mode: signal must exceed this magnitude to open a gate (default 0.0)",
    )
    parser.add_argument(
        "--size-cap",
        type=float,
        default=1.0,
        help="STANDALONE mode: max absolute position size as fraction of portfolio (default 1.0)",
    )
    parser.add_argument("--no-hmm", action="store_true")
    parser.add_argument("--json", action="store_true", help="Emit JSON to stdout only")
    parser.add_argument(
        "--skip-verification",
        action="store_true",
        help="Skip FIX 2 label verification (use only for debugging)",
    )
    args = parser.parse_args(argv)

    if not args.json:
        describe_modes()
        print(f"  Running in mode: {args.mode}\n")

    try:
        if args.ticker:
            if not args.json:
                print(f"  Fetching {args.ticker} from Yahoo Finance...", file=sys.stderr)
            close = fetch_ticker(args.ticker, years=args.years)
            source = args.ticker
        else:
            close = load_csv(args.csv)
            source = args.csv

        result = analyze_v2(
            close,
            source=source,
            window=args.window,
            threshold=args.threshold,
            min_train=args.min_train,
            mode=args.mode,
            filter_threshold=args.filter_threshold,
            size_cap=args.size_cap,
            hmm=not args.no_hmm,
            skip_verification=args.skip_verification,
        )
    except Exception as exc:
        if args.json:
            print(json.dumps({"error": str(exc)}))
        else:
            print(f"\nERROR: {exc}\n", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(result, default=str))
    else:
        _print_pretty(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
