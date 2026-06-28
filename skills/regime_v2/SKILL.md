---
name: markov-2-hedge-fund-method
description: >-
  Markov 2.0 Hedge Fund Method — three documented flaws from v1 fixed.
  Builds a stride-sampled transition matrix (no overlapping-window autocorrelation),
  self-verifies label mapping before displaying any output, and asks the user to
  choose FILTER or STANDALONE mode at the start of every run.
  Works on any ticker (via yfinance) or user CSV. Framework by Roan (@RohOnChain).
---

# Markov 2.0 — Hedge Fund Method

Slash command: `/markov-2-hedge-fund-method`
Script: `scripts/markov_regime_v2.py`
Skill doc: `skills/regime_v2/SKILL.md`

## The three fixes

### FIX 1 — Stride sampling (the autocorrelation flaw)
The v1 matrix counted day-to-day transitions between consecutive 20-day rolling
windows. Those windows share 19 days, which fakes persistence on the diagonal —
Bear→Bear and Bull→Bull looked far stickier than they actually are.

v2 ALWAYS computes BOTH matrices:
- **Overlapping (legacy):** transitions between every consecutive pair of daily labels
- **Stride-sampled (honest):** transitions between every 20th label so consecutive
  samples come from non-overlapping windows

They are shown side-by-side with a one-line warning. Only the stride-sampled
matrix is used for signal generation and backtesting.

**SPY 10-year example (observed diagonal inflation):**
```
Bear:     77.5% (overlapping)  →  14.3% (stride)   = +63 pp inflation
Sideways: 93.1% (overlapping)  →  85.0% (stride)   = + 8 pp inflation
Bull:     71.9% (overlapping)  →  11.8% (stride)   = +60 pp inflation
```

### FIX 2 — Label verification
After labelling but BEFORE displaying any matrix, the script programmatically
finds three historical reference periods and checks the label assigned to each:

| Check | Period | Expected label |
|---|---|---|
| Worst drawdown | date of most negative 20-day return | Bear |
| Strongest rally | date of most positive 20-day return | Bull |
| Flattest stretch | date closest to zero 20-day return | Sideways |

If any check fails, the script raises an error and exits. The matrix is never
shown when the labels are inverted — this is the exact class of bug in v1.

### FIX 3 — Two explicit modes

**FILTER (default):** The regime gates an existing strategy. The script emits
a gating signal: +1 (longs allowed), -1 (shorts allowed), 0 (flat / chop). The
user's strategy makes entry decisions and controls position sizing; Markov 2.0
decides only WHEN it is allowed to act.

**STANDALONE:** Trade the bull-minus-bear differential directly. Position size =
`signal × size_cap`, signed by direction. The user sets the cap (default 1.0).
The walk-forward Sharpe and max drawdown printed below are what they are signing
up for in this mode.

## Invocation

```bash
# FILTER mode (default)
uv run scripts/markov_regime_v2.py --ticker SPY --mode FILTER

# STANDALONE mode with 50% cap
uv run scripts/markov_regime_v2.py --ticker BTC-USD --mode STANDALONE --size-cap 0.5

# Your own data
uv run scripts/markov_regime_v2.py --csv ./prices.csv --mode FILTER

# JSON output for agent consumption
uv run scripts/markov_regime_v2.py --ticker QQQ --json
```

## JSON contract (key fields)

| Field | Meaning |
|---|---|
| `fix1.stride_matrix` | The honest 3×3 matrix to use for trading |
| `fix1.overlapping_matrix` | Legacy matrix — shown for comparison only |
| `fix2.verification_passed` | true/false — false means a label inversion was caught |
| `fix3.gate` | FILTER mode: +1 / 0 / -1 |
| `fix3.position_size` | STANDALONE mode: signed fraction of portfolio |
| `signal` | bull_prob − bear_prob from the stride matrix, current state |
| `walk_forward_before_fix1` | Legacy backtest (inflated diagonals) |
| `walk_forward_after_fix1` | Honest backtest (stride-sampled) |
| `hmm.green_light` | true if HMM and threshold labels agree >65% |

## Notes

- `--skip-verification` disables FIX 2 — only for debugging. Never use in production.
- HMM requires hmmlearn. If absent, `hmm.available` is false and everything else is valid.
- The walk-forward refits nothing at each step using incremental counting — O(n), no lookahead.
- Defaults: window=20, threshold=±5%, years=10, min_train=252.
