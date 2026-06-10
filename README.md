# Trustworthy Backtest Framework

**A backtesting framework built for one purpose: to make a backtest you can
actually believe.**

> Repository: `backtest-framework` · A quant-research portfolio project by
> [@oimosanAI](https://github.com/oimosanAI)

The interesting question in quantitative research is rarely "is this strategy
impressive?" — it is "*can I believe this backtest?*". This project is organised
entirely around the second question. The strategy included (a moving-average
cross) is deliberately ordinary; the rigour around it is the point.

**In one line:** the three classic lies a backtest tells — *look-ahead bias*,
*statistical noise mistaken for skill*, and *over-fitting* — are each met with a
specific, tested defence. Look-ahead bias is made *structurally impossible* and
*proven absent* (`assert_causal`); an edge is checked against *luck* (permutation
test + bootstrap) and against *data snooping* (multiple-testing correction); and
*over-fitting* is exposed by *walk-forward analysis*. The verification, not the
strategy, is the deliverable.

[日本語版 README はこちら / Japanese README](./README.ja.md)

---

## Why this project (relevance to a quant hiring process)

Most blown-up "great" backtests die of the same disease: the simulation quietly
used information that would not have been available at the time of the trade.
A framework that an employer can trust is one where the classic lies a backtest
tells — look-ahead bias above all — are made *structurally impossible* and then
*proven absent by tests*, rather than avoided by careful convention. This
repository is small on purpose so that every one of those guarantees is legible
and checkable in a few minutes of reading.

---

## The three pillars of a believable backtest

A backtest can lie in three distinct ways. Each pillar below names one lie and
the specific, tested defence against it. **A result is only trustworthy when it
clears all three.**

### Pillar 1 — Look-ahead bias · *“did the simulation use information it could not have had?”*

**Eliminated structurally, not by convention.** A signal decided at the close of
bar `t` uses only information available up to `t`. It must not earn the price
move that *ended* at `t` — you only learned of the signal once that move was
over. The engine enforces this in exactly **one place**:

```python
# src/backtest/engine.py
position = target_position.shift(EXECUTION_LAG).fillna(0.0)  # EXECUTION_LAG = 1
```

Because the execution lag lives in the **engine** and not in each strategy, no
strategy — however carelessly written — can trade on the same bar that produced
its signal.

**Proven absent, not merely asserted.** `assert_causal()` perturbs every price
*after* a cut point and asserts that no signal at or *before* the cut point
changes. The test-suite runs it both ways: it **passes** for the moving-average
cross and **catches** a deliberately planted forward-looking strategy
(`_PerfectForesight`), proving the test actually bites.

> "Changing the future must never change the past." If it does, the backtest is
> fiction.

### Pillar 2 — Statistical significance · *“is the edge real, or just luck?”*

A finite sample can hand a coin-flip strategy a flattering Sharpe ratio.
`src/backtest/significance.py` measures whether the edge is distinguishable from
chance — every function takes an explicit `seed`, so every p-value and interval
is exactly reproducible.

- **Permutation test** (`permutation_test`) shuffles the strategy's positions
  against the in-order returns thousands of times, rebuilding the score each
  time. The resulting p-value is *how often pure luck matches or beats the real
  result*. **Lie it catches:** a noise strategy whose good run was coincidence.
- **Bootstrap confidence interval** (`bootstrap_statistic`) resamples the
  realised returns to put an honest error bar around the Sharpe (with an
  optional moving-block variant for serial correlation). **Lie it catches:** a
  point estimate quoted without its uncertainty — if the interval straddles
  zero, you cannot even claim the Sharpe is positive.
- **Multiple-testing / data-snooping correction** (`bonferroni_adjust`,
  `sidak_adjust`, and the permutation Reality Check
  `best_strategy_permutation_test`) accounts for having tried many strategies
  and reported the best. The Reality Check's null is the *maximum* score across
  all candidates under a shared shuffle, so the cherry-picking is baked into the
  null. **Lie it catches:** the best of fifty parameter sets dressed up as a
  single discovery.

### Pillar 3 — Over-fitting · *“was the strategy tuned to the very data it is judged on?”*

`src/backtest/walkforward.py` repeatedly selects parameters on an **in-sample**
window and then scores that choice on the **out-of-sample** window immediately
after it — data the selection never saw. Both **anchored** (expanding) and
**rolling** (fixed) schemes are supported. The gap between in-sample and
out-of-sample performance — the **degradation** — *is* the over-fitting, measured
directly. Crucially, parameter selection touches only the in-sample slice and the
out-of-sample slice lies strictly later in time, so the base engine's look-ahead
guarantee holds inside every window: **out-of-sample information can never leak
into the fit.** **Lie it catches:** a curve fitted to its own answer key.

---

## Supporting rigour

### Transaction costs and slippage

Costs are first-class parameters (`commission`, `slippage`), charged on
**turnover** — including the initial entry from a flat book. A test verifies
both the *direction* (costs are a drag) and the *exact magnitude*.

### Honest, verifiable metrics

Every annualisation factor is a named constant with its reasoning in a comment
(252 trading days; volatility scales with `sqrt(252)`; sample std uses `ddof=1`).
Each metric is tested against a series simple enough to compute by hand
(a 50% drawdown, a 2× over two years → CAGR of `sqrt(2) − 1`, etc.).

### Honest demos

`notebooks/strategy_vs_benchmark_SPY.ipynb` runs the strategy on **real SPY
data** versus buy-&-hold; on the sample shown it **loses on every metric** and
says so plainly. `notebooks/significance_and_walkforward.ipynb` puts the same
strategy through all three pillars and prints a *data-driven* verdict: on real
SPY the apparent edge is **statistically indistinguishable from luck**. Showing a
losing result faithfully is the demonstration.

---

## Project layout

```
src/backtest/
  engine.py        # vectorised engine + the look-ahead guard + assert_causal
  strategies.py    # causal signal generators (moving-average cross)
  metrics.py       # annualised return/vol, Sharpe, drawdown, win rate, payoff
  significance.py  # PILLAR 2: permutation test, bootstrap, multiple-testing
  walkforward.py   # PILLAR 3: anchored/rolling walk-forward + degradation
  data.py          # yfinance loader with strict validation
tests/             # the main event — see below
notebooks/
  strategy_vs_benchmark_SPY.ipynb      # real-data study, strategy vs benchmark (JA/EN)
  significance_and_walkforward.ipynb   # the three pillars, with a data-driven verdict (JA/EN)
```

---

## Installation

```bash
git clone https://github.com/oimosanAI/backtest-framework.git
cd backtest-framework

python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

(Requires Python ≥ 3.10.)

---

## Usage

```python
import yfinance as yf
from src.backtest.data import load_prices
from src.backtest.engine import run_backtest, BacktestConfig, assert_causal
from src.backtest.strategies import MovingAverageCrossStrategy
from src.backtest import metrics

prices = load_prices("SPY", "2010-01-01", "2023-12-31")

strategy = MovingAverageCrossStrategy(fast_window=50, slow_window=200)

# Prove there is no look-ahead bias *before* trusting any P&L.
assert_causal(strategy, prices)

config = BacktestConfig(commission=0.0005, slippage=0.0005)
result = run_backtest(prices, strategy, config)

print(metrics.summary(result.returns, risk_free_rate=0.02))
```

### Pillar 2 — is the edge real, or luck?

```python
from src.backtest.significance import permutation_test, bootstrap_statistic

bar_returns = prices.pct_change().fillna(0.0)

# How often does pure luck match the strategy's timing skill?
perm = permutation_test(result.positions, bar_returns, n_permutations=3000, seed=0)
print(f"permutation p-value: {perm.p_value:.4f}")  # high p => indistinguishable from luck

# An honest error bar on the Sharpe ratio.
boot = bootstrap_statistic(result.returns, n_resamples=3000, seed=0)
print(f"Sharpe 95% CI: [{boot.lower:.2f}, {boot.upper:.2f}]")  # straddling 0 => no claim
```

### Pillar 3 — did we over-fit?

```python
from src.backtest.walkforward import run_walkforward

grid = [
    MovingAverageCrossStrategy(fast, slow)
    for fast in (10, 20, 50)
    for slow in (100, 150, 200)
    if fast < slow
]

wf = run_walkforward(prices, grid, train_size=500, test_size=125, mode="anchored")
print(f"mean in-sample Sharpe : {wf.mean_in_sample:.2f}")
print(f"mean out-of-sample    : {wf.mean_out_of_sample:.2f}")
print(f"degradation (overfit) : {wf.degradation:.2f}")  # positive => over-fitting
```

---

## Running the tests

```bash
pytest                # all tests
pytest -v             # verbose
pytest tests/test_engine.py::TestNoLookAhead              # Pillar 1: look-ahead proofs
pytest tests/test_significance.py                         # Pillar 2: luck vs edge
pytest tests/test_walkforward.py                          # Pillar 3: over-fitting detection
```

The significance and walk-forward tests are validated on **synthetic data with a
known answer** — a permutation test must find *no* edge in random signals and a
*clear* edge in aligned ones; walk-forward must structurally place every
out-of-sample window after its in-sample window and detect inflated in-sample
scores. No test touches the network.

Formatting and linting:

```bash
black src tests
ruff check src tests
```

---

## Limitations (stated honestly)

This framework optimises for *validity of the simulation*, not for breadth of
features. It deliberately does **not** yet address:

- **Survivorship bias.** It backtests whatever tickers you hand it. If you feed
  it only stocks that exist *today*, you have already baked in the survivors;
  the framework does not supply a point-in-time / delisted-inclusive universe.
- **Single-asset only.** The engine runs one price series at a time; there is no
  cross-sectional portfolio construction, position netting, or risk model.
- **Close-to-close fills.** Execution is modelled at the next close via a
  one-bar lag. Intraday fills, partial fills, and the open/close auction gap are
  not modelled.
- **Costs are a simple linear model.** Commission and slippage are proportional
  to turnover; market impact, the bid-ask spread by name, and borrow cost for
  shorts are not modelled.
- **Significance tests assume a representative sample.** The bootstrap and
  permutation tests quantify sampling and timing luck *within the data you
  supply*; they cannot correct for a regime that simply never appears in the
  sample, and the i.i.d. bootstrap understates uncertainty for strongly
  autocorrelated returns (use the moving-block option). A *deflated* Sharpe ratio
  is not yet provided as a closed-form alternative to the Reality Check.
- **Walk-forward selects from a supplied grid.** Optimisation is a search over
  the candidate strategies you pass in, not a continuous optimiser; the
  over-fitting it measures is that of *grid selection*, which is the common case
  but not the only one.

These are limits, not bugs: naming them is part of the same commitment to
honesty that motivates the rest of the project.

---

## License

MIT.
