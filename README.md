# Trustworthy Backtest Framework

**A backtesting framework built for one purpose: to make a backtest you can
actually believe.**

> Repository: `backtest-framework` · A quant-research portfolio project by
> [@oimosanAI](https://github.com/oimosanAI)

The interesting question in quantitative research is rarely "is this strategy
impressive?" — it is "*can I believe this backtest?*". This project is organised
entirely around the second question. The strategy included (a moving-average
cross) is deliberately ordinary; the rigour around it is the point.

**In one line:** look-ahead bias — the single most common reason a backtest
overstates returns — is made *structurally impossible* here, and then *proven
absent by an automated test* (`assert_causal`). That proof, not the strategy,
is the deliverable.

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

## How the classic pitfalls are avoided

### 1. Look-ahead bias — eliminated *structurally*, not by convention

A signal decided at the close of bar `t` uses only information available up to
`t`. It must not be able to earn the price move that *ended* at `t` — you only
learned of the signal once that move was over. The engine enforces this in
exactly **one place**:

```python
# src/backtest/engine.py
position = target_position.shift(EXECUTION_LAG).fillna(0.0)  # EXECUTION_LAG = 1
```

Because the execution lag lives in the **engine** and not in each strategy, no
strategy — however carelessly written — can trade on the same bar that produced
its signal. The direction of the shift is the whole ballgame and is commented in
detail at the call site.

### 2. Look-ahead bias — *proven absent* by tests

Structure is not enough; we prove it. `assert_causal()` perturbs every price
*after* a cut point and asserts that no signal at or *before* the cut point
changes. A strategy that peeked into the future would fail this immediately. The
test-suite runs it both ways:

- it **passes** for the moving-average cross, and
- it **catches** a deliberately planted forward-looking strategy
  (`_PerfectForesight`), proving the test actually bites.

> "Changing the future must never change the past." If it does, the backtest is
> fiction.

### 3. Transaction costs and slippage

Costs are first-class parameters (`commission`, `slippage`), charged on
**turnover** — including the initial entry from a flat book. A test verifies
both the *direction* (costs are a drag) and the *exact magnitude*.

### 4. Honest, verifiable metrics

Every annualisation factor is a named constant with its reasoning in a comment
(252 trading days; volatility scales with `sqrt(252)`; sample std uses `ddof=1`).
Each metric is tested against a series simple enough to compute by hand
(a 50% drawdown, a 2× over two years → CAGR of `sqrt(2) − 1`, etc.).

### 5. Honest demo

`notebooks/strategy_vs_benchmark_SPY.ipynb` runs the strategy on **real SPY
data** and compares it to
buy-&-hold. On the sample shown, the strategy **loses on every metric** —
including a larger drawdown — and the notebook says so plainly. Showing a losing
result faithfully is the demonstration.

---

## Project layout

```
src/backtest/
  engine.py       # vectorised engine + the look-ahead guard + assert_causal
  strategies.py   # causal signal generators (moving-average cross)
  metrics.py      # annualised return/vol, Sharpe, drawdown, win rate, payoff
  data.py         # yfinance loader with strict validation
tests/            # the main event — see below
notebooks/
  strategy_vs_benchmark_SPY.ipynb   # real-data study, strategy vs benchmark (JA/EN)
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

---

## Running the tests

```bash
pytest                # all tests
pytest -v             # verbose
pytest tests/test_engine.py::TestNoLookAhead   # just the look-ahead proofs
```

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
- **No multiple-testing / overfitting controls.** There is no built-in
  parameter-search penalty, walk-forward split, or deflated Sharpe ratio. The
  causality guarantee says the *single* backtest is honest — it says nothing
  about a parameter you cherry-picked across many runs.

These are limits, not bugs: naming them is part of the same commitment to
honesty that motivates the rest of the project.

---

## License

MIT.
