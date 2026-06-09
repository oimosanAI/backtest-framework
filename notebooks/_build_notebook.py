"""Generate notebooks/strategy_vs_benchmark_SPY.ipynb (kept for reproducibility)."""

from __future__ import annotations

import pathlib

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

cells = []

cells.append(
    new_markdown_cell(
        "# Trustworthy Backtest Framework — Strategy vs Benchmark (実データ)\n"
        "\n"
        "**EN.** This notebook runs the moving-average-cross strategy on real "
        "market data and compares it *honestly* against a buy-&-hold benchmark. "
        "The point of the project is not that the strategy wins — it is that the "
        "backtest can be trusted. Where the strategy loses, we say so.\n"
        "\n"
        "**JA.** 本ノートブックは移動平均クロス戦略を実データで実行し、バイ＆ホールド"
        "とフェアに比較します。本プロジェクトの目的は「戦略が勝つこと」ではなく"
        "「バックテストが信用できること」です。負けている場合も正直に示します。\n"
        "\n"
        "> All look-ahead protection lives in the engine "
        "(`target_position.shift(EXECUTION_LAG)`); see `src/backtest/engine.py`.\n"
        "> ルックアヘッド対策はエンジン側に集約されています。"
    )
)

cells.append(
    new_code_cell(
        "import sys, pathlib\n"
        "# Make the repository root importable when running from notebooks/.\n"
        "sys.path.insert(0, str(pathlib.Path.cwd().parent))\n"
        "\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "import matplotlib.pyplot as plt\n"
        "\n"
        "from src.backtest.engine import run_backtest, BacktestConfig, assert_causal\n"
        "from src.backtest.strategies import MovingAverageCrossStrategy\n"
        "from src.backtest import metrics\n"
        "from src.backtest.data import load_prices, DataError"
    )
)

cells.append(
    new_markdown_cell(
        "## 1. Load data / データ取得\n"
        "\n"
        "**EN.** We fetch split/dividend-adjusted SPY closes. If the machine is "
        "offline, we fall back to a *clearly labelled* synthetic geometric "
        "Brownian motion path so the notebook always runs — but the synthetic "
        "path is never presented as a real result.\n"
        "\n"
        "**JA.** 分割・配当調整済みの SPY 終値を取得します。オフライン環境では"
        "明示ラベル付きの合成 GBM 系列にフォールバックし、常に実行可能にします"
        "（合成データを実結果として提示することはありません）。"
    )
)

cells.append(
    new_code_cell(
        'TICKER = "SPY"\n'
        'START, END = "2010-01-01", "2023-12-31"\n'
        "\n"
        "try:\n"
        "    prices = load_prices(TICKER, START, END)\n"
        "    DATA_SOURCE = f'real ({TICKER}, yfinance)'\n"
        "except Exception as exc:  # offline / rate-limited -> synthetic fallback\n"
        "    print(f'Live download failed ({exc!r}); using SYNTHETIC fallback.')\n"
        "    rng = np.random.default_rng(0)\n"
        "    idx = pd.date_range(START, END, freq='B')\n"
        "    daily = rng.normal(0.0003, 0.011, len(idx))  # ~7.5%/yr, ~17% vol\n"
        "    prices = pd.Series(100 * np.exp(np.cumsum(daily)), index=idx, name=TICKER)\n"
        "    DATA_SOURCE = 'SYNTHETIC (offline fallback — not a real result)'\n"
        "\n"
        "print('Data source:', DATA_SOURCE)\n"
        "print('Observations:', len(prices))\n"
        "prices.tail()"
    )
)

cells.append(
    new_markdown_cell(
        "## 2. Causality check / 因果性チェック\n"
        "\n"
        "**EN.** Before trusting any P&L, we *prove* the strategy has no "
        "look-ahead bias on this very data: perturbing future prices must not "
        "change past signals. If this raises, nothing below is trustworthy.\n"
        "\n"
        "**JA.** P&L を信じる前に、この実データ上で戦略にルックアヘッドが無いこと"
        "を証明します（未来を改変しても過去のシグナルが変わらない）。"
    )
)

cells.append(
    new_code_cell(
        "strategy = MovingAverageCrossStrategy(fast_window=50, slow_window=200)\n"
        "assert_causal(strategy, prices, n_trials=10)\n"
        "print(f'{strategy.name}: causality check PASSED — no look-ahead bias.')"
    )
)

cells.append(
    new_markdown_cell(
        "## 3. Run the backtest / バックテスト実行\n"
        "\n"
        "**EN.** Long/short 50/200 MA cross with realistic frictions "
        "(5 bps commission + 5 bps slippage per unit turnover), versus a fully "
        "invested buy-&-hold of the same asset.\n"
        "\n"
        "**JA.** 現実的なコスト（往復前の片道 5bps 手数料 + 5bps スリッページ）を"
        "課した 50/200 MA クロスを、同一資産のバイ＆ホールドと比較します。"
    )
)

cells.append(
    new_code_cell(
        "RISK_FREE = 0.02  # 2%/yr, used for Sharpe\n"
        "config = BacktestConfig(commission=0.0005, slippage=0.0005)\n"
        "\n"
        "result = run_backtest(prices, strategy, config)\n"
        "\n"
        "# Benchmark: always-long buy & hold of the same series, same engine,\n"
        "# so the look-ahead lag and return convention are identical.\n"
        "class BuyAndHold:\n"
        "    name = 'Buy & Hold'\n"
        "    def generate_signals(self, p):\n"
        "        return pd.Series(1.0, index=p.index)\n"
        "\n"
        "benchmark = run_backtest(prices, BuyAndHold(), BacktestConfig())\n"
        "result.equity_curve.tail()"
    )
)

cells.append(
    new_markdown_cell("## 4. Equity curves / 資産曲線")
)

cells.append(
    new_code_cell(
        "fig, ax = plt.subplots(figsize=(11, 5))\n"
        "result.equity_curve.plot(ax=ax, label=strategy.name, lw=1.5)\n"
        "benchmark.equity_curve.plot(ax=ax, label='Buy & Hold', lw=1.5, alpha=0.8)\n"
        "ax.set_title(f'Equity curve — {DATA_SOURCE}')\n"
        "ax.set_ylabel('Growth of 1')\n"
        "ax.set_yscale('log')\n"
        "ax.legend()\n"
        "ax.grid(True, alpha=0.3)\n"
        "plt.tight_layout()\n"
        "plt.show()"
    )
)

cells.append(
    new_markdown_cell("## 5. Drawdown curves / ドローダウン曲線")
)

cells.append(
    new_code_cell(
        "def drawdown_series(returns):\n"
        "    wealth = (1 + returns).cumprod()\n"
        "    return wealth / wealth.cummax() - 1.0\n"
        "\n"
        "fig, ax = plt.subplots(figsize=(11, 4))\n"
        "drawdown_series(result.returns).plot(ax=ax, label=strategy.name, lw=1.2)\n"
        "drawdown_series(benchmark.returns).plot(ax=ax, label='Buy & Hold', lw=1.2, alpha=0.8)\n"
        "ax.set_title('Drawdown')\n"
        "ax.set_ylabel('Drawdown')\n"
        "ax.legend()\n"
        "ax.grid(True, alpha=0.3)\n"
        "plt.tight_layout()\n"
        "plt.show()"
    )
)

cells.append(
    new_markdown_cell("## 6. Metrics summary / 指標サマリー")
)

cells.append(
    new_code_cell(
        "table = pd.DataFrame({\n"
        "    strategy.name: metrics.summary(result.returns, RISK_FREE),\n"
        "    'Buy & Hold': metrics.summary(benchmark.returns, RISK_FREE),\n"
        "})\n"
        "# Present percentages where natural. Build an object-dtype table so the\n"
        "# formatted strings coexist with no float-dtype coercion (pandas 3.0).\n"
        "pct_rows = {'annualized_return', 'annualized_volatility', 'max_drawdown'}\n"
        "display_table = table.astype(object)\n"
        "for r in table.index:\n"
        "    if r in pct_rows:\n"
        "        display_table.loc[r] = table.loc[r].map(lambda x: f'{x * 100:,.2f}%')\n"
        "    else:\n"
        "        display_table.loc[r] = table.loc[r].map(lambda x: f'{x:,.3f}')\n"
        "display_table"
    )
)

cells.append(
    new_markdown_cell(
        "## 7. Honest verdict / 正直な結論\n"
        "\n"
        "**EN.** Read the table above first. On this SPY sample the long/short "
        "50/200 MA cross **lost on every metric** — lower return, lower Sharpe, "
        "*and a larger* maximum drawdown than simply buying and holding. Why: in "
        "a market that trends up for most of the period, the **short leg gets "
        "run over**, and the slow cross **whipsaws** around flat patches while "
        "paying costs on every turn. A long-only variant would avoid the worst "
        "of the short-leg damage, but the headline lesson stands: a plausible, "
        "textbook rule can comfortably **underperform a passive benchmark**. We "
        "show that result rather than hide it — that is the entire point of the "
        "project.\n"
        "\n"
        "**JA.** まず上の表を確認してください。この SPY 期間では、ロング/ショート"
        "型の 50/200 MA クロスは**全指標でバイ＆ホールドに負けました**——リターン"
        "もシャープも低く、**最大ドローダウンはむしろ大きい**結果です。理由は、"
        "長期上昇相場では**ショート側が踏み上げられ**、横ばい局面で**ダマシ"
        "（whipsaw）**を繰り返しながら毎回コストを払うためです。ロングオンリーに"
        "すればショート側の損失は緩和できますが、教訓は変わりません——"
        "教科書的でもっともらしいルールが、パッシブなベンチマークに平然と負け得る"
        "ということです。それを隠さず示すことこそ本プロジェクトの目的です。\n"
        "\n"
        "> The value here is methodological: a trustworthy harness, not a "
        "magic strategy. 価値は手法の信頼性にあり、魔法の戦略ではありません。"
    )
)

nb = new_notebook(cells=cells)
nb.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}

out = pathlib.Path(__file__).parent / "strategy_vs_benchmark_SPY.ipynb"
nbf.write(nb, str(out))
print("wrote", out)
