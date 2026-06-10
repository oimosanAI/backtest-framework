"""Generate notebooks/significance_and_walkforward.ipynb (kept for reproducibility).

This notebook demonstrates the two pillars added on top of the look-ahead guard:
statistical significance (permutation test, bootstrap, multiple-testing) and
over-fitting detection (walk-forward analysis). As with the sibling study
notebook, results are shown *honestly* -- including when the strategy turns out
to be indistinguishable from luck or to decay out-of-sample.
"""

from __future__ import annotations

import pathlib

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

cells = []

cells.append(
    new_markdown_cell(
        "# Trustworthy Backtest Framework — Significance & Walk-Forward\n"
        "## 信じられるバックテストの三本柱（後半二本）\n"
        "\n"
        "**EN.** A believable backtest must survive three distinct lies. This "
        "notebook covers the second and third:\n"
        "\n"
        "1. **Look-ahead bias** — handled structurally in the engine "
        "(`assert_causal`); recapped briefly below.\n"
        "2. **Could it be luck?** — a *permutation test* and a *bootstrap* "
        "confidence interval (`src/backtest/significance.py`).\n"
        "3. **Did we over-fit?** — *walk-forward analysis* separating in-sample "
        "fit from out-of-sample reality (`src/backtest/walkforward.py`).\n"
        "\n"
        "**JA.** 信じられるバックテストは三つの異なる「嘘」を乗り越える必要があります。"
        "本ノートブックはそのうち二本目と三本目を扱います：\n"
        "\n"
        "1. **ルックアヘッド・バイアス** — エンジン側で構造的に排除（`assert_causal`）。"
        "下記で簡単に再確認します。\n"
        "2. **偶然ではないか？** — *並べ替え検定*と*ブートストラップ*信頼区間。\n"
        "3. **過剰最適化していないか？** — *ウォークフォワード分析*で、インサンプルの"
        "当てはまりとアウトオブサンプルの現実を分離します。\n"
        "\n"
        "> The strategy (a 50/200 MA cross) is deliberately ordinary. The point "
        "is the verification around it, and that we report whatever it says — "
        "even an unflattering verdict.\n"
        "> 戦略（50/200 移動平均クロス）はあえて平凡です。主役はその周囲の検証であり、"
        "結果が芳しくなくても正直に報告します。"
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
        "from src.backtest.data import load_prices\n"
        "from src.backtest.significance import (\n"
        "    permutation_test,\n"
        "    bootstrap_statistic,\n"
        "    best_strategy_permutation_test,\n"
        "    bonferroni_adjust,\n"
        ")\n"
        "from src.backtest.walkforward import run_walkforward\n"
        "\n"
        "# One seed governs every Monte-Carlo result below, so the whole notebook\n"
        "# is reproducible end to end.\n"
        "SEED = 0"
    )
)

cells.append(
    new_markdown_cell(
        "## 1. Data & the strategy under test / データと検証対象の戦略\n"
        "\n"
        "**EN.** We fetch split/dividend-adjusted SPY closes, falling back to a "
        "*clearly labelled* synthetic random walk if offline so the notebook "
        "always runs. A driftless random walk has **no edge to find** — a useful "
        "honesty check: on it, every test below *should* report no significance.\n"
        "\n"
        "**JA.** 分割・配当調整済みの SPY 終値を取得し、オフライン時は*明示ラベル付き*"
        "の合成ランダムウォークにフォールバックします。ドリフトの無いランダムウォークには"
        "**見つけるべき優位性が存在しません**——以下の検定が「有意でない」と正しく報告"
        "するかの誠実性チェックになります。"
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
        "    rng = np.random.default_rng(SEED)\n"
        "    idx = pd.date_range(START, END, freq='B')\n"
        "    # Driftless random walk: deliberately edge-free.\n"
        "    daily = rng.normal(0.0, 0.011, len(idx))\n"
        "    prices = pd.Series(100 * np.exp(np.cumsum(daily)), index=idx, name=TICKER)\n"
        "    DATA_SOURCE = 'SYNTHETIC random walk (offline fallback — no real edge)'\n"
        "\n"
        "print('Data source:', DATA_SOURCE)\n"
        "print('Observations:', len(prices))\n"
        "\n"
        "strategy = MovingAverageCrossStrategy(fast_window=50, slow_window=200)\n"
        "RISK_FREE = 0.02"
    )
)

cells.append(
    new_markdown_cell(
        "## 2. Pillar 1 recap — prove no look-ahead / 第一の柱（再確認）\n"
        "\n"
        "**EN.** Before any significance test means anything, the strategy must be "
        "causal on this data. If `assert_causal` raises, stop — nothing below is "
        "trustworthy.\n"
        "\n"
        "**JA.** 有意性検定に意味を持たせる前に、戦略がこのデータ上で因果的である必要が"
        "あります。`assert_causal` が例外を出したら、以降の結果は信用できません。"
    )
)

cells.append(
    new_code_cell(
        "assert_causal(strategy, prices, n_trials=10, seed=SEED)\n"
        "print(f'{strategy.name}: causality check PASSED — no look-ahead bias.')\n"
        "\n"
        "config = BacktestConfig(commission=0.0005, slippage=0.0005)\n"
        "result = run_backtest(prices, strategy, config)\n"
        "\n"
        "# The permutation test works on the *held* position (already execution-\n"
        "# lagged by the engine) versus the per-bar asset return it was exposed to.\n"
        "bar_returns = prices.pct_change().fillna(0.0)\n"
        "\n"
        "print('Strategy Sharpe:', round(metrics.sharpe_ratio(result.returns, RISK_FREE), 3))"
    )
)

cells.append(
    new_markdown_cell(
        "## 3. Pillar 2a — Permutation test / 並べ替え検定\n"
        "\n"
        "**EN.** Null hypothesis: the strategy's positions carry no information "
        "about the contemporaneous return. We simulate that null by shuffling the "
        "positions against the (in-order) returns thousands of times and rebuilding "
        "the Sharpe each time. The **p-value** is how often luck matches or beats "
        "the real Sharpe. The histogram shows the null distribution; the red line "
        "is the strategy's actual score.\n"
        "\n"
        "**JA.** 帰無仮説：戦略のポジションは同時点リターンについて何の情報も持たない。"
        "ポジションを（順序そのままの）リターンに対して数千回シャッフルし、その都度"
        "シャープを再計算してこの帰無仮説を再現します。**p値**は「偶然が実際のシャープ"
        "以上を出す頻度」です。ヒストグラムが帰無分布、赤線が戦略の実スコアです。"
    )
)

cells.append(
    new_code_cell(
        "perm = permutation_test(\n"
        "    result.positions, bar_returns, n_permutations=3000, seed=SEED\n"
        ")\n"
        "# The permutation tests *timing skill*: position x return, before costs and\n"
        "# before the risk-free adjustment, so this Sharpe differs slightly from the\n"
        "# net, rf-adjusted headline number above.\n"
        "print(f'Observed Sharpe (gross): {perm.observed:.3f}')\n"
        "print(f'Permutation p-value    : {perm.p_value:.4f}')\n"
        "verdict = 'SIGNIFICANT' if perm.p_value < 0.05 else 'NOT significant'\n"
        "print(f'At the 5% level        : {verdict}')"
    )
)

cells.append(
    new_code_cell(
        "fig, ax = plt.subplots(figsize=(10, 4.5))\n"
        "ax.hist(perm.null_distribution, bins=60, color='steelblue', alpha=0.75,\n"
        "        label='null (shuffled) Sharpe')\n"
        "ax.axvline(perm.observed, color='crimson', lw=2,\n"
        "           label=f'observed Sharpe = {perm.observed:.2f}')\n"
        "ax.set_title(f'Permutation null distribution — p = {perm.p_value:.4f}  ({DATA_SOURCE})')\n"
        "ax.set_xlabel('Sharpe ratio under the no-edge null')\n"
        "ax.set_ylabel('frequency')\n"
        "ax.legend()\n"
        "ax.grid(True, alpha=0.3)\n"
        "plt.tight_layout()\n"
        "plt.show()"
    )
)

cells.append(
    new_markdown_cell(
        "## 4. Pillar 2b — Bootstrap confidence interval / ブートストラップ信頼区間\n"
        "\n"
        "**EN.** A single Sharpe number hides its own uncertainty. We resample the "
        "realised returns with replacement and rebuild the Sharpe to get a 95% "
        "confidence interval. **If that interval straddles zero, we cannot even "
        "claim the Sharpe is positive.** The shaded band marks the interval; the "
        "dashed line marks zero.\n"
        "\n"
        "**JA.** 単一のシャープ値はその不確実性を隠します。実現リターンを復元抽出で"
        "リサンプリングしてシャープを再構築し、95% 信頼区間を得ます。**区間が 0 をまたぐ"
        "なら、シャープが正であるとすら主張できません。** 影の帯が信頼区間、破線が 0 です。"
    )
)

cells.append(
    new_code_cell(
        "boot = bootstrap_statistic(\n"
        "    result.returns,\n"
        "    statistic=lambda r: metrics.sharpe_ratio(r, RISK_FREE),\n"
        "    n_resamples=3000,\n"
        "    confidence_level=0.95,\n"
        "    seed=SEED,\n"
        ")\n"
        "print(f'Point Sharpe           : {boot.point_estimate:.3f}')\n"
        "print(f'95% CI                 : [{boot.lower:.3f}, {boot.upper:.3f}]')\n"
        "print(f'Standard error         : {boot.standard_error:.3f}')\n"
        "straddles = boot.lower <= 0.0 <= boot.upper\n"
        "print('CI straddles zero      :', straddles,\n"
        "      '(cannot claim a positive Sharpe)' if straddles else '(Sharpe sign is robust)')"
    )
)

cells.append(
    new_code_cell(
        "fig, ax = plt.subplots(figsize=(10, 4.5))\n"
        "ax.hist(boot.distribution, bins=60, color='seagreen', alpha=0.75,\n"
        "        label='bootstrap Sharpe')\n"
        "ax.axvspan(boot.lower, boot.upper, color='gold', alpha=0.3,\n"
        "           label=f'95% CI [{boot.lower:.2f}, {boot.upper:.2f}]')\n"
        "ax.axvline(boot.point_estimate, color='darkgreen', lw=2,\n"
        "           label=f'point = {boot.point_estimate:.2f}')\n"
        "ax.axvline(0.0, color='black', ls='--', lw=1, label='zero')\n"
        "ax.set_title(f'Bootstrap distribution of the Sharpe ratio  ({DATA_SOURCE})')\n"
        "ax.set_xlabel('Sharpe ratio')\n"
        "ax.set_ylabel('frequency')\n"
        "ax.legend()\n"
        "ax.grid(True, alpha=0.3)\n"
        "plt.tight_layout()\n"
        "plt.show()"
    )
)

cells.append(
    new_markdown_cell(
        "## 5. Pillar 2c — Data snooping / 多重検定（データスヌーピング）\n"
        "\n"
        "**EN.** The deadliest self-deception: trying many parameter sets and "
        "reporting the best. The *best of N* looks good even when none has edge. "
        "Two corrections: a crude **Bonferroni** scaling of the single-strategy "
        "p-value, and a permutation **Reality Check** whose null is the *maximum* "
        "Sharpe across all candidates under the same shuffle — so the "
        "cherry-picking is already baked into the null.\n"
        "\n"
        "**JA.** 最も危険な自己欺瞞は、多数のパラメータを試して最良だけを報告すること"
        "です。*N 個中の最良*は、どれも優位でなくても良く見えます。二つの補正：単一戦略"
        "p値を粗く割り増す**ボンフェローニ**と、同一シャッフル下での全候補の*最大*シャープ"
        "を帰無分布とする並べ替え版**リアリティ・チェック**——後者はチェリーピッキングを"
        "最初から帰無分布に織り込みます。"
    )
)

cells.append(
    new_code_cell(
        "# A small grid of MA settings — exactly the kind of search that invites\n"
        "# data snooping.\n"
        "grid = [\n"
        "    MovingAverageCrossStrategy(fast, slow)\n"
        "    for fast in (10, 20, 50)\n"
        "    for slow in (100, 150, 200)\n"
        "    if fast < slow\n"
        "]\n"
        "positions_list = [run_backtest(prices, s, config).positions for s in grid]\n"
        "\n"
        "rc = best_strategy_permutation_test(\n"
        "    positions_list, bar_returns, n_permutations=3000, seed=SEED\n"
        ")\n"
        "best = grid[rc.best_index]\n"
        "print(f'Candidates tried       : {rc.n_candidates}')\n"
        "print(f'Best candidate         : {best.name}  (Sharpe {rc.observed_best:.3f})')\n"
        "print(f'Reality-Check p-value  : {rc.p_value:.4f}  (multiplicity-corrected)')\n"
        "print(f'Naive Bonferroni bound : {bonferroni_adjust(perm.p_value, len(grid)):.4f}')"
    )
)

cells.append(
    new_code_cell(
        "fig, ax = plt.subplots(figsize=(10, 4.5))\n"
        "ax.hist(rc.null_distribution, bins=60, color='slateblue', alpha=0.75,\n"
        "        label='null: best-of-N Sharpe (shuffled)')\n"
        "ax.axvline(rc.observed_best, color='crimson', lw=2,\n"
        "           label=f'observed best = {rc.observed_best:.2f}')\n"
        "ax.set_title(f'Reality Check — best of {rc.n_candidates} strategies, p = {rc.p_value:.4f}')\n"
        "ax.set_xlabel('maximum Sharpe across candidates under the no-edge null')\n"
        "ax.set_ylabel('frequency')\n"
        "ax.legend()\n"
        "ax.grid(True, alpha=0.3)\n"
        "plt.tight_layout()\n"
        "plt.show()"
    )
)

cells.append(
    new_markdown_cell(
        "## 6. Pillar 3 — Walk-forward analysis / ウォークフォワード分析\n"
        "\n"
        "**EN.** Each fold selects the best MA setting on an **in-sample** window, "
        "then scores that choice on the **out-of-sample** window immediately after "
        "it — data the selection never saw. The gap between the two, the "
        "**degradation**, is the over-fitting made visible. We use the anchored "
        "(expanding) scheme.\n"
        "\n"
        "**JA.** 各フォールドで**インサンプル**窓の最良 MA を選び、その選択を直後の"
        "**アウトオブサンプル**窓で評価します——選択時に見ていないデータです。両者の差"
        "（**劣化, degradation**）が過剰最適化を可視化します。アンカード（拡大窓）方式を"
        "用います。"
    )
)

cells.append(
    new_code_cell(
        "wf = run_walkforward(\n"
        "    prices,\n"
        "    grid,\n"
        "    train_size=500,   # ~2 trading years in-sample\n"
        "    test_size=125,    # ~0.5 trading year out-of-sample\n"
        "    mode='anchored',\n"
        "    config=config,\n"
        ")\n"
        "\n"
        "rows = [\n"
        "    {\n"
        "        'in_sample_end': w.train_end_date.date(),\n"
        "        'oos_end': w.test_end_date.date(),\n"
        "        'chosen': w.best_strategy_name,\n"
        "        'IS Sharpe': round(w.in_sample_score, 3),\n"
        "        'OOS Sharpe': round(w.out_of_sample_score, 3),\n"
        "    }\n"
        "    for w in wf.windows\n"
        "]\n"
        "table = pd.DataFrame(rows)\n"
        "print(f'Folds                  : {len(wf.windows)}')\n"
        "print(f'Mean IS Sharpe         : {wf.mean_in_sample:.3f}')\n"
        "print(f'Mean OOS Sharpe        : {wf.mean_out_of_sample:.3f}')\n"
        "print(f'Degradation (IS - OOS) : {wf.degradation:.3f}')\n"
        "print(f'Walk-forward efficiency: {wf.efficiency:.2f}  (OOS / IS)')\n"
        "table"
    )
)

cells.append(
    new_code_cell(
        "fig, ax = plt.subplots(figsize=(11, 4.5))\n"
        "x = np.arange(len(wf.windows))\n"
        "labels = [w.test_end_date.date() for w in wf.windows]\n"
        "ax.bar(x - 0.2, wf.in_sample_scores, width=0.4, label='in-sample', color='cornflowerblue')\n"
        "ax.bar(x + 0.2, wf.out_of_sample_scores, width=0.4, label='out-of-sample', color='salmon')\n"
        "ax.axhline(wf.mean_in_sample, color='cornflowerblue', ls='--', lw=1,\n"
        "           label=f'mean IS = {wf.mean_in_sample:.2f}')\n"
        "ax.axhline(wf.mean_out_of_sample, color='salmon', ls='--', lw=1,\n"
        "           label=f'mean OOS = {wf.mean_out_of_sample:.2f}')\n"
        "ax.axhline(0.0, color='black', lw=0.8)\n"
        "ax.set_xticks(x)\n"
        "ax.set_xticklabels(labels, rotation=45, ha='right')\n"
        "ax.set_title(f'Walk-forward: in-sample vs out-of-sample Sharpe per fold  ({DATA_SOURCE})')\n"
        "ax.set_ylabel('Sharpe ratio')\n"
        "ax.legend(ncol=2, fontsize=8)\n"
        "ax.grid(True, axis='y', alpha=0.3)\n"
        "plt.tight_layout()\n"
        "plt.show()"
    )
)

cells.append(
    new_markdown_cell(
        "## 7. Honest verdict / 正直な結論\n"
        "\n"
        "**EN.** Read the numbers, not your hopes — and read the *next cell*, which "
        "states the verdict from whatever the data actually produced (it is not "
        "hard-coded, so it stays honest as the data updates). The methodological "
        "point is fixed even when the numbers move:\n"
        "\n"
        "- **Permutation test** answers *could this edge be luck?* A high p-value "
        "means luck reproduces it easily.\n"
        "- **Bootstrap interval** answers *how precise is the Sharpe?* An interval "
        "straddling zero means we cannot even sign it.\n"
        "- **Reality Check** answers *did searching many settings fool us?* It "
        "corrects the best-of-N for the search itself.\n"
        "- **Walk-forward degradation** answers *did we over-fit?* Positive "
        "degradation (IS > OOS) is the over-fitting signature; near-zero or "
        "negative degradation means the in-sample fit did **not** inflate — but "
        "note this alone does *not* establish an edge: a strategy can avoid "
        "over-fitting and still be statistically indistinguishable from zero.\n"
        "\n"
        "A strategy is only believable when it clears **all** of these at once. "
        "*On the synthetic random-walk fallback there is no edge by construction, "
        "so honest tests must fail to find one.*\n"
        "\n"
        "**JA.** 期待ではなく数値を、そして**次のセル**を読んでください。次のセルは実際の"
        "データが出した結果から結論を述べます（ハードコードしていないので、データが更新"
        "されても誠実なままです）。数値が動いても手法上の要点は不変です：\n"
        "\n"
        "- **並べ替え検定**＝*この優位性は偶然か？* p値が高ければ偶然で容易に再現できる。\n"
        "- **ブートストラップ区間**＝*シャープはどれだけ精確か？* 区間が 0 をまたげば符号"
        "すら確定できない。\n"
        "- **リアリティ・チェック**＝*多数の設定を試して自分を欺いたか？* 探索そのものを"
        "補正する。\n"
        "- **ウォークフォワードの劣化**＝*過剰最適化したか？* 正の劣化（IS > OOS）が過剰"
        "最適化の兆候。0 近傍や負なら当てはめは過大評価していない——ただしそれだけでは優位性"
        "の証明にはならない（過剰最適化を免れても 0 と区別できないことはある）。\n"
        "\n"
        "戦略が信じられるのは、これら**すべて**を同時にクリアしたときだけです。*合成ランダム"
        "ウォークでは構造上優位性が無く、誠実な検定はそれを見つけられないはずです。*\n"
        "\n"
        "> The deliverable is the verification, not a winning strategy. A framework "
        "that can tell you 'this is probably nothing' is worth far more than one "
        "that always flatters you.\n"
        "> 成果物は勝てる戦略ではなく検証そのものです。『これはおそらく無価値だ』と言える"
        "フレームワークは、常に甘く評価するものよりはるかに価値があります。"
    )
)

cells.append(
    new_code_cell(
        "# A data-driven verdict: assembled from the actual results above, so it\n"
        "# can never drift out of sync with the numbers.\n"
        "checks = {\n"
        "    'permutation finds a real edge (p < 0.05)': perm.p_value < 0.05,\n"
        "    'bootstrap Sharpe is robustly positive (CI > 0)': boot.lower > 0.0,\n"
        "    'best-of-N survives data snooping (p < 0.05)': rc.p_value < 0.05,\n"
        "    'no over-fitting (degradation <= 0)': wf.degradation <= 0.0,\n"
        "}\n"
        "for label, passed in checks.items():\n"
        "    print(f\"[{'PASS' if passed else 'FAIL'}]  {label}\")\n"
        "\n"
        "believable = checks['permutation finds a real edge (p < 0.05)'] and \\\n"
        "    checks['bootstrap Sharpe is robustly positive (CI > 0)'] and \\\n"
        "    checks['best-of-N survives data snooping (p < 0.05)']\n"
        "print()\n"
        "print('VERDICT:', 'a believable, statistically supported edge.'\n"
        "      if believable else\n"
        "      'NOT a believable edge — consistent with luck. Reported honestly.')"
    )
)

nb = new_notebook(cells=cells)
nb.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}

out = pathlib.Path(__file__).parent / "significance_and_walkforward.ipynb"
nbf.write(nb, str(out))
print("wrote", out)
