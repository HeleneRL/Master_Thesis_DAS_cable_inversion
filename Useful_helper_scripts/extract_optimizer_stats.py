"""
extract_optimizer_stats.py
==========================
Reads optimizer history CSV files and extracts statistics that show when the data term (cost_rel) stops improving vs when regularization (cost_prior, cost_curv, cost_spacing) continues to change.

Usage:
    python extract_optimizer_stats.py --files run_215.csv run_813warm.csv
                                      --labels "215 cp" "813 cp warm"

Outputs:
    - Prints a summary table per run
    - Saves a figure: optimizer_data_vs_reg.png
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

# ── tuneable thresholds ───────────────────────────────────────────────────────
FLAT_THRESHOLD   = 0.01    # 1% of total range
ROLLING_WINDOW   = 5000


def load(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df


def find_saturation_point(series, threshold=FLAT_THRESHOLD, window=ROLLING_WINDOW):
    """
    Return the evaluation index where `series` is considered flat.
    Flatness = rolling range / TOTAL RANGE of series < threshold.
    Using total range as denominator makes this scale-invariant,
    so warm-start runs (already near the floor) are treated correctly.
    Returns None if it never flattens.
    """
    total_range = series.max() - series.min()
    if total_range == 0:
        return 0
    rolling_range = series.rolling(window).apply(
        lambda x: (x.max() - x.min()) / total_range
    )
    flat = rolling_range < threshold
    idx = flat.idxmax() if flat.any() else None
    # idxmax returns 0 if no True found — check it's actually True
    if idx is not None and not flat.iloc[idx]:
        idx = None
    return idx


def summarise(df, label):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Total evaluations : {len(df)}")
    print(f"  Initial cost_total: {df['cost_total'].iloc[0]:,.1f}")
    print(f"  Final   cost_total: {df['cost_total'].iloc[-1]:,.1f}")
    print(f"  Total reduction   : {df['cost_total'].iloc[0] - df['cost_total'].iloc[-1]:,.1f}  "
          f"({(df['cost_total'].iloc[0] - df['cost_total'].iloc[-1]) / df['cost_total'].iloc[0] * 100:.1f}%)")

    terms = ['cost_rel', 'cost_prior', 'cost_curv', 'cost_spacing']
    sat_evals = {}
    for t in terms:
        if t not in df.columns:
            continue
        s = find_saturation_point(df[t])
        sat_evals[t] = s
        start = df[t].iloc[0]
        end   = df[t].iloc[-1]
        total_range = df[t].max() - df[t].min()
        tag = f"saturates at eval ~{s}" if s is not None else "never flat within budget"
        print(f"\n  {t:<16}  start={start:>14.3f}  end={end:>14.3f}  range={total_range:>14.3f}  → {tag}")

    print(f"\n  Cost reduction breakdown:")
    total_drop = df['cost_total'].iloc[0] - df['cost_total'].iloc[-1]
    for t in terms:
        if t not in df.columns:
            continue
        drop = df[t].iloc[0] - df[t].iloc[-1]
        print(f"    {t:<16}  Δ={drop:>14.3f}  ({drop/total_drop*100 if total_drop else 0:.1f}% of total)")

    return sat_evals


def plot_runs(runs, output='optimizer_data_vs_reg.png'):
    n = len(runs)
    fig, axes = plt.subplots(2, n, figsize=(7 * n, 9), squeeze=False)
    fig.patch.set_facecolor('white')

    blue   = '#1d6fa4'
    orange = '#e07b1a'
    green  = '#2a9d6e'
    red    = '#c0392b'

    for col, (df, label) in enumerate(runs):
        evals = df['eval'] if 'eval' in df.columns else df.index

        # ── top: data term ────────────────────────────────────────────────
        ax_top = axes[0][col]
        ax_top.set_facecolor('white')

        if 'cost_rel' in df.columns:
            ax_top.plot(evals, df['cost_rel'], color=blue, lw=1.8, zorder=3,
                        label='Relative traveltime (data term)')

            s = find_saturation_point(df['cost_rel'])
            y_max = df['cost_rel'].max()
            y_min = df['cost_rel'].min()

            # if s is not None:
            #     ax_top.axvline(s, color=blue, lw=1.2, ls='--', alpha=0.6)
            #     ax_top.text(s, y_max - (y_max - y_min) * 0.05,
            #                 f' sat. ~{s:,}', fontsize=9, color=blue,
            #                 va='top', alpha=0.8)
            # else:
            #     # annotate that it never saturates
            #     ax_top.text(0.98, 0.95, 'never saturates\nwithin budget',
            #                 transform=ax_top.transAxes, fontsize=9,
            #                 color=blue, ha='right', va='top', alpha=0.8,
            #                 style='italic')

        ax_top.set_title(f'{label}\nData term (relative traveltime)',
                         fontsize=13, pad=8, color='#222222')
        ax_top.set_xlabel('Function evaluation', fontsize=12)
        ax_top.set_ylabel('cost_rel', fontsize=12, color=blue)
        ax_top.tick_params(axis='y', colors=blue, labelsize=11)
        ax_top.tick_params(axis='x', labelsize=11)
        ax_top.spines['top'].set_visible(False)
        ax_top.spines['right'].set_visible(False)
        ax_top.spines['left'].set_color(blue)
        ax_top.spines['bottom'].set_color('#bbbbbb')
        ax_top.yaxis.grid(True, color='#eeece6', lw=0.7, zorder=0)
        ax_top.set_axisbelow(True)
        ax_top.legend(fontsize=10, framealpha=0.8)

        # secondary right axis: % of total range remaining
        if 'cost_rel' in df.columns:
            ax_pct = ax_top.twinx()
            total_range = df['cost_rel'].max() - df['cost_rel'].min()
            if total_range > 0:
                pct_range_remaining = (df['cost_rel'] - df['cost_rel'].min()) / total_range * 100
                ax_pct.plot(evals, pct_range_remaining, color=blue, lw=1.8, alpha=0.0)
                # sync limits so axis labels match
                ax_pct.set_ylim(
                    (ax_top.get_ylim()[0] - df['cost_rel'].min()) / total_range * 100,
                    (ax_top.get_ylim()[1] - df['cost_rel'].min()) / total_range * 100,
                )
            ax_pct.set_ylabel('% of total range remaining', fontsize=10, color='#888880')
            ax_pct.tick_params(axis='y', colors='#888880', labelsize=10)
            ax_pct.spines['right'].set_color('#cccccc')
            for sp in ['top', 'left', 'bottom']:
                ax_pct.spines[sp].set_visible(False)

        # ── bottom: regularization terms ──────────────────────────────────
        ax_bot = axes[1][col]
        ax_bot.set_facecolor('white')

        reg_terms = {
            'cost_prior':   (orange, 'Prior penalty'),
            'cost_curv':    (green,  'Curvature penalty'),
            'cost_spacing': (red,    'Spacing penalty'),
        }
        for col_name, (color, lbl) in reg_terms.items():
            if col_name in df.columns:
                vals = df[col_name].replace(0, np.nan)
                ax_bot.plot(evals, vals, color=color, lw=1.8, zorder=3, label=lbl)

                # saturation marker for reg terms too
                s_reg = find_saturation_point(df[col_name].fillna(method='ffill'))
                if s_reg is not None:
                    ax_bot.axvline(s_reg, color=color, lw=0.9, ls=':', alpha=0.5)

        ax_bot.set_yscale('log')
        ax_bot.set_title('Regularization terms', fontsize=13, pad=8, color='#222222')
        ax_bot.set_xlabel('Function evaluation', fontsize=12)
        ax_bot.set_ylabel('Cost (log scale)', fontsize=12)
        ax_bot.tick_params(axis='both', labelsize=11)
        ax_bot.spines['top'].set_visible(False)
        ax_bot.spines['right'].set_visible(False)
        ax_bot.spines['bottom'].set_color('#bbbbbb')
        ax_bot.yaxis.grid(True, color='#eeece6', lw=0.7, zorder=0)
        ax_bot.set_axisbelow(True)
        ax_bot.legend(fontsize=10, framealpha=0.8)

        x_max = evals.max() if hasattr(evals, 'max') else max(evals)
        ax_top.set_xlim(0, x_max)
        ax_bot.set_xlim(0, x_max)

    fig.suptitle('Data term saturation vs regularization evolution',
                 fontsize=15, y=1.01, color='#222222')
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\nFigure saved: {output}")


# ── main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Extract optimizer saturation statistics and plot data vs regularization.')
    parser.add_argument('--files',  nargs='+', required=True,
                        help='Paths to optimizer history CSV files')
    parser.add_argument('--labels', nargs='+', default=None,
                        help='Labels for each run (same order as --files)')
    parser.add_argument('--output', default='optimizer_data_vs_reg.png',
                        help='Output figure filename')
    parser.add_argument('--threshold', type=float, default=FLAT_THRESHOLD,
                        help=f'Flatness threshold as fraction of total range (default: {FLAT_THRESHOLD})')
    parser.add_argument('--window', type=int, default=ROLLING_WINDOW,
                        help=f'Rolling window for flatness detection (default: {ROLLING_WINDOW})')
    args = parser.parse_args()

    labels = args.labels if args.labels else [Path(f).stem for f in args.files]
    assert len(labels) == len(args.files), "Number of labels must match number of files"

    FLAT_THRESHOLD = args.threshold
    ROLLING_WINDOW = args.window

    runs = []
    for path, label in zip(args.files, labels):
        print(f"Loading {path} ...")
        df = load(path)
        summarise(df, label)
        runs.append((df, label))

    plot_runs(runs, output=args.output)