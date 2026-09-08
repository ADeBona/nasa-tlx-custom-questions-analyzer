#!/usr/bin/env python3
# ---------------------------------------------------------------------------
# Group-level analysis of NASA-TLX + custom-item results across all
# participants, for the results paper.
#
# Reads the CSV produced by nasa-tlx.py (one row per participant x
# condition) and produces:
#
#   1. Descriptive statistics (n, mean, SD, SEM, 95% CI, median, min, max)
#      for the six raw NASA-TLX subscales, the unweighted overall Raw-TLX
#      score, and every custom Likert item (confidence, usefulness,
#      naturalness) - broken down by condition.
#   2. Within-subject paired comparisons between conditions (paired t-test,
#      Wilcoxon signed-rank test, Cohen's dz) for every measure above.
#   3. Analysis of the free-text "did it feel like torque?" comment, which
#      is only meaningful for conditions that actually render haptic
#      feedback:
#         'X'      -> camera-only condition, question does not apply,
#                     row excluded from this analysis.
#         'Y'/'N'  -> feedback WAS rendered; counted and reported as
#                     percentages per condition.
#      A vibrotactile-alert condition is not part of this study and is
#      dropped if an old row for it is present, with a note in the console.
#   4. Publication-ready figures (PNG, 300 dpi) written to
#      ./analysis_output/.
#   5. A Markdown summary (analysis_output/results_summary.md) with every
#      number pre-formatted ("mean ± SD") for direct use in the paper.
#
# Requires: pandas, numpy, matplotlib. scipy is optional (enables the
# paired-comparison tests and 95% CIs); without it those sections are
# skipped with a warning instead of crashing.
#
# Run:  python3 analyze_results.py [path/to/nasa-tlx-results.csv]
# ---------------------------------------------------------------------------

import sys
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from scipy import stats as sstats
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False

# ===========================================================================
# CONFIGURATION
# ===========================================================================

DEFAULT_RESULTS_FILE = "nasa-tlx-results.csv"
OUTPUT_DIR_NAME = "analysis_output"

ID_COLS = ["experiment", "user_id", "condition"]
COMMENT_COL = "comment"

# The two conditions actually run, in canonical (baseline-first) order.
# Mirrors the `conditions` list in nasa-tlx.py - keep the two in sync.
CONDITION_ORDER = ["Camera only", "Camera + torque display"]

# Conditions to drop before analysis (substring match, case-insensitive).
# A vibrotactile-alert condition was considered early on but never run and
# is no longer offered in nasa-tlx.py; this is just a safeguard in case an
# old row from that option ever turns up in the CSV.
EXCLUDED_CONDITION_PATTERNS = ["vibrotactile"]

# Torque-interpretation comment codes.
COMMENT_NOT_APPLICABLE = "X"   # camera-only: no haptic feedback to judge
COMMENT_VALUES = ["Y", "N"]    # only meaningful when feedback was present

CONFIDENCE_LEVEL = 0.95

# --- Validated categorical palette (dataviz skill reference palette) ------
# Fixed hue order, assigned via CONDITION_ORDER (baseline first) so colors
# stay stable across figures and across re-runs as more participants are added.
PALETTE = [
    "#2a78d6",  # slot 1 - blue
    "#eb6834",  # slot 2 - orange
    "#1baf7a",  # slot 3 - aqua
    "#eda100",  # slot 4 - yellow
    "#e87ba4",  # slot 5 - magenta
]
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"
STATUS_GOOD = "#0ca30c"
STATUS_CRITICAL = "#d03b3b"

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": BASELINE,
    "axes.labelcolor": INK_PRIMARY,
    "text.color": INK_PRIMARY,
    "xtick.color": INK_SECONDARY,
    "ytick.color": INK_SECONDARY,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 11,
})

TLX_SUBSCALE_NAMES = {
    "tlx_mental": "Mental\nDemand",
    "tlx_physical": "Physical\nDemand",
    "tlx_temporal": "Temporal\nDemand",
    "tlx_performance": "Performance",
    "tlx_effort": "Effort",
    "tlx_frustration": "Frustration",
}

# ===========================================================================
# Small stats helpers (so the script degrades gracefully without scipy)
# ===========================================================================


def sem(series):
    """Standard error of the mean; NaN if fewer than 2 observations."""
    series = pd.Series(series).dropna()
    if len(series) < 2:
        return np.nan
    return series.std(ddof=1) / np.sqrt(len(series))


def ci95_halfwidth(series):
    """Half-width of the 95% CI of the mean, using a t critical value.
    NaN if fewer than 2 observations or scipy is unavailable."""
    series = pd.Series(series).dropna()
    n = len(series)
    if n < 2:
        return np.nan
    se = series.std(ddof=1) / np.sqrt(n)
    if HAVE_SCIPY:
        tcrit = sstats.t.ppf(1 - (1 - CONFIDENCE_LEVEL) / 2, df=n - 1)
    else:
        tcrit = 1.96  # large-sample normal approximation fallback
    return tcrit * se


def fmt(mean, sd, n, decimals=1):
    if n == 0 or pd.isna(mean):
        return "n/a"
    if n == 1 or pd.isna(sd):
        return f"{mean:.{decimals}f} (n=1)"
    return f"{mean:.{decimals}f} ± {sd:.{decimals}f}"


def sort_conditions(conditions):
    """Order conditions with the baseline first (per CONDITION_ORDER, which
    mirrors nasa-tlx.py's `conditions` list), so it always gets the first
    palette color and plots first left-to-right. Any condition not in that
    list (e.g. old data, or a condition added later) is appended
    alphabetically after the known ones."""
    conditions = list(dict.fromkeys(conditions))  # de-dupe, keep order
    known = [c for c in CONDITION_ORDER if c in conditions]
    unknown = sorted(c for c in conditions if c not in CONDITION_ORDER)
    return known + unknown


def axis_label(condition):
    """Wrap a condition name onto two lines so it fits compactly as an
    axis tick without truncating it to something unrecognizable."""
    return "\n".join(textwrap.wrap(str(condition), width=14)) or str(condition)


def style_axes(ax):
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=GRIDLINE, linewidth=0.8, zorder=0)
    ax.xaxis.grid(False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)


# ===========================================================================
# Load & prepare data
# ===========================================================================


def load_data(csv_path):
    df = pd.read_csv(csv_path, dtype={"user_id": str})
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip()

    excluded_mask = df["condition"].str.contains(
        "|".join(EXCLUDED_CONDITION_PATTERNS), case=False, regex=True)
    if excluded_mask.any():
        dropped = sorted(df.loc[excluded_mask, "condition"].unique())
        print(f"Excluding {excluded_mask.sum()} row(s) from not-yet-run "
              f"condition(s): {', '.join(dropped)}")
        df = df.loc[~excluded_mask].copy()

    tlx_cols = [c for c in df.columns if c.startswith("tlx_")]
    custom_cols = [c for c in df.columns
                   if c not in ID_COLS + tlx_cols + [COMMENT_COL]]

    for col in tlx_cols + custom_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["tlx_overall"] = df[tlx_cols].mean(axis=1)

    return df, tlx_cols, custom_cols


# ===========================================================================
# Descriptive statistics
# ===========================================================================


def descriptive_table(df, measure_cols):
    rows = []
    for condition in sort_conditions(df["condition"].unique()):
        sub = df[df["condition"] == condition]
        for col in measure_cols:
            values = sub[col].dropna()
            n = len(values)
            mean = values.mean() if n else np.nan
            sd = values.std(ddof=1) if n > 1 else np.nan
            rows.append({
                "condition": condition,
                "measure": col,
                "n": n,
                "mean": mean,
                "sd": sd,
                "sem": sem(values),
                "ci95_halfwidth": ci95_halfwidth(values),
                "median": values.median() if n else np.nan,
                "min": values.min() if n else np.nan,
                "max": values.max() if n else np.nan,
            })
    return pd.DataFrame(rows)


# ===========================================================================
# Paired within-subject comparisons
# ===========================================================================


def paired_comparisons(df, measure_cols):
    conditions = sort_conditions(df["condition"].unique())
    rows = []
    for i in range(len(conditions)):
        for j in range(i + 1, len(conditions)):
            cond_a, cond_b = conditions[i], conditions[j]
            wide = df.pivot_table(index="user_id", columns="condition",
                                   values=measure_cols, aggfunc="mean")
            for col in measure_cols:
                if (col, cond_a) not in wide.columns or (col, cond_b) not in wide.columns:
                    continue
                paired = wide[[(col, cond_a), (col, cond_b)]].dropna()
                n = len(paired)
                a = paired[(col, cond_a)].values
                b = paired[(col, cond_b)].values
                row = {
                    "measure": col,
                    "condition_a": cond_a,
                    "condition_b": cond_b,
                    "n_pairs": n,
                    "mean_diff (a-b)": np.mean(a - b) if n else np.nan,
                }
                if n >= 2 and HAVE_SCIPY:
                    diff = a - b
                    dz = np.mean(diff) / np.std(diff, ddof=1) if np.std(diff, ddof=1) > 0 else np.nan
                    t_res = sstats.ttest_rel(a, b)
                    row["t"] = t_res.statistic
                    row["p_ttest"] = t_res.pvalue
                    row["cohens_dz"] = dz
                    try:
                        w_res = sstats.wilcoxon(a, b)
                        row["W"] = w_res.statistic
                        row["p_wilcoxon"] = w_res.pvalue
                    except ValueError:
                        # all differences zero, or n too small
                        row["W"] = np.nan
                        row["p_wilcoxon"] = np.nan
                else:
                    row["t"] = row["p_ttest"] = row["cohens_dz"] = np.nan
                    row["W"] = row["p_wilcoxon"] = np.nan
                rows.append(row)
    return pd.DataFrame(rows)


# ===========================================================================
# Torque-interpretation comment
# ===========================================================================


def comment_analysis(df):
    """Only rows coded Y or N are analysable (X = camera-only, N/A)."""
    codes = df[COMMENT_COL].str.strip().str.upper()
    applicable = df.loc[codes.isin(COMMENT_VALUES)].copy()
    applicable["comment_code"] = codes.loc[applicable.index]

    n_total = len(df)
    n_na = int((codes == COMMENT_NOT_APPLICABLE).sum())
    n_unlabelled = n_total - n_na - len(applicable)

    per_condition = (
        applicable.groupby("condition")["comment_code"]
        .value_counts()
        .unstack(fill_value=0)
    )
    for code in COMMENT_VALUES:
        if code not in per_condition.columns:
            per_condition[code] = 0
    per_condition = per_condition[COMMENT_VALUES]
    per_condition["n"] = per_condition.sum(axis=1)
    per_condition["pct_Y"] = np.where(
        per_condition["n"] > 0, 100 * per_condition["Y"] / per_condition["n"], np.nan)
    per_condition["pct_N"] = np.where(
        per_condition["n"] > 0, 100 * per_condition["N"] / per_condition["n"], np.nan)
    per_condition = per_condition.reindex(sort_conditions(per_condition.index))

    overall_n = len(applicable)
    overall_pct_y = 100 * (applicable["comment_code"] == "Y").sum() / overall_n if overall_n else np.nan

    return {
        "applicable": applicable,
        "per_condition": per_condition,
        "n_total_rows": n_total,
        "n_not_applicable": n_na,
        "n_unlabelled": n_unlabelled,
        "overall_n": overall_n,
        "overall_pct_y": overall_pct_y,
    }


# ===========================================================================
# Plotting
# ===========================================================================


def color_map_for(conditions):
    ordered = sort_conditions(conditions)
    return {c: PALETTE[i % len(PALETTE)] for i, c in enumerate(ordered)}


def plot_overall_tlx(df, out_dir, colors):
    conditions = sort_conditions(df["condition"].unique())
    means = [df.loc[df.condition == c, "tlx_overall"].mean() for c in conditions]
    sems_ = [sem(df.loc[df.condition == c, "tlx_overall"]) for c in conditions]

    fig, ax = plt.subplots(figsize=(5.5, 5))
    x = np.arange(len(conditions))
    ax.bar(x, means, width=0.5, color=[colors[c] for c in conditions],
           yerr=sems_, capsize=4, zorder=3,
           error_kw=dict(ecolor=INK_SECONDARY, elinewidth=1.2))

    # Individual participants, connected across conditions they completed.
    pivot = df.pivot_table(index="user_id", columns="condition",
                            values="tlx_overall", aggfunc="mean")
    pivot = pivot.reindex(columns=conditions)
    for _, row in pivot.iterrows():
        present = [(i, row[c]) for i, c in enumerate(conditions) if pd.notna(row[c])]
        if len(present) >= 2:
            xs, ys = zip(*present)
            ax.plot(xs, ys, color=INK_MUTED, alpha=0.5, linewidth=1, zorder=2)
        for i, y in present:
            ax.scatter(i, y, color=INK_PRIMARY, s=20, zorder=4, alpha=0.75)

    ax.set_xticks(x)
    ax.set_xticklabels([axis_label(c) for c in conditions])
    ax.set_ylabel("Raw NASA-TLX overall score (0–100)")
    ax.set_ylim(0, 100)
    ax.set_title("Overall workload by condition")
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(out_dir / "overall_tlx.png", dpi=300)
    plt.close(fig)


def plot_subscales(df, tlx_cols, out_dir, colors):
    conditions = sort_conditions(df["condition"].unique())
    n_cond = len(conditions)
    width = 0.8 / n_cond
    x = np.arange(len(tlx_cols))

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for i, c in enumerate(conditions):
        sub = df[df.condition == c]
        means = [sub[col].mean() for col in tlx_cols]
        sems_ = [sem(sub[col]) for col in tlx_cols]
        offset = (i - (n_cond - 1) / 2) * width
        ax.bar(x + offset, means, width=width * 0.9, color=colors[c],
               yerr=sems_, capsize=3, zorder=3, label=c,
               error_kw=dict(ecolor=INK_SECONDARY, elinewidth=1))

    ax.set_xticks(x)
    ax.set_xticklabels([TLX_SUBSCALE_NAMES.get(c, c) for c in tlx_cols])
    ax.set_ylabel("Rating (0–100)")
    ax.set_ylim(0, 100)
    ax.set_title("Raw NASA-TLX subscales by condition")
    ax.legend(frameon=False, title="Condition")
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(out_dir / "subscales_by_condition.png", dpi=300)
    plt.close(fig)


def plot_custom_items(df, custom_cols, out_dir, colors):
    if not custom_cols:
        return
    conditions = sort_conditions(df["condition"].unique())
    n_cond = len(conditions)
    width = 0.8 / n_cond
    x = np.arange(len(custom_cols))

    fig, ax = plt.subplots(figsize=(7, 5.5))
    for i, c in enumerate(conditions):
        sub = df[df.condition == c]
        means = [sub[col].mean() for col in custom_cols]
        sems_ = [sem(sub[col]) for col in custom_cols]
        offset = (i - (n_cond - 1) / 2) * width
        ax.bar(x + offset, means, width=width * 0.9, color=colors[c],
               yerr=sems_, capsize=3, zorder=3, label=c,
               error_kw=dict(ecolor=INK_SECONDARY, elinewidth=1))

    ax.set_xticks(x)
    ax.set_xticklabels([col.capitalize() for col in custom_cols])
    ax.set_ylabel("Rating (1–7)")
    ax.set_ylim(1, 7)
    ax.set_title("Custom questionnaire items by condition")
    ax.legend(frameon=False, title="Condition")
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(out_dir / "custom_items_by_condition.png", dpi=300)
    plt.close(fig)


def plot_comment_breakdown(comment_result, out_dir):
    per_condition = comment_result["per_condition"]
    if per_condition.empty:
        print("No Y/N torque-interpretation responses to plot "
              "(all rows were 'X' / camera-only, or the comment column is empty).")
        return

    conditions = list(per_condition.index)
    x = np.arange(len(conditions))
    width = 0.35

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.bar(x - width / 2, per_condition["pct_Y"], width=width,
           color=STATUS_GOOD, zorder=3, label="Felt like torque (Y)")
    ax.bar(x + width / 2, per_condition["pct_N"], width=width,
           color=STATUS_CRITICAL, zorder=3, label="Did not feel like torque (N)")

    for i, cond in enumerate(conditions):
        n = int(per_condition.loc[cond, "n"])
        ax.text(i, 102, f"n={n}", ha="center", va="bottom",
                fontsize=9, color=INK_SECONDARY)

    ax.set_xticks(x)
    ax.set_xticklabels([axis_label(c) for c in conditions])
    ax.set_ylabel("Participants (%)")
    ax.set_ylim(0, 112)
    ax.set_title("Perceived realism of haptic torque feedback")
    ax.legend(frameon=False)
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(out_dir / "comment_torque_breakdown.png", dpi=300)
    plt.close(fig)


# ===========================================================================
# Markdown summary
# ===========================================================================


def write_markdown_summary(path, df, tlx_cols, custom_cols, desc, pairs, comment_result):
    conditions = sort_conditions(df["condition"].unique())
    n_participants = df["user_id"].nunique()
    measure_cols = tlx_cols + ["tlx_overall"] + custom_cols

    lines = []
    lines.append("# NASA-TLX results summary\n")
    lines.append(f"- Participants: **N = {n_participants}**")
    lines.append(f"- Conditions analysed: {', '.join(conditions)}")
    lines.append("- The vibrotactile-alert condition (C2) was not run in this "
                  "study and is excluded.")
    lines.append("- The torque-interpretation comment is analysed only for "
                  "rows coded Y/N (haptic feedback present); rows coded X "
                  "(camera-only, not applicable) are excluded from that part.\n")

    lines.append("## Descriptive statistics\n")
    lines.append("Mean ± SD (n); SEM and 95% CI reported where n ≥ 2.\n")
    lines.append("| Condition | Measure | n | Mean ± SD | SEM | 95% CI half-width | Median | Min | Max |")
    lines.append("|---|---|---|---|---|---|---|---|---|")

    def cell(v, decimals=2):
        return "n/a" if pd.isna(v) else f"{v:.{decimals}f}"

    for _, r in desc.iterrows():
        lines.append(
            f"| {r['condition']} | {r['measure']} | {int(r['n'])} | "
            f"{fmt(r['mean'], r['sd'], r['n'])} | "
            f"{cell(r['sem'])} | "
            f"{cell(r['ci95_halfwidth'])} | "
            f"{cell(r['median'], 1)} | "
            f"{cell(r['min'], 1)} | "
            f"{cell(r['max'], 1)} |")
    lines.append("")

    lines.append("## Within-subject paired comparisons\n")
    if pairs.empty:
        lines.append("Not enough participants completed more than one "
                      "condition to run paired comparisons yet.\n")
    else:
        if not HAVE_SCIPY:
            lines.append("_scipy is not installed - t-test / Wilcoxon "
                          "columns are omitted. Install with `pip install scipy` "
                          "to enable significance testing._\n")
        lines.append("| Measure | A vs B | n pairs | Mean diff (A−B) | "
                      "t | p (paired t) | Cohen's dz | W | p (Wilcoxon) |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for _, r in pairs.iterrows():
            def cell(v, decimals=2):
                return "n/a" if pd.isna(v) else f"{v:.{decimals}f}"
            lines.append(
                f"| {r['measure']} | {r['condition_a']} vs "
                f"{r['condition_b']} | {int(r['n_pairs'])} | "
                f"{cell(r['mean_diff (a-b)'])} | {cell(r.get('t'))} | "
                f"{cell(r.get('p_ttest'), 3)} | {cell(r.get('cohens_dz'))} | "
                f"{cell(r.get('W'))} | {cell(r.get('p_wilcoxon'), 3)} |")
        lines.append("")

    lines.append("## Torque-interpretation comment\n")
    lines.append(f"- Total rows: {comment_result['n_total_rows']}")
    lines.append(f"- Excluded as not applicable (camera-only, coded 'X'): "
                 f"{comment_result['n_not_applicable']}")
    if comment_result["n_unlabelled"]:
        lines.append(f"- Excluded as unlabelled/blank: {comment_result['n_unlabelled']}")
    if comment_result["overall_n"]:
        lines.append(
            f"- Of the {comment_result['overall_n']} responses where haptic "
            f"feedback was present, {comment_result['overall_pct_y']:.1f}% "
            f"reported the feedback felt like torque (Y).\n")
        pc = comment_result["per_condition"]
        lines.append("| Condition | n | Y | N | % felt like torque |")
        lines.append("|---|---|---|---|---|")
        for cond, r in pc.iterrows():
            lines.append(f"| {cond} | {int(r['n'])} | {int(r['Y'])} | "
                         f"{int(r['N'])} | {r['pct_Y']:.1f}% |")
    else:
        lines.append("- No Y/N responses were available to analyse yet.")
    lines.append("")

    lines.append("## Figures\n")
    lines.append("- `overall_tlx.png` - overall Raw-TLX score by condition "
                 "(mean ± SEM, individual participants overlaid)")
    lines.append("- `subscales_by_condition.png` - the six Raw-TLX subscales by condition")
    if custom_cols:
        lines.append("- `custom_items_by_condition.png` - "
                     f"{', '.join(custom_cols)} by condition")
    lines.append("- `comment_torque_breakdown.png` - % of participants who "
                 "reported the haptic feedback felt like torque")

    path.write_text("\n".join(lines), encoding="utf-8")


# ===========================================================================
# Main
# ===========================================================================


def main():
    script_dir = Path(__file__).resolve().parent
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else script_dir / DEFAULT_RESULTS_FILE
    if not csv_path.exists():
        sys.exit(f"Results file not found: {csv_path}")

    out_dir = script_dir / OUTPUT_DIR_NAME
    out_dir.mkdir(exist_ok=True)

    df, tlx_cols, custom_cols = load_data(csv_path)
    measure_cols = tlx_cols + ["tlx_overall"] + custom_cols
    colors = color_map_for(df["condition"].unique())

    print(f"Loaded {len(df)} rows, {df['user_id'].nunique()} participant(s), "
          f"conditions: {', '.join(sort_conditions(df['condition'].unique()))}\n")

    if not HAVE_SCIPY:
        print("NOTE: scipy is not installed - paired significance tests and "
              "95% CIs will be skipped. Run `pip install scipy` to enable them.\n")

    desc = descriptive_table(df, measure_cols)
    desc.to_csv(out_dir / "descriptive_stats.csv", index=False)
    print("=== Descriptive statistics ===")
    with pd.option_context("display.width", 140, "display.max_columns", 20):
        print(desc.round(2).to_string(index=False))
    print()

    pairs = paired_comparisons(df, measure_cols)
    if not pairs.empty:
        pairs.to_csv(out_dir / "paired_comparisons.csv", index=False)
        print("=== Paired within-subject comparisons ===")
        with pd.option_context("display.width", 160, "display.max_columns", 20):
            print(pairs.round(3).to_string(index=False))
        print()
    else:
        print("No participant completed more than one condition yet - "
              "skipping paired comparisons.\n")

    comment_result = comment_analysis(df)
    print("=== Torque-interpretation comment ===")
    print(f"Not applicable (camera-only, 'X'): {comment_result['n_not_applicable']}")
    if comment_result["n_unlabelled"]:
        print(f"Unlabelled/blank: {comment_result['n_unlabelled']}")
    if comment_result["overall_n"]:
        print(comment_result["per_condition"].round(1).to_string())
        print(f"\nOverall: {comment_result['overall_pct_y']:.1f}% of "
              f"{comment_result['overall_n']} applicable responses felt "
              f"the feedback was torque-like.")
    else:
        print("No Y/N responses yet.")
    print()

    plot_overall_tlx(df, out_dir, colors)
    plot_subscales(df, tlx_cols, out_dir, colors)
    plot_custom_items(df, custom_cols, out_dir, colors)
    plot_comment_breakdown(comment_result, out_dir)

    write_markdown_summary(out_dir / "results_summary.md", df, tlx_cols,
                           custom_cols, desc, pairs, comment_result)

    print(f"Figures, CSV tables and results_summary.md written to {out_dir}")


if __name__ == "__main__":
    main()
