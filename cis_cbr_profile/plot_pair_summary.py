#!/usr/bin/env python3
"""
plot_pair_summary.py
====================
Per-chromosome dot plot of cognate CBR interaction across developmental
timepoints.

Input is the ``*_pair_summary.txt`` table written by
``cis_cbr_profile.py --partner_mode pairwise_end --pair_summary``.

Three things this script does that the built-in summary plot does not:

1. **Excludes coverage-flagged CBRs.** A cognate CBR contact needs mappable
   sequence at both ends, so one unmappable window drives the measured value to
   zero for purely technical reasons. Rows with ``low_coverage == True`` are
   dropped.

2. **Collapses to one value per somatic chromosome.** The two CBRs of a
   chromosome (``chrNN.L`` and ``chrNN.R``) are computed from largely
   overlapping read sets and are not independent observations; several
   chromosomes carry identical values at both ends. Plotting them separately, as
   the built-in summary figure does, doubles the apparent n. Values are averaged
   within ``somatic`` and reported as n = number of chromosomes.

3. **Plots timepoints only, with no CBR grouping.** In the published analysis the
   gap-size and segment-size splits both returned null results, so colouring by
   them would imply a difference that is not supported. All regions are drawn in
   one colour.

Axis labels use developmental stage rather than timepoint: the summary table
keys its columns by hour, but the figures are labelled by cell stage. The
built-in map covers the *Parascaris* timecourse (10hr, 17hr, 36hr ->
1-2 cells (pre-PDE), 2-4 cells (during PDE), 4-8 cells (post-PDE)); use
``--display_labels`` for any other set.

The panel is drawn at its final printed size (default 3.125 in = 225 pt wide)
rather than large-and-scaled, so type sizes are the ones that reach the page.
Points are laid out as a beeswarm rather than randomly jittered, which keeps the
distribution readable at that width, and exponents in the P-value annotations are
drawn as raised runs of ordinary characters rather than Unicode superscripts,
which many fonts render incompletely.

By default the germline library is excluded, to match the embryonic timecourse
used elsewhere in the figure.

Statistics (optional, requires scipy): because the same chromosomes are measured
at every stage, consecutive timepoints are compared with a paired Wilcoxon
signed-rank test rather than an unpaired test.

Outputs
-------
  <output_prefix>.{png,svg,pdf}   dot plot
  <output_prefix>_table.txt       the per-chromosome values actually plotted

Example
-------
  python plot_pair_summary.py \\
      --summary cbr_pairwise_output/cbr_profile_pair_summary.txt \\
      --stages 10hr 17hr 36hr \\
      --metric oe \\
      --output_dir figures \\
      --output_prefix pair_summary_by_chrom
"""

import argparse
import logging
import re
import os
import sys

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Figure standards: Arial, no top/right spines, large axis labels,
# Illustrator-editable SVG text.
# ---------------------------------------------------------------------------
# Figure standards are applied at the chosen point size rather than fixed, because
# this panel is built at its final printed width (~225 pt) instead of being drawn
# large and scaled down. Text is left as text in SVG for Illustrator.
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
matplotlib.rcParams["svg.fonttype"] = "none"
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["axes.spines.top"] = False
matplotlib.rcParams["axes.spines.right"] = False


def _apply_style(base):
    """Set type sizes from one base size, so the panel scales coherently."""
    matplotlib.rcParams["axes.labelsize"] = base
    matplotlib.rcParams["axes.titlesize"] = base
    matplotlib.rcParams["xtick.labelsize"] = base * 0.92
    matplotlib.rcParams["ytick.labelsize"] = base * 0.92
    matplotlib.rcParams["axes.linewidth"] = max(0.5, base / 12.0)
    matplotlib.rcParams["xtick.major.width"] = max(0.5, base / 12.0)
    matplotlib.rcParams["ytick.major.width"] = max(0.5, base / 12.0)
    matplotlib.rcParams["xtick.major.size"] = base * 0.35
    matplotlib.rcParams["ytick.major.size"] = base * 0.35


# Default stage labels. The summary table keys columns by timepoint, but the
# figures use developmental stage, so map one to the other unless overridden.
STAGE_LABELS = {
    "10hr": "1\u20132 cells\n(pre-PDE)",
    "17hr": "2\u20134 cells\n(during PDE)",
    "36hr": "4\u20138 cells\n(post-PDE)",
    "germline": "germline",
}

POINT_COLOR = "#3B6FB6"

log = logging.getLogger("pair_dot")


# ===========================================================================
# Data preparation
# ===========================================================================
def load_summary(path):
    """Read the pair-summary table written by cis_cbr_profile.py."""
    df = pd.read_csv(path, sep="\t")
    required = {"name", "somatic"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required column(s): {sorted(missing)}")
    return df


def detect_stages(df, metric_suffix):
    """Infer available stage labels from the '<stage>__<metric>' columns."""
    return [c[: -len(metric_suffix)] for c in df.columns if c.endswith(metric_suffix)]


def drop_low_coverage(df):
    """
    Remove CBRs flagged as coverage-limited.

    Older summary tables predate the coverage QC and lack the column; in that
    case nothing is dropped and a warning is emitted, because the resulting
    figure may contain technical zeros.
    """
    if "low_coverage" not in df.columns:
        log.warning("No 'low_coverage' column in the summary table -- this file "
                    "predates the coverage QC. Regenerate it if any region shows "
                    "an unexplained zero.")
        return df, []
    flagged = df.loc[df["low_coverage"].astype(bool), "name"].tolist()
    kept = df.loc[~df["low_coverage"].astype(bool)].copy()
    if flagged:
        log.info("Excluded %d coverage-flagged CBRs: %s",
                 len(flagged), ", ".join(flagged))
    return kept, flagged


def collapse_to_chromosomes(df, stages, metric_suffix):
    """
    Average the two CBRs of each somatic chromosome into a single value.

    Returns a DataFrame indexed by `somatic` with one column per stage.
    """
    cols = {f"{s}{metric_suffix}": s for s in stages}
    sub = df[["somatic"] + list(cols)].rename(columns=cols)
    out = sub.groupby("somatic", as_index=False).mean(numeric_only=True)
    n_cbr = df.groupby("somatic", as_index=False).size().rename(
        columns={"size": "n_cbr"})
    return out.merge(n_cbr, on="somatic")


# ===========================================================================
# Statistics
# ===========================================================================
def build_comparisons(stages, mode):
    """Which stage pairs to test: consecutive only, or every pair."""
    if mode == "none":
        return []
    if mode == "adjacent":
        return list(zip(stages[:-1], stages[1:]))
    return [(a, b) for i, a in enumerate(stages) for b in stages[i + 1:]]


def run_tests(tab, comparisons, test):
    """
    Compare stage pairs.

    The default is a Wilcoxon signed-rank test, which is paired: the same
    chromosomes are measured at every timepoint, so an unpaired test would
    discard that structure and lose power. Mann-Whitney is offered for cases
    where the same regions are not present in both groups.

    Returns a list of (stage_a, stage_b, n, statistic, p_value). Returns an
    empty list if scipy is unavailable, so the figure still renders.
    """
    try:
        from scipy.stats import wilcoxon, mannwhitneyu
    except ImportError:
        log.warning("scipy not available; skipping statistics and annotation")
        return []

    results = []
    for a, b in comparisons:
        m = tab[a].notna() & tab[b].notna()
        x, y = tab.loc[m, a].values, tab.loc[m, b].values
        if m.sum() < 3:
            results.append((a, b, int(m.sum()), np.nan, np.nan))
            continue
        if test == "wilcoxon":
            # Wilcoxon drops zero differences; an all-zero difference vector is
            # degenerate and would raise.
            if np.allclose(y - x, 0):
                results.append((a, b, int(m.sum()), np.nan, np.nan))
                continue
            stat, pv = wilcoxon(x, y)
        else:
            stat, pv = mannwhitneyu(x, y, alternative="two-sided")
        results.append((a, b, int(m.sum()), float(stat), float(pv)))
    return results


def format_p(p, style):
    """Render a p-value for display on the figure."""
    if not np.isfinite(p):
        return "n.d."
    if style == "stars":
        if p < 0.001:
            return "***"
        if p < 0.01:
            return "**"
        if p < 0.05:
            return "*"
        return "n.s."
    if style == "plain":
        return f"P = {p:.3g}"
    # Exponents are emitted as ^{...} markup and drawn as a raised, reduced-size
    # run of ordinary characters. Unicode superscripts are not safe here: Arial
    # carries the superscript digits but not U+207B (superscript minus), so the
    # exponent loses its sign once a viewer substitutes its own font.
    if p >= 0.001:
        return f"P = {p:.3g}"
    exp = int(np.floor(np.log10(p)))
    mant = p / (10.0 ** exp)
    return f"P = {mant:.1f}\u00d710^{{{exp}}}"


def draw_rich_center(ax, x_data, y_data, text, fontsize,
                     sup_scale=0.72, sup_rise=0.33):
    """
    Draw text centred on a data point, rendering ``^{...}`` as true superscript.

    Segment widths are measured from the font and converted through the axes
    transform, so the run stays centred and every character remains editable
    text in the SVG.
    """
    fig = ax.figure
    rend = fig.canvas.get_renderer()
    parts = [q for q in re.split(r"(\^\{[^}]*\})", text) if q]

    seg = []
    for q in parts:
        sup = q.startswith("^{")
        inner = q[2:-1] if sup else q
        fs = fontsize * sup_scale if sup else fontsize
        probe = fig.text(0, 0, inner, fontsize=fs)
        w_px = probe.get_window_extent(renderer=rend).width
        probe.remove()
        seg.append((inner, fs, sup, w_px))

    total_px = sum(t[3] for t in seg)
    x_px, y_px = ax.transData.transform((x_data, y_data))
    cx = x_px - total_px / 2.0
    inv = ax.transData.inverted()
    for inner, fs, sup, w_px in seg:
        dy_px = sup_rise * fontsize * fig.dpi / 72.0 if sup else 0.0
        xd, yd = inv.transform((cx, y_px + dy_px))
        ax.text(xd, yd, inner, ha="left", va="baseline", fontsize=fs,
                clip_on=False, zorder=6)
        cx += w_px


def beeswarm_offsets(values, spacing_x, row_h):
    """
    Symmetric side-by-side offsets for points that would otherwise overlap.

    Random jitter turns 33 points into an indistinct blob once the panel is only
    a couple of inches wide. Binning by value and fanning each row out from the
    centre keeps every point visible and makes the shape of the distribution
    readable at small size.
    """
    values = np.asarray(values, dtype=float)
    offs = np.zeros(len(values))
    if row_h <= 0:
        return offs
    rows = np.round(values / row_h).astype(int)
    for r in np.unique(rows):
        idx = np.where(rows == r)[0]
        idx = idx[np.argsort(values[idx])]
        for j, i in enumerate(idx):
            k = (j + 1) // 2
            offs[i] = k * spacing_x * (1 if j % 2 else -1)
    return offs


def draw_brackets(ax, stages, tests, p_style, y_positions, fontsize, lw, span):
    """Significance brackets in the reserved band above the data."""
    idx = {s_: i for i, s_ in enumerate(stages)}
    tick = 0.016 * span
    for (a_, b_, n_, stat, pv), y in zip(tests, y_positions):
        xa, xb = idx[a_], idx[b_]
        ax.plot([xa, xa, xb, xb], [y - tick, y, y, y - tick],
                color="black", lw=lw, solid_capstyle="butt", clip_on=False,
                zorder=6)
        draw_rich_center(ax, (xa + xb) / 2.0, y + 0.012 * span,
                         format_p(pv, p_style), fontsize)


def dot_plot(tab, stages, ylabel, title, out_base, dpi, seed=0,
             connect=False, show_iqr=False, ylim=None,
             tests=None, p_style="sci", display=None,
             width=3.125, height=3.0, base_fs=8.0, n_style="onplot",
             swarm=True, point_size=13.0):
    """
    One column per stage, one point per somatic chromosome.

    Laid out for a final width of roughly 225 pt, so sizes are chosen to be
    legible without further scaling.
    """
    _apply_style(base_fs)
    rng = np.random.default_rng(seed)
    n_per = [int(np.isfinite(tab[s_].values.astype(float)).sum()) for s_ in stages]

    fig, ax = plt.subplots(figsize=(width, height))

    # ---- vertical extent: reserve a band above the data for the brackets ----
    allv = np.concatenate([tab[s_].values.astype(float) for s_ in stages])
    allv = allv[np.isfinite(allv)]
    data_top = float(np.nanmax(allv)) if allv.size else 1.0
    y0 = 0.0
    n_tiers = len(tests) if tests else 0
    gap, tier_h = 0.075, 0.115
    span0 = max(data_top - y0, 1e-9)
    head = (gap + n_tiers * tier_h + 0.05) * span0 if n_tiers else 0.04 * span0
    top = ylim[1] if ylim else data_top + head
    ax.set_ylim(ylim[0] if ylim else y0, top)
    ax.set_xlim(-0.62, len(stages) - 0.38)

    # Marker geometry converted into data units, so swarm spacing and row height
    # track the real printed size of a point.
    fig.canvas.draw()
    bbox = ax.get_window_extent()
    px_per_pt = fig.dpi / 72.0
    d_pt = 2.0 * np.sqrt(point_size / np.pi)
    x_units = (d_pt * px_per_pt) / bbox.width * (ax.get_xlim()[1] - ax.get_xlim()[0])
    y_units = (d_pt * px_per_pt) / bbox.height * (ax.get_ylim()[1] - ax.get_ylim()[0])

    if connect:
        vals = tab[stages].values
        ok = np.isfinite(vals).all(axis=1)
        for row in vals[ok]:
            ax.plot(range(len(stages)), row, color="grey", lw=0.4, alpha=0.35,
                    zorder=1)

    for xi, stage in enumerate(stages):
        y = tab[stage].values.astype(float)
        yv = y[np.isfinite(y)]
        if yv.size == 0:
            continue
        if swarm:
            off = beeswarm_offsets(yv, x_units * 1.02, y_units * 0.95)
            off = np.clip(off, -0.34, 0.34)
        else:
            off = (rng.random(yv.size) - 0.5) * 0.30
        ax.scatter(np.full(yv.size, xi) + off, yv,
                   s=point_size, alpha=0.85, color=POINT_COLOR,
                   edgecolor="white", linewidth=0.25, zorder=3)
        if show_iqr:
            q1, q3 = np.percentile(yv, [25, 75])
            ax.vlines(xi, q1, q3, color="black", lw=base_fs / 9.0, zorder=4)
        ax.hlines(np.median(yv), xi - 0.32, xi + 0.32,
                  color="black", lw=base_fs / 4.6, zorder=5)

    # ---- axis labels; n either folded into the label or set on the plot ----
    shown = display if display else [STAGE_LABELS.get(s_, s_) for s_ in stages]
    if n_style == "parenthetical":
        lab = []
        for d, nn in zip(shown, n_per):
            if d.endswith(")"):
                lab.append(d[:-1] + f", n = {nn})")
            else:
                lab.append(d + f"\n(n = {nn})")
        shown = lab
    ax.set_xticks(range(len(stages)))
    ax.set_xticklabels(shown)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)

    if n_style == "onplot":
        if len(set(n_per)) == 1:
            # Constant across stages, so state it once instead of three times.
            ax.text(0.015, 0.985, f"n = {n_per[0]}", transform=ax.transAxes,
                    ha="left", va="top", fontsize=base_fs * 0.75, color="0.35")
        else:
            for xi, nn in enumerate(n_per):
                ax.text(xi, ax.get_ylim()[0], f"n = {nn}", ha="center",
                        va="bottom", fontsize=base_fs * 0.75, color="0.35")

    # Lay out before the brackets so their measured offsets use final transforms.
    fig.tight_layout(pad=0.35)
    fig.canvas.draw()
    if tests:
        ys = [data_top + gap * span0 + t * tier_h * span0 for t in range(n_tiers)]
        order = sorted(range(n_tiers),
                       key=lambda i: abs(stages.index(tests[i][1])
                                         - stages.index(tests[i][0])))
        draw_brackets(ax, stages, [tests[i] for i in order], p_style, ys,
                      base_fs * 0.80, max(0.6, base_fs / 10.0), span0)

    for ext in ("png", "svg", "pdf"):
        fig.savefig(f"{out_base}.{ext}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# ===========================================================================
# CLI
# ===========================================================================
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Per-chromosome dot plot of paired-end CBR interaction.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--summary", required=True,
                   help="*_pair_summary.txt from cis_cbr_profile.py "
                        "(--partner_mode pairwise_end --pair_summary)")
    p.add_argument("--stages", nargs="+", default=["10hr", "17hr", "36hr"],
                   help="Stage labels to plot, in order. Germline is omitted by "
                        "default to match the embryonic timecourse.")
    p.add_argument("--metric", choices=["oe", "raw", "max"], default="oe",
                   help="Which per-CBR value to plot: normalised mean (oe), "
                        "raw summed counts (raw), or peak bin (max)")
    p.add_argument("--keep_low_coverage", action="store_true",
                   help="Do NOT exclude coverage-flagged CBRs (diagnostic only)")
    p.add_argument("--per_cbr", action="store_true",
                   help="Plot one point per CBR instead of per chromosome. "
                        "Diagnostic only: the two CBRs of a chromosome are not "
                        "independent, so this inflates the apparent n.")
    p.add_argument("--connect", action="store_true",
                   help="Draw faint lines joining each chromosome across stages")
    p.add_argument("--iqr", action="store_true",
                   help="Draw an interquartile bar alongside the median")
    p.add_argument("--width", type=float, default=3.125,
                   help="Panel width in inches (3.125 in = 225 pt)")
    p.add_argument("--height", type=float, default=3.0,
                   help="Panel height in inches")
    p.add_argument("--fontsize", type=float, default=8.0,
                   help="Base type size in points; ticks, brackets, and the n "
                        "annotation are derived from it")
    p.add_argument("--point_size", type=float, default=13.0,
                   help="Marker area in points squared")
    p.add_argument("--n_style", choices=["onplot", "parenthetical", "none"],
                   default="onplot",
                   help="Where the sample size goes: small text on the plot "
                        "(single annotation when n is the same at every stage), "
                        "folded into the x-axis label, or omitted")
    p.add_argument("--no_swarm", action="store_true",
                   help="Use random jitter instead of the beeswarm layout")
    p.add_argument("--display_labels", nargs="+", default=None,
                   help="X-axis tick labels, one per --stages entry. Use to show "
                        "developmental stages (e.g. '1-2 cells (pre-PDE)') while "
                        "the summary table keeps its timepoint column names. "
                        "Embed a newline with \\n for a two-line label.")
    p.add_argument("--annotate_p", choices=["none", "adjacent", "all"],
                   default="adjacent",
                   help="Significance brackets above the data: consecutive "
                        "stage pairs only, every pair, or none")
    p.add_argument("--p_format", choices=["sci", "stars", "plain"], default="sci",
                   help="How p-values are rendered on the figure")
    p.add_argument("--test", choices=["wilcoxon", "mannwhitney"],
                   default="wilcoxon",
                   help="wilcoxon is paired (same chromosomes at every stage) "
                        "and is the correct default here")
    p.add_argument("--ylabel", default=None,
                   help="Y-axis label (default depends on --metric)")
    p.add_argument("--title", default="",
                   help="Plot title (default: none, for figure-panel use)")
    p.add_argument("--ylim", type=float, nargs=2, default=None,
                   metavar=("LOW", "HIGH"), help="Y-axis limits")
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--seed", type=int, default=0, help="Jitter RNG seed")
    p.add_argument("--output_dir", default=".")
    p.add_argument("--output_prefix", default="pair_summary_by_chrom")
    p.add_argument("--debug", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )
    for noisy in ("matplotlib", "matplotlib.font_manager", "fontTools", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    # Resolve inputs before changing directory so relative paths keep working.
    summary_path = os.path.abspath(args.summary)
    outdir = os.path.abspath(args.output_dir)
    os.makedirs(outdir, exist_ok=True)
    os.chdir(outdir)
    log.info("Working directory: %s", outdir)

    metric_suffix = {"oe": "__pair_oe",
                     "raw": "__pair_raw",
                     "max": "__pair_max"}[args.metric]

    df = load_summary(summary_path)
    available = detect_stages(df, metric_suffix)
    log.info("Stages available in %s: %s",
             os.path.basename(summary_path), ", ".join(available))

    unknown = [s for s in args.stages if s not in available]
    if unknown:
        raise ValueError(f"Requested stage(s) {unknown} not in the summary table; "
                         f"available: {available}")

    if not args.keep_low_coverage:
        df, flagged = drop_low_coverage(df)
    else:
        flagged = []
        log.warning("Keeping coverage-flagged CBRs (--keep_low_coverage)")

    if args.per_cbr:
        tab = df[["name", "somatic"] + [f"{s}{metric_suffix}" for s in args.stages]]
        tab = tab.rename(columns={f"{s}{metric_suffix}": s for s in args.stages})
        log.warning("Plotting per CBR (n=%d): the two ends of a chromosome are "
                    "not independent observations", len(tab))
    else:
        tab = collapse_to_chromosomes(df, args.stages, metric_suffix)
        log.info("Collapsed to %d somatic chromosomes (%d CBRs retained)",
                 len(tab), len(df))

    table_path = f"{args.output_prefix}_table.txt"
    tab.to_csv(table_path, sep="\t", index=False, float_format="%.6g")

    # Medians and paired tests for the figure legend.
    log.info("medians: %s",
             "  ".join(f"{s}={np.nanmedian(tab[s].values.astype(float)):.3f}"
                       for s in args.stages))
    tests = run_tests(tab, build_comparisons(args.stages, args.annotate_p),
                      args.test)
    for a, b, n, stat, pv in tests:
        log.info("%s %s vs %s: n=%d  stat=%.1f  p=%.3g",
                 args.test, a, b, n, stat, pv)

    ylabel = args.ylabel or {
        "oe": "Paired-end contacts (O/E)",
        "raw": "Paired-end contacts (raw)",
        "max": "Paired-end contacts (peak bin, O/E)",
    }[args.metric]

    display = args.display_labels
    if display:
        if len(display) != len(args.stages):
            raise ValueError("--display_labels must have one entry per --stages")
        display = [d.replace("\\n", "\n") for d in display]

    dot_plot(tab, args.stages, ylabel, args.title, args.output_prefix,
             args.dpi, seed=args.seed, connect=args.connect,
             show_iqr=args.iqr,
             ylim=tuple(args.ylim) if args.ylim else None,
             tests=tests, p_style=args.p_format, display=display,
             width=args.width, height=args.height, base_fs=args.fontsize,
             n_style=args.n_style, swarm=not args.no_swarm,
             point_size=args.point_size)

    log.info("Wrote %s.{png,svg,pdf} and %s", args.output_prefix, table_path)
    if flagged:
        log.info("Excluded from the figure: %s", ", ".join(flagged))


if __name__ == "__main__":
    main()
