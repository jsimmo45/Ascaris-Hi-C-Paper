#!/usr/bin/env python3
"""
make_null_tests_table.py
========================
Tests for systematic variation in cognate CBR interaction across somatic
chromosomes, rendered both as machine-readable tables and as a typeset panel.

Input is the ``*_pair_summary.txt`` table written by
``cis_cbr_profile.py --partner_mode pairwise_end --pair_summary``.

Three predictors are tested against interaction strength:

  1. Somatic chromosome length          (Spearman rank correlation)
  2. Adjacent eliminated block size     (Spearman rank correlation)
  3. X-derived vs autosomal identity    (two-sided Mann-Whitney U)

against four response variables: the normalised value at each of the three
embryonic timepoints, and the raw within-chromosome ratio of the last two.

Two analysis choices matter and are enforced here:

* **Coverage-flagged CBRs are excluded.** A cognate CBR contact needs mappable
  sequence at both ends, so one unmappable window forces the value to zero for
  technical reasons.

* **Values are collapsed to one per somatic chromosome.** The two CBRs of a
  chromosome are computed from largely overlapping read sets and are not
  independent; treating them as separate observations doubles the apparent n.
  Each chromosome's adjacent-gap covariate is the mean of its two flanking
  eliminated blocks.

Post-elimination caveat
-----------------------
Because E(s) is estimated over the germline coordinate system while
post-elimination material is physically fragmented into separate somatic
chromosomes, single-stage normalised values are not comparable across
chromosomes of differing length at that stage. Columns named with
``--post_elimination`` are marked with a dagger and the raw within-chromosome
ratio is reported alongside as the appropriate test. The caveat is written into
the table footnote automatically.

Typography
----------
The rendered panel is built at its final printed width so no rescaling is
needed, text is left editable in the SVG, footnotes are wrapped and justified to
the measured width of the table rules, and exponents are drawn as raised runs of
ordinary characters rather than Unicode superscripts, which many fonts render
incompletely. ``--ascii`` produces pure-ASCII output for pipelines that mishandle
encodings.

Column headings use developmental stage rather than timepoint, matching the
figures; ``--timepoint_headers`` restores the raw timepoint names.

Outputs
-------
  <output_prefix>.tsv        long form, one row per predictor x response
  <output_prefix>_wide.tsv   wide form, one row per predictor
  <output_prefix>.txt        aligned text
  <output_prefix>.{svg,pdf,png}  typeset panel; place the SVG in Illustrator

Example
-------
  python make_null_tests_table.py \\
      --summary cbr_pairwise_output/cbr_profile_pair_summary.txt \\
      --stages 10hr 17hr 36hr \\
      --ratio 36hr 17hr \\
      --post_elimination 36hr \\
      --output_dir tables --output_prefix null_tests_table
"""

import argparse
import logging
import os
import sys

import re
import textwrap

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Figure standards: Arial, and text kept as text in SVG so the panel can be
# edited in Illustrator rather than arriving as outlines.
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
matplotlib.rcParams["svg.fonttype"] = "none"
matplotlib.rcParams["pdf.fonttype"] = 42

log = logging.getLogger("table_s")

# The summary table keys its columns by timepoint, but the figures
# label by developmental stage. Column headings follow the figures; the short
# form is used because the table is narrow.
STAGE_LABELS = {
    "10hr": "1\u20132 cells",
    "17hr": "2\u20134 cells",
    "36hr": "4\u20138 cells",
    "germline": "germline",
}
STAGE_LONG = {
    "10hr": "1\u20132 cells (pre-PDE)",
    "17hr": "2\u20134 cells (during PDE)",
    "36hr": "4\u20138 cells (post-PDE)",
}


# ===========================================================================
# Data preparation
# ===========================================================================
def load_and_collapse(path, stages, ratio_pair, keep_low_coverage=False,
                      x_prefix="chrX"):
    """
    Read the summary table, drop coverage-flagged CBRs, and collapse to one
    row per somatic chromosome.

    Returns (DataFrame, list_of_response_column_names, n_flagged).
    """
    df = pd.read_csv(path, sep="\t")

    flagged = []
    if "low_coverage" in df.columns and not keep_low_coverage:
        flagged = df.loc[df["low_coverage"].astype(bool), "name"].tolist()
        df = df.loc[~df["low_coverage"].astype(bool)].copy()
        if flagged:
            log.info("Excluded %d coverage-flagged CBRs: %s",
                     len(flagged), ", ".join(flagged))
    elif "low_coverage" not in df.columns:
        log.warning("No 'low_coverage' column: this summary predates the "
                    "coverage QC. Regenerate it before using these numbers.")

    for s in stages:
        if f"{s}__pair_oe" not in df.columns:
            raise ValueError(f"stage '{s}' not found in {os.path.basename(path)}")

    # Per-chromosome aggregation. seg_len is identical for both CBRs of a
    # chromosome; gap differs (left vs right flank) and is averaged.
    agg = {f"{s}__pair_oe": "mean" for s in stages}
    agg.update({"seg_len": "first", "gap": "mean"})
    num, den = ratio_pair
    if num and den:
        agg[f"{num}__pair_raw"] = "sum"
        agg[f"{den}__pair_raw"] = "sum"

    c = df.groupby("somatic", as_index=False).agg(agg)
    c = c.rename(columns={f"{s}__pair_oe": s for s in stages})
    c = c.rename(columns={"gap": "mean_gap"})

    responses = list(stages)
    if num and den:
        with np.errstate(divide="ignore", invalid="ignore"):
            c["ratio"] = np.where(c[f"{den}__pair_raw"] > 0,
                                  c[f"{num}__pair_raw"] / c[f"{den}__pair_raw"],
                                  np.nan)
        responses.append("ratio")

    c["isX"] = c["somatic"].str.startswith(x_prefix)
    log.info("Collapsed to %d somatic chromosomes (%d X-derived, %d autosomal)",
             len(c), int(c.isX.sum()), int((~c.isX).sum()))
    return c, responses, flagged


# ===========================================================================
# Tests
# ===========================================================================
def spearman_row(c, predictor, responses):
    """Spearman rank correlation of a continuous predictor against each response."""
    from scipy.stats import spearmanr
    out = {}
    for r in responses:
        m = c[predictor].notna() & c[r].notna()
        if m.sum() < 3:
            out[r] = (np.nan, np.nan, int(m.sum()))
            continue
        rho, p = spearmanr(c.loc[m, predictor], c.loc[m, r])
        out[r] = (float(rho), float(p), int(m.sum()))
    return out


def mwu_row(c, responses):
    """Two-sided Mann-Whitney U comparing X-derived with autosomal chromosomes."""
    from scipy.stats import mannwhitneyu
    out = {}
    for r in responses:
        x = c.loc[c.isX, r].dropna()
        a = c.loc[~c.isX, r].dropna()
        if len(x) < 2 or len(a) < 2:
            out[r] = (np.nan, np.nan, np.nan, len(x), len(a))
            continue
        _, p = mannwhitneyu(x, a, alternative="two-sided")
        out[r] = (float(np.median(x)), float(np.median(a)), float(p),
                  len(x), len(a))
    return out


# ===========================================================================
# Formatting
# ===========================================================================
ASCII = False   # set by --ascii; swaps Unicode symbols for plain equivalents


def sym(name):
    """Symbol lookup honouring --ascii."""
    table = {
        "rho": ("\u03c1", "rho"),
        "times": ("\u00d7", "x"),
        "endash": ("\u2013", "-"),
    }
    uni, asc = table[name]
    return asc if ASCII else uni


def fmt_p(p):
    """P-values: three significant figures, scientific below 0.001."""
    if not np.isfinite(p):
        return "n.d."
    if p >= 0.01:
        return f"{p:.3f}"
    if p >= 0.001:
        return f"{p:.4f}"
    exp = int(np.floor(np.log10(p)))
    if ASCII:
        return f"{p / 10 ** exp:.1f}x10^{exp}"
    # Markup form; converted to Unicode for text outputs and drawn as a true
    # raised run in the rendered panel.
    return f"{p / 10 ** exp:.1f}\u00d710^{{{exp}}}"


def markup_to_unicode(txt):
    """Turn '10^{-6}' into '10\u207b\u2076' for the plain-text outputs."""
    def _conv(m):
        return _sup(m.group(1))
    return re.sub(r"\^\{([^}]*)\}", _conv, txt)


def _sup(n):
    sup = {"0": "\u2070", "1": "\u00b9", "2": "\u00b2", "3": "\u00b3",
           "4": "\u2074", "5": "\u2075", "6": "\u2076", "7": "\u2077",
           "8": "\u2078", "9": "\u2079", "-": "\u207b", "\u2212": "\u207b"}
    return "".join(sup.get(ch, ch) for ch in str(n))


def build_rows(c, responses, stages, ratio_pair):
    """Assemble the long-form table: one record per predictor x response."""
    rows = []
    # Same developmental-stage naming as the rendered panel and the figures.
    label = {s: f"{_pretty_col(s)} (O/E)" for s in stages}
    num, den = ratio_pair
    if "ratio" in responses:
        label["ratio"] = f"{_pretty_col(f'{num}/{den}')} (raw ratio)"

    for pred, key, pretty in [
        ("seg_len", "seg_len", "Somatic chromosome length"),
        ("mean_gap", "mean_gap", "Adjacent eliminated block size"),
    ]:
        res = spearman_row(c, key, responses)
        for r in responses:
            rho, p, n = res[r]
            rows.append(dict(predictor=pretty, test=f"Spearman {sym('rho')}",
                             response=label[r], n=n,
                             statistic=(f"{sym('rho')} = {rho:+.3f}"
                                        if np.isfinite(rho) else "n.d."),
                             effect="", p_value=fmt_p(p), p_raw=p))

    res = mwu_row(c, responses)
    for r in responses:
        mx, ma, p, nx, na = res[r]
        rows.append(dict(
            predictor="X-derived vs autosomal",
            test=f"Mann{sym('endash')}Whitney U",
            response=label[r], n=f"{nx} vs {na}",
            statistic="",
            effect=(f"{mx:.3f} vs {ma:.3f}" if np.isfinite(mx) else "n.d."),
            p_value=fmt_p(p), p_raw=p))
    return pd.DataFrame(rows)


def write_aligned(tab, path, n_chrom, flagged, ratio_pair):
    """Write a fixed-width version that can be pasted straight into a document."""
    cols = ["predictor", "response", "n", "statistic", "effect", "p_value"]
    head = {"predictor": "Predictor", "response": "Response", "n": "n",
            "statistic": "Statistic", "effect": "Median (X vs auto)",
            "p_value": "P"}
    w = {c: max(len(head[c]), *(len(str(v)) for v in tab[c])) for c in cols}

    lines = []
    lines.append("Tests for systematic variation in cognate CBR "
                 "interaction between somatic chromosomes.")
    lines.append("")
    lines.append("  ".join(head[c].ljust(w[c]) for c in cols).rstrip())
    lines.append("  ".join("-" * w[c] for c in cols))
    prev = None
    for _, r in tab.iterrows():
        # Blank line between predictor blocks for readability.
        if prev is not None and r["predictor"] != prev:
            lines.append("")
        lines.append("  ".join(str(r[c]).ljust(w[c]) for c in cols).rstrip())
        prev = r["predictor"]

    num, den = ratio_pair
    lines.append("")
    lines.append(f"Values are per somatic chromosome (n = {n_chrom}); the two CBRs of each")
    lines.append("chromosome were averaged, and the adjacent-gap covariate is the mean of the")
    lines.append("two flanking eliminated blocks. Chromosomes excluded for low window coverage:")
    lines.append(f"  {', '.join(sorted({f.rsplit('.', 1)[0] for f in flagged})) or 'none'}.")
    post = STAGE_LONG.get(num, _pretty_col(num)) if num else "the post-elimination stage"
    ratio_lbl = (f"{_pretty_col(num)} / {_pretty_col(den)}"
                 if num and den else "the raw ratio")
    lines.append("Because E(s) is estimated over the germline coordinate system while")
    lines.append("post-elimination material is physically fragmented, single-stage normalised")
    lines.append("values are not strictly comparable across chromosomes of differing length")
    lines.append(f"at {post}; the raw {ratio_lbl} ratio is the appropriate test for "
                 "post-elimination")
    lines.append("size dependence.")

    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


# ===========================================================================
# CLI
# ===========================================================================
def build_wide(c, responses, stages, ratio_pair, post_elim):
    """
    Transposed layout: one row per predictor, one column per response.

    All twelve tests fit in three rows, which is compact enough to sit as a
    panel beneath the other supplementary panels while keeping every null
    visible. Returns (header_rows, body_rows, dagger_flags).
    """
    num, den = ratio_pair
    ratio_label = f"{num}/{den}" if num and den else None

    col_labels = list(stages) + ([ratio_label] if ratio_label else [])
    body, flags = [], []

    for pred_key, pretty in [("seg_len", "Chromosome length"),
                             ("mean_gap", "Adjacent eliminated block size")]:
        res = spearman_row(c, pred_key, responses)
        cells, dag = [], []
        for r in responses:
            rho, pv, _ = res[r]
            cells.append(_minus(f"{rho:+.3f} ({fmt_p(pv)})")
                         if np.isfinite(rho) else "n.d.")
            dag.append(r in post_elim)
        body.append([pretty, f"Spearman {sym('rho')}"] + cells)
        flags.append([False, False] + dag)

    res = mwu_row(c, responses)
    cells, dag = [], []
    for r in responses:
        mx, ma, pv, _, _ = res[r]
        cells.append(_minus(f"{mx:.3f} vs {ma:.3f} ({fmt_p(pv)})")
                     if np.isfinite(mx) else "n.d.")
        dag.append(r in post_elim)
    body.append(["X-derived vs autosomal", f"Mann{sym('endash')}Whitney U"] + cells)
    flags.append([False, False] + dag)

    header = ["Predictor", "Test"] + [_pretty_col(lbl) for lbl in col_labels]
    return header, body, flags, len(stages)


def _minus(txt):
    """Typographic minus (U+2212) for negative numbers; hyphen in --ascii mode."""
    if ASCII:
        return txt
    return re.sub(r"(?<![\w)])-(?=[.\d])", "\u2212", txt)


def _pretty_col(label, labels=None):
    """
    Column heading for a stage or a stage ratio.

    Uses the developmental-stage names by default so the table matches the
    figures; falls back to a tidied timepoint for any stage not in the map.
    """
    labels = STAGE_LABELS if labels is None else labels
    if "/" in label:
        a, b = label.split("/", 1)
        pa, pb = _pretty_col(a, labels), _pretty_col(b, labels)
        # Compact the ratio heading: "4-8 cells / 2-4 cells" would widen the
        # column far more than the values in it need.
        suffix = " cells"
        if pa.endswith(suffix) and pb.endswith(suffix):
            return f"{pa[:-len(suffix)]} / {pb[:-len(suffix)]}{suffix}"
        return f"{pa} / {pb}"
    if label in labels:
        return labels[label]
    return label.replace("hr", " hr").strip()


def render_table(header, body, flags, n_stage_cols, out_base, n_chrom, n_x,
                 n_auto, flagged, ratio_pair, dpi=300, width=10.0,
                 fontsize=9.0, footnote_wrap=0, justify=True,
                 dagger="\u2020"):
    """
    Draw the table as vector art.

    Rules are horizontal only (booktabs style), which is what most journals
    want and what reads cleanly as a figure panel. Column widths are allocated
    from the widest string in each column so nothing collides.
    """
    ncol = len(header)
    # Character-width heuristic for Arial at the given point size; deterministic,
    # and close enough that columns never overlap.
    # Measure each column against its widest rendered cell. Superscript runs are
    # narrower than their markup, so markup is stripped before measuring.
    raw_w = []
    for j in range(ncol):
        cells = [header[j]] + [r[j] for r in body]
        raw_w.append(max(_text_width_in(markup_to_unicode(c_), fontsize)
                         for c_ in cells) + 0.24)
    total = sum(raw_w)
    scale = (width - 0.3) / total
    col_w = [w * scale for w in raw_w]
    x_left = [0.15 + sum(col_w[:j]) for j in range(ncol)]

    row_h = fontsize * 2.05 / 72.0
    n_rows = len(body)

    # Wrap the footnote to the full width of the table rules rather than to a
    # fixed character count, so the block reaches the right-hand rule instead of
    # stopping short.
    foot_fs = fontsize * 0.86
    avail_in = width - 0.24
    foot_lines = _footnote_lines(n_chrom, n_x, n_auto, flagged, ratio_pair,
                                 dagger, wrap=footnote_wrap,
                                 fontsize=foot_fs, max_in=avail_in)
    foot_h = len(foot_lines) * (foot_fs * 1.45 / 72.0) + 0.12
    height = row_h * (n_rows + 2.4) + foot_h + 0.30

    fig = plt.figure(figsize=(width, height))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.axis("off")

    y = height - 0.18
    rule_x0, rule_x1 = 0.12, width - 0.12

    def rule(yy, x0=None, x1=None, lw=1.1):
        ax.plot([x0 if x0 is not None else rule_x0,
                 x1 if x1 is not None else rule_x1],
                [yy, yy], color="black", lw=lw, solid_capstyle="butt")

    # Spanning header over the normalised-contact columns.
    span0 = x_left[2]
    span1 = x_left[2 + n_stage_cols] - 0.12 if ncol > 2 + n_stage_cols else rule_x1
    rule(y)
    y -= row_h * 0.78
    ax.text((span0 + span1) / 2.0, y, "Normalised contact (O/E)",
            ha="center", va="baseline", fontsize=fontsize, style="italic")
    if ncol > 2 + n_stage_cols:
        ax.text((x_left[-1] + rule_x1) / 2.0, y, "Raw ratio",
                ha="center", va="baseline", fontsize=fontsize, style="italic")
    rule(y - row_h * 0.30, span0, span1, lw=0.7)

    y -= row_h * 0.92
    for j, h in enumerate(header):
        ax.text(x_left[j], y, h, ha="left", va="baseline",
                fontsize=fontsize, fontweight="bold")
    y -= row_h * 0.34
    rule(y)

    for i, row in enumerate(body):
        y -= row_h
        for j, cell in enumerate(row):
            txt = cell + (dagger if flags[i][j] else "")
            draw_rich(ax, x_left[j], y, txt, fontsize)
    y -= row_h * 0.34
    rule(y)

    y -= 0.16
    for line, is_last in foot_lines:
        y -= foot_fs * 1.45 / 72.0
        if justify and not is_last:
            draw_justified(ax, rule_x0, y, line, foot_fs, rule_x1 - rule_x0)
        else:
            ax.text(rule_x0, y, line, ha="left", va="baseline", fontsize=foot_fs)

    for ext in ("svg", "pdf", "png"):
        fig.savefig(f"{out_base}.{ext}", dpi=dpi, bbox_inches="tight",
                    facecolor="white")
    plt.close(fig)


_PROBE = {}


def _text_width_in(txt, fontsize):
    """
    Width of a string in inches at the given point size, measured from the font.

    A single hidden figure is reused for every measurement; text extents need a
    renderer, and creating one per call would be needlessly slow.
    """
    if "fig" not in _PROBE:
        fig = plt.figure(figsize=(2, 1))
        fig.canvas.draw()
        _PROBE["fig"] = fig
        _PROBE["rend"] = fig.canvas.get_renderer()
    fig, rend = _PROBE["fig"], _PROBE["rend"]
    t = fig.text(0, 0, txt, fontsize=fontsize)
    w = t.get_window_extent(renderer=rend).width / fig.dpi
    t.remove()
    return w


def draw_justified(ax, x0, y, line, fontsize, target_w):
    """
    Draw a line with inter-word spaces stretched so it ends exactly at the rule.

    Each word becomes its own text object, which is the cost of justification;
    ``--no_justify`` keeps one object per line with a ragged right edge instead.
    """
    words = line.split()
    if len(words) < 2:
        ax.text(x0, y, line, ha="left", va="baseline", fontsize=fontsize)
        return
    widths = [_text_width_in(w, fontsize) for w in words]
    space = (target_w - sum(widths)) / (len(words) - 1)
    # Never compress below a normal space; overlong lines simply stay ragged.
    space = max(space, _text_width_in(" ", fontsize))
    cx = x0
    for w, wd in zip(words, widths):
        ax.text(cx, y, w, ha="left", va="baseline", fontsize=fontsize)
        cx += wd + space


def draw_rich(ax, x, y, text, fontsize, sup_scale=0.70, sup_rise=0.32):
    """
    Draw text in which ``^{...}`` marks a superscript run.

    Superscripts are drawn as separate raised, reduced-size text using ordinary
    characters rather than Unicode superscript codepoints. Arial and many other
    fonts carry the superscript digits but not U+207B (superscript minus), so a
    Unicode exponent renders as a missing glyph in SVG once the viewer
    substitutes its own font. Drawing the run explicitly is font-independent and
    leaves every character editable.
    """
    cx = x
    for part in re.split(r"(\^\{[^}]*\})", text):
        if not part:
            continue
        if part.startswith("^{"):
            inner = part[2:-1]
            fs = fontsize * sup_scale
            ax.text(cx, y + sup_rise * fontsize / 72.0, inner,
                    ha="left", va="baseline", fontsize=fs)
            cx += _text_width_in(inner, fs)
        else:
            ax.text(cx, y, part, ha="left", va="baseline", fontsize=fontsize)
            cx += _text_width_in(part, fontsize)
    return cx


def _wrap_measured(text, fontsize, max_in):
    """Greedy word wrap to a measured width, so lines reach the right rule."""
    lines, cur = [], []
    for w in text.split():
        if not cur or _text_width_in(" ".join(cur + [w]), fontsize) <= max_in:
            cur.append(w)
        else:
            lines.append(" ".join(cur))
            cur = [w]
    if cur:
        lines.append(" ".join(cur))
    return lines


def _footnote_lines(n_chrom, n_x, n_auto, flagged, ratio_pair, dagger,
                    wrap=0, fontsize=8.0, max_in=9.0):
    """
    Footnote text, wrapped to the measured width of the table rules.

    Character-count wrapping leaves a ragged margin because the count has to be
    conservative; measuring each candidate line lets the block reach the rule.
    """
    num, den = ratio_pair
    post = STAGE_LONG.get(num, _pretty_col(num)) if num else "the final timepoint"
    ratio_lbl = (f"{_pretty_col(num)} / {_pretty_col(den)}"
                 if num and den else "the raw ratio")
    chrom = sorted({f.rsplit(".", 1)[0] for f in flagged})
    parts = [
        f"Spearman {sym('rho')} or median (X-derived vs autosomal), with P in "
        f"parentheses. n = {n_chrom} somatic chromosomes ({n_x} X-derived, "
        f"{n_auto} autosomal); the two CBRs of each chromosome were averaged, and "
        "the adjacent-block covariate is the mean of the two flanking eliminated blocks.",
        f"{dagger} Elimination is complete by {post}, so the "
        "genome is fragmented into separate somatic chromosomes while E(s) is estimated "
        "over germline coordinates; distance-normalised values are not comparable across "
        f"chromosomes of differing length at this stage. The {ratio_lbl} ratio of raw "
        "counts is distance-matched within each chromosome and is the appropriate test.",
    ]
    if chrom:
        # Folded into the first paragraph rather than standing alone: it belongs
        # with the description of which chromosomes were used, and each extra
        # paragraph adds another short, unjustified final line.
        parts[0] += (" Chromosomes excluded for low window coverage: "
                     + ", ".join(chrom) + ".")
    lines = []
    for para in parts:
        wrapped = (textwrap.wrap(para, wrap) if wrap
                   else _wrap_measured(para, fontsize, max_in))
        for i, ln in enumerate(wrapped):
            # Paragraph-final lines are never justified; stretching them is
            # wrong typographically and looks worse than the ragged edge.
            lines.append((ln, i == len(wrapped) - 1))
    return lines


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Tests for systematic variation in cognate CBR "
                    "interaction between somatic chromosomes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--summary", required=True,
                   help="*_pair_summary.txt from cis_cbr_profile.py "
                        "(--partner_mode pairwise_end --pair_summary)")
    p.add_argument("--stages", nargs="+", default=["10hr", "17hr", "36hr"],
                   help="Timepoints to test, in order")
    p.add_argument("--ratio", nargs=2, default=["36hr", "17hr"],
                   metavar=("NUMERATOR", "DENOMINATOR"),
                   help="Raw within-chromosome ratio to test as a fourth "
                        "response; pass '' '' to omit")
    p.add_argument("--x_prefix", default="chrX",
                   help="Somatic chromosome name prefix marking X-derived segments")
    p.add_argument("--timepoint_headers", action="store_true",
                   help="Head the columns with timepoints (10 hr, 17 hr, 36 hr) "
                        "instead of the developmental stages used in the figures")
    p.add_argument("--post_elimination", nargs="*", default=["36hr"],
                   help="Stage label(s) at which elimination is already complete. "
                        "Their normalised columns are marked with a dagger, since "
                        "cross-chromosome comparison is invalid once the genome is "
                        "fragmented.")
    p.add_argument("--no_justify", action="store_true",
                   help="Leave the footnote ragged-right. By default lines are "
                        "justified to the table rules, which splits each line "
                        "into one text object per word.")
    p.add_argument("--footnote_wrap", type=int, default=0,
                   help="Footnote wrap width in characters; 0 measures the font "
                        "and fills the full table width")
    p.add_argument("--no_render", action="store_true",
                   help="Skip the rendered SVG/PDF/PNG panel")
    p.add_argument("--width", type=float, default=10.0,
                   help="Rendered panel width in inches")
    p.add_argument("--fontsize", type=float, default=9.0,
                   help="Rendered panel font size in points")
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--ascii", action="store_true",
                   help="Write plain-ASCII symbols (rho, x10^-6, Mann-Whitney) "
                        "instead of Unicode. Use if the table is going into a "
                        "pipeline that mishandles encodings.")
    p.add_argument("--keep_low_coverage", action="store_true",
                   help="Do NOT exclude coverage-flagged CBRs (diagnostic only)")
    p.add_argument("--output_dir", default=".")
    p.add_argument("--output_prefix", default="table_S_nulls")
    p.add_argument("--debug", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S", stream=sys.stdout,
    )

    global ASCII, STAGE_LABELS
    ASCII = args.ascii
    if args.timepoint_headers:
        STAGE_LABELS = {}

    summary_path = os.path.abspath(args.summary)
    outdir = os.path.abspath(args.output_dir)
    os.makedirs(outdir, exist_ok=True)
    os.chdir(outdir)
    log.info("Working directory: %s", outdir)

    ratio_pair = (args.ratio[0] or None, args.ratio[1] or None)

    c, responses, flagged = load_and_collapse(
        summary_path, args.stages, ratio_pair,
        keep_low_coverage=args.keep_low_coverage, x_prefix=args.x_prefix)

    tab = build_rows(c, responses, args.stages, ratio_pair)

    tsv = f"{args.output_prefix}.tsv"
    txt = f"{args.output_prefix}.txt"
    # utf-8-sig: without the BOM, Excel opens the file as Latin-1 and mangles
    # every non-ASCII symbol. Harmless for pandas, R, and awk.
    tab = tab.copy()
    for col in ("statistic", "effect", "p_value"):
        tab[col] = tab[col].map(markup_to_unicode)
    tab.drop(columns=["p_raw"]).to_csv(
        tsv, sep="\t", index=False,
        encoding="ascii" if args.ascii else "utf-8-sig")
    write_aligned(tab, txt, len(c), flagged, ratio_pair)

    # Echo to the log so the result is visible without opening a file.
    log.info("Null tests table:")
    for line in open(txt, encoding="utf-8").read().splitlines():
        print("    " + line)

    sig = tab.loc[tab["p_raw"] < 0.05, ["predictor", "response", "p_value"]]
    if len(sig):
        log.warning("%d of %d tests reach P < 0.05 (uncorrected); inspect before "
                    "describing the table as uniformly null:\n%s",
                    len(sig), len(tab), sig.to_string(index=False))
    else:
        log.info("All %d tests are non-significant at P < 0.05 (uncorrected).",
                 len(tab))
    # Wide layout: the figure panel, plus a matching TSV that is easy to open.
    header, body, flags, n_stage_cols = build_wide(
        c, responses, args.stages, ratio_pair, set(args.post_elimination))
    wide = pd.DataFrame([[markup_to_unicode(c_) for c_ in row] for row in body],
                        columns=header)
    wide_tsv = f"{args.output_prefix}_wide.tsv"
    wide.to_csv(wide_tsv, sep="\t", index=False,
                encoding="ascii" if args.ascii else "utf-8-sig")

    log.info("Wrote %s, %s and %s", tsv, txt, wide_tsv)

    if not args.no_render:
        render_table(header, body, flags, n_stage_cols, args.output_prefix,
                     len(c), int(c.isX.sum()), int((~c.isX).sum()), flagged,
                     ratio_pair, dpi=args.dpi, width=args.width,
                     fontsize=args.fontsize, footnote_wrap=args.footnote_wrap,
                     justify=not args.no_justify,
                     dagger="*" if ASCII else "\u2020")
        log.info("Wrote %s.{svg,pdf,png} \u2014 place the SVG in Illustrator",
                 args.output_prefix)


if __name__ == "__main__":
    main()
