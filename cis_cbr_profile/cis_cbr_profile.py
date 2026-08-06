#!/usr/bin/env python3
"""
cis_cbr_profile.py
==================
Metaprofiles of Hi-C contacts around chromosomal breakage regions (CBRs) in a
genome whose retained material is tiled along a single germline chromosome.

This is the cis counterpart of ``trans_cbr_profile.py``. That script counts
*trans* (inter-chromosomal) contacts per bin, which works when the germline
karyotype has many chromosomes and CBR-flanking regions can contact other
chromosomes. Where the germline genome is one chromosome, every contact is cis
and the trans definition returns nothing.

The structural equivalent is the *inter-segment* contact. The germline
chromosome is tiled by the retained blocks that become the somatic chromosomes,
separated by interstitial eliminated DNA. A contact is counted for an anchor bin
if its partner end falls in a retained segment that is NOT the anchor's own
(for an anchor inside eliminated DNA, "own" means either flanking retained
segment). This is the exact analogue of "different chromosome", works identically
for pre- and post-elimination libraries as long as both are mapped to the
germline assembly, and yields the per-partner component breakdown for free.

A second mode, ``--partner_mode pairwise_end``, restricts partners to the CBR at
the opposite end of the same retained segment - the two CBRs that will become the
two ends of one somatic chromosome (a *cognate* CBR pair).

Two complications are handled explicitly:

1. Distance decay. Unlike true trans contacts, inter-segment cis contacts decay
   with genomic separation. Two mitigations are provided:
     * ``--min_sep``  drops near-diagonal partners. This MUST stay below the
       smallest retained segment or small segments become invisible from
       adjacent anchors; the script checks this and refuses otherwise.
     * ``--distance_norm oe``  weights every observed contact by 1/E(s), where
       E(s) is the expected contacts per bin-pair at separation s, estimated per
       library from its own cis contacts and its own bin occupancy.

2. Window collisions. CBR spacing is usually uneven, so a fixed flank runs the
   trace from one CBR into the next. Each CBR window is therefore clipped
   independently: the retained side stops at ``--clip_frac`` of the way to the
   cognate CBR, and the eliminated side stops at ``--clip_frac`` of the way to
   the next CBR across the eliminated block. Clipped positions are NaN, not zero,
   so they are excluded from the metaprofile mean. The number of contributing
   CBRs is tracked per bin and written to the output table; ``--min_cbr_n`` blanks
   bins supported by too few CBRs.

Orientation and CBR definition
------------------------------
CBRs are derived directly from the retained-segment BED: every retained block
contributes two CBRs, at its left and right boundaries, so N segments give 2N
CBRs. The left boundary has eliminated DNA on the left (orientation ER) and is
flipped; the right boundary is RE and is not. After flipping, retained DNA is
always on the negative (left) axis and eliminated DNA on the positive (right)
axis.

Inputs
------
--segments     BED of retained blocks: chrom, start, end, name. One row per
               prospective somatic chromosome, on the germline chromosome.
--chrom_sizes  Two-column chrom.sizes for the germline assembly.
--samples      HiC-Pro ``allValidPairs`` files. Columns 2,3,5,6 (1-based) are
               read as chr1, pos1, chr2, pos2; other columns are ignored.

Outputs
-------
  <output_dir>/
    <output_table>                       per-bin metaprofile values + CBR counts
    <output_plot>                        metaprofile, all CBRs
    <prefix>_<group>.{png,svg,pdf}       metaprofile per split group
    <prefix>_pair_summary.txt            per-CBR summary (--pair_summary)
    components/                          stacked-area partner-segment breakdown
    individual_cbrs/                     per-CBR traces + data tables
    <grid_plot>                          grid overview of all CBRs
    segment_geometry.txt                 derived CBR table (position, orientation,
                                         clip limits, group assignment)

Dependencies
------------
numpy, pandas, matplotlib

Example
-------
The published analysis of *Parascaris univalens*, whose germline genome is a
single chromosome (chrX) tiled by 36 retained segments giving 72 CBRs:

  python cis_cbr_profile.py \\
      --segments germ_to_soma_mapping.bed \\
      --chrom_sizes genome.chrom.sizes \\
      --chrom chrX \\
      --samples sample_10hr.allValidPairs sample_17hr.allValidPairs \\
                sample_36hr.allValidPairs \\
      --labels 10hr 17hr 36hr \\
      --binsize 5000 --flank_retained 500000 --flank_eliminated 100000 \\
      --min_sep 1000000 --distance_norm oe --downsample auto \\
      --split_by gap --gap_threshold 100000 --min_cbr_n 10 \\
      --threads 3 \\
      --output_dir cbr_profile_output

Hi-C matrices for the published analysis are available from GEO under accession
GSE315650 (Parascaris) and GSE314626 (Ascaris).
"""

import argparse
import logging
import warnings
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Figure standards (Arial, no top/right spines, Illustrator-editable SVG text)
# ---------------------------------------------------------------------------
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
matplotlib.rcParams["svg.fonttype"] = "none"
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["axes.spines.top"] = False
matplotlib.rcParams["axes.spines.right"] = False
matplotlib.rcParams["axes.labelsize"] = 20
matplotlib.rcParams["axes.titlesize"] = 20
matplotlib.rcParams["xtick.labelsize"] = 16
matplotlib.rcParams["ytick.labelsize"] = 16
matplotlib.rcParams["legend.fontsize"] = 14

RETAINED_COLOR = "#3B6FB6"   # blue, retained / left
ELIMINATED_COLOR = "#C8102E"  # red, eliminated / right

log = logging.getLogger("cis_cbr")


# ===========================================================================
# Geometry: segments, CBRs, per-bin lookup arrays
# ===========================================================================
def load_chrom_size(chrom_sizes_file, chrom):
    """Return the length of `chrom` from a two-column chrom.sizes file."""
    with open(chrom_sizes_file) as fh:
        for line in fh:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.split()
            if parts[0] == chrom:
                return int(parts[1])
    raise ValueError(f"{chrom} not found in {chrom_sizes_file}")


def load_segments(bed_file, chrom):
    """
    Load retained segments (future somatic chromosomes).

    BED columns: chrom  start  end  [name]
    Returns a DataFrame sorted by start with columns start, end, name.
    """
    rows = []
    with open(bed_file) as fh:
        for line in fh:
            if not line.strip() or line.startswith(("#", "track")):
                continue
            parts = line.split()
            if len(parts) < 3 or parts[0] != chrom:
                continue
            start, end = int(parts[1]), int(parts[2])
            name = parts[3] if len(parts) >= 4 else f"seg{len(rows) + 1:02d}"
            rows.append((start, end, name))
    if not rows:
        raise ValueError(f"No segments on {chrom} in {bed_file}")
    seg = pd.DataFrame(rows, columns=["start", "end", "name"]).sort_values("start")
    seg = seg.reset_index(drop=True)
    # Segments must not overlap; overlapping segments break the partner logic.
    if (seg["start"].values[1:] < seg["end"].values[:-1]).any():
        raise ValueError("Retained segments overlap; check the segment BED.")
    return seg


def build_bin_lookups(seg, nbins, binsize):
    """
    Per-bin arrays used in the hot loop.

    seg_of_bin  : retained segment index for the bin, or -1 if eliminated
    own_left    : segment index immediately to the left  (-1 if none)
    own_right   : segment index immediately to the right (-1 if none)

    For a bin inside a retained segment own_left == own_right == that segment.
    For a bin in eliminated DNA they are the two flanking segments, and a partner
    landing in either of them is NOT counted as inter-segment.
    """
    centers = (np.arange(nbins, dtype=np.float64) + 0.5) * binsize
    starts = seg["start"].values.astype(np.float64)
    ends = seg["end"].values.astype(np.float64)

    prev_idx = np.searchsorted(starts, centers, side="right") - 1  # last seg starting before center
    inside = np.zeros(nbins, dtype=bool)
    ok = prev_idx >= 0
    inside[ok] = centers[ok] < ends[prev_idx[ok]]

    seg_of_bin = np.where(inside, prev_idx, -1).astype(np.int32)
    own_left = np.where(inside, prev_idx, prev_idx).astype(np.int32)          # -1 before first seg
    own_right = np.where(inside, prev_idx, prev_idx + 1).astype(np.int32)
    own_right[own_right >= len(seg)] = -1
    return seg_of_bin, own_left, own_right


def build_cbrs(seg, chrom_len, flank_ret, flank_elim, clip_frac):
    """
    Derive the 72 CBRs from the 36 retained segments and clip each window so it
    cannot reach the neighbouring CBR.

    Returns a DataFrame, one row per CBR:
      name, somatic, side (L/R), pos, orient (ER/RE),
      seg_idx, seg_len, gap (adjacent eliminated block size),
      lim_ret, lim_elim (usable flank in bp on each side, after clipping)
    """
    starts = seg["start"].values
    ends = seg["end"].values
    names = seg["name"].values
    n = len(seg)

    rows = []
    for i in range(n):
        seg_len = ends[i] - starts[i]
        # left CBR of this segment: eliminated on the left -> ER
        gap_left = starts[i] - (ends[i - 1] if i > 0 else 0)
        # right CBR of this segment: eliminated on the right -> RE
        gap_right = (starts[i + 1] if i < n - 1 else chrom_len) - ends[i]

        for side, pos, orient, gap in (
            ("L", starts[i], "ER", gap_left),
            ("R", ends[i], "RE", gap_right),
        ):
            # retained side: stop before reaching the paired CBR of the same segment
            lim_ret = min(flank_ret, int(seg_len * clip_frac))
            # eliminated side: stop before reaching the next CBR across the gap
            lim_elim = min(flank_elim, int(gap * clip_frac))
            rows.append(
                dict(
                    name=f"{names[i]}.{side}",
                    somatic=names[i],
                    side=side,
                    pos=int(pos),
                    orient=orient,
                    seg_idx=i,
                    seg_len=int(seg_len),
                    gap=int(gap),
                    lim_ret=int(lim_ret),
                    lim_elim=int(lim_elim),
                )
            )
    cbr = pd.DataFrame(rows).sort_values("pos").reset_index(drop=True)
    return cbr


def assign_groups(cbr, split_by, gap_threshold, size_threshold):
    """Add a `group` column used to split the metaprofile into two panels."""
    if split_by == "none":
        cbr["group"] = "all"
    elif split_by == "gap":
        cbr["group"] = np.where(
            cbr["gap"] < gap_threshold,
            f"narrow_gap_lt{int(gap_threshold / 1000)}kb",
            f"wide_gap_ge{int(gap_threshold / 1000)}kb",
        )
    elif split_by == "size":
        cbr["group"] = np.where(
            cbr["seg_len"] < size_threshold,
            f"small_chrom_lt{int(size_threshold / 1e6)}Mb",
            f"large_chrom_ge{int(size_threshold / 1e6)}Mb",
        )
    elif split_by == "side":
        cbr["group"] = np.where(cbr["side"] == "L", "left_end", "right_end")
    else:
        raise ValueError(f"unknown --split_by {split_by}")
    return cbr


def build_window_index(cbr, nbins, binsize, flank_ret, flank_elim):
    """
    Map genomic bins onto (CBR, relative-bin) slots.

    Relative axis runs from -flank_ret to +flank_elim after orientation flipping,
    so retained DNA is always negative and eliminated DNA always positive.

    Returns
    -------
    win_cbr  : int32[nbins], CBR index for the bin or -1
    win_rel  : int32[nbins], relative-bin index for the bin or -1
    rel_centers : float[n_rel], relative bin centres in bp
    valid    : bool[n_cbr, n_rel], geometry mask (False where clipped away)
    """
    rel_edges = np.arange(-flank_ret, flank_elim + binsize, binsize)
    rel_centers = (rel_edges[:-1] + rel_edges[1:]) / 2.0
    n_rel = len(rel_centers)
    n_cbr = len(cbr)

    win_cbr = np.full(nbins, -1, dtype=np.int32)
    win_rel = np.full(nbins, -1, dtype=np.int32)
    best_abs = np.full(nbins, np.inf, dtype=np.float64)

    contested = 0
    for ci, row in cbr.iterrows():
        pos = row["pos"]
        # d = +1 when eliminated DNA lies to the right of the CBR (RE, no flip)
        d = 1 if row["orient"].upper() == "RE" else -1
        lo = pos - d * row["lim_ret"]      # far end of the retained side
        hi = pos + d * row["lim_elim"]     # far end of the eliminated side
        b0, b1 = sorted((lo, hi))
        bin_lo = max(0, int(np.floor(b0 / binsize)))
        bin_hi = min(nbins - 1, int(np.floor((b1 - 1) / binsize)))
        if bin_hi < bin_lo:
            continue
        bins = np.arange(bin_lo, bin_hi + 1)
        centers = (bins + 0.5) * binsize
        rel = (centers - pos) * d
        rel_idx = np.digitize(rel, rel_edges) - 1
        keep = (rel_idx >= 0) & (rel_idx < n_rel)
        bins, rel_idx, rel = bins[keep], rel_idx[keep], rel[keep]
        if bins.size == 0:
            continue
        # Windows of two CBRs facing each other across a short eliminated block
        # meet at the gap midpoint and can contest the boundary bin.  The bin is
        # given to the nearer CBR, and the loser simply lacks that position.
        a = np.abs(rel)
        contested += int(((win_cbr[bins] != -1) & (a < best_abs[bins])).sum())
        take = a < best_abs[bins]
        win_cbr[bins[take]] = ci
        win_rel[bins[take]] = rel_idx[take]
        best_abs[bins[take]] = a[take]

    if contested:
        log.info("%d boundary bins contested between adjacent CBRs; "
                 "assigned to the nearer CBR", contested)

    # Derive the geometry mask from the final assignment so that a CBR which lost
    # a contested bin is correctly marked as missing that position.
    valid = np.zeros((n_cbr, n_rel), dtype=bool)
    has = win_cbr >= 0
    valid[win_cbr[has], win_rel[has]] = True
    return win_cbr, win_rel, rel_centers, valid


# ===========================================================================
# Distance-decay expectation
# ===========================================================================
def _autocorr(v, k_max):
    """Autocorrelation of a 1-D vector at lags 0..k_max, via FFT."""
    n = len(v)
    if n == 0 or k_max < 0:
        return np.zeros(max(k_max + 1, 0), dtype=np.float64)
    size = 1
    while size < 2 * n:
        size *= 2
    f = np.fft.rfft(v.astype(np.float64), size)
    ac = np.fft.irfft(f * np.conj(f), size)[: k_max + 1]
    return np.maximum(ac, 0.0)


def occupancy_pair_counts(occ, k_max):
    """
    Number of occupied bin pairs at every separation 1..k_max across the whole
    chromosome.  Needed so that the expected contact frequency E(s) is per
    *usable* bin pair rather than per nominal bin pair (the assembly is heavily
    masked).
    """
    return _autocorr(occ, k_max)


def occupancy_pair_counts_within(occ, seg_of_bin, n_seg, k_max):
    """
    Number of occupied bin pairs at every separation 1..k_max that fall inside
    the SAME retained segment, summed over segments.

    This is the denominator for a within-somatic-chromosome distance decay.  It
    matters post-PDE: once the eliminated DNA is gone the germline chromosome is
    physically 36 separate somatic chromosomes, and a whole-chromosome
    denominator counts bin pairs that no longer coexist on one molecule.  At
    large separations almost none of those pairs are within a segment, so the
    whole-chromosome E(s) is too small there and O/E is inflated for exactly the
    largest segments -- which manufactures an apparent size dependence.

    Segments occupy contiguous bin ranges, so each is handled as one slice.
    """
    total = np.zeros(k_max + 1, dtype=np.float64)
    for si in range(n_seg):
        idx = np.flatnonzero(seg_of_bin == si)
        if idx.size == 0:
            continue
        sub = occ[idx[0]: idx[-1] + 1]
        ac = _autocorr(sub, min(k_max, len(sub) - 1))
        total[: len(ac)] += ac
    return total


def make_distance_bins(k_max, n_dbins):
    """Log-spaced separation bins (in units of genomic bins), starting at k=1."""
    edges = np.unique(
        np.round(np.geomspace(1, max(k_max, 2), num=n_dbins + 1)).astype(np.int64)
    )
    edges[-1] = max(edges[-1], k_max + 1)
    return edges


def expected_per_dbin(dist_hist, npairs_k, dbin_edges, min_k=1):
    """
    E[d] = observed contacts in separation bin d / usable bin pairs in d,
    rescaled so that the median expectation over the separations actually used
    (k >= min_k) equals 1.

    The rescaling keeps the weights 1/E near unity, so an O/E-weighted profile
    stays on the same numeric scale as a raw contact count and can be read the
    same way as the Ascaris figures.

    dist_hist : int64[k_max+1] observed contacts at each separation k
    npairs_k  : float[k_max+1] usable bin pairs at each separation k, from
                occupancy_pair_counts (whole chromosome) or
                occupancy_pair_counts_within (same segment only).  Numerator and
                denominator must share the same scope.
    """
    obs = np.add.reduceat(dist_hist[1:], np.maximum(dbin_edges[:-1] - 1, 0))
    npair = np.add.reduceat(npairs_k[1:], np.maximum(dbin_edges[:-1] - 1, 0))
    with np.errstate(divide="ignore", invalid="ignore"):
        E = np.where(npair > 0, obs / npair, np.nan)
    # Fill empty high-separation bins by forward-filling the last finite value so
    # that no contact ever receives an infinite weight.
    finite = np.isfinite(E) & (E > 0)
    if not finite.any():
        raise RuntimeError("Distance-decay expectation is empty; check input.")
    idx = np.where(finite, np.arange(len(E)), -1)
    idx = np.maximum.accumulate(idx)
    idx[idx < 0] = np.argmax(finite)
    E = E[idx]

    used = dbin_edges[:-1] >= min_k
    ref = np.median(E[used]) if used.any() else np.median(E)
    if not np.isfinite(ref) or ref <= 0:
        ref = np.median(E)
    return E / ref


# ===========================================================================
# allValidPairs scanning
# ===========================================================================
def _chunks(path, chunksize):
    """Yield (chr1, pos1, chr2, pos2) chunks from a HiC-Pro allValidPairs file."""
    reader = pd.read_csv(
        path,
        sep="\t",
        header=None,
        usecols=[1, 2, 4, 5],
        names=["c1", "p1", "c2", "p2"],
        dtype={"c1": str, "p1": np.int64, "c2": str, "p2": np.int64},
        chunksize=chunksize,
        on_bad_lines="skip",
        engine="c",
    )
    for chunk in reader:
        yield chunk


def scan_pass1(args_tuple):
    """
    First pass: total cis contacts, separation histogram, and bin occupancy.
    Everything needed to build the distance-decay expectation and to decide the
    downsampling depth.
    """
    (path, label, chrom, binsize, nbins, chunksize,
     seg_of_bin, expected_scope,
     win_cbr, pair_lo_bin, pair_hi_bin, exclude_signal) = args_tuple
    dist_hist = np.zeros(nbins, dtype=np.int64)
    occ_counts = np.zeros(nbins, dtype=np.int64)
    total = 0
    within = expected_scope == "within_segment"

    for chunk in _chunks(path, chunksize):
        m = (chunk["c1"].values == chrom) & (chunk["c2"].values == chrom)
        if not m.any():
            continue
        p1 = chunk["p1"].values[m]
        p2 = chunk["p2"].values[m]
        b1 = np.clip(p1 // binsize, 0, nbins - 1)
        b2 = np.clip(p2 // binsize, 0, nbins - 1)
        k = np.abs(b1 - b2)
        if exclude_signal:
            # Drop the paired-end contacts themselves from E(s).  They are
            # within-segment contacts at separation ~= seg_len, and at that
            # separation only one or two segments contribute usable bin pairs,
            # so leaving them in lets the signal inflate its own expectation and
            # normalise itself away.
            sig = np.zeros(b1.size, dtype=bool)
            for anchor, partner in ((b1, b2), (b2, b1)):
                ci = win_cbr[anchor]
                has = ci >= 0
                if not has.any():
                    continue
                hit = np.zeros(b1.size, dtype=bool)
                hit[has] = ((partner[has] >= pair_lo_bin[ci[has]]) &
                            (partner[has] <= pair_hi_bin[ci[has]]))
                sig |= hit
            if sig.any():
                keep = ~sig
                b1, b2, k = b1[keep], b2[keep], k[keep]
                if b1.size == 0:
                    continue
        if within:
            # Restrict the observed side of E(s) to same-segment pairs so that
            # numerator and denominator share scope.
            s1 = seg_of_bin[b1]
            same = (s1 >= 0) & (s1 == seg_of_bin[b2])
            if same.any():
                dist_hist += np.bincount(k[same], minlength=nbins)[:nbins]
        else:
            dist_hist += np.bincount(k, minlength=nbins)[:nbins]
        occ_counts += np.bincount(b1, minlength=nbins)[:nbins]
        occ_counts += np.bincount(b2, minlength=nbins)[:nbins]
        total += int(m.sum())

    log.info("[%s] pass 1: %d cis %s contacts", label, total, chrom)
    return label, dict(dist_hist=dist_hist,
                       occ=(occ_counts > 0).astype(np.float64),
                       cov=occ_counts.astype(np.float64),
                       total=total)


def scan_pass2(args_tuple):
    """
    Second pass: accumulate contacts into CBR windows.

    Two partner modes:
      * inter_segment (default) -- partner must land in a retained segment that is
        not the anchor's own segment(s).  This is the Ascaris-style "different
        chromosome" analogue.
      * pairwise_end -- partner must land within +/- pair_window of the CBR at the
        OTHER end of the anchor's own somatic chromosome (its cognate CBR).  This
        renders the cognate CBR pair interaction as a break-centered profile.

    Returns raw and O/E-weighted profiles [n_cbr, n_rel] plus the partner-segment
    component breakdown [n_cbr, n_rel, n_seg] (component is meaningful only for
    inter_segment; in pairwise_end every partner is the anchor's own segment).
    """
    (path, label, chrom, binsize, nbins, chunksize, min_sep,
     win_cbr, win_rel, seg_of_bin, own_left, own_right,
     n_cbr, n_rel, n_seg, dbin_edges, E, p_keep, seed,
     partner_mode, pair_lo_bin, pair_hi_bin) = args_tuple

    prof_raw = np.zeros(n_cbr * n_rel, dtype=np.float64)
    prof_oe = np.zeros(n_cbr * n_rel, dtype=np.float64)
    comp_oe = np.zeros(n_cbr * n_rel * n_seg, dtype=np.float64)
    rng = np.random.default_rng(seed)
    kept_total = 0

    # np.digitize on separation -> index into the log-spaced distance bins
    n_dbin = len(dbin_edges) - 1

    for chunk in _chunks(path, chunksize):
        m = (chunk["c1"].values == chrom) & (chunk["c2"].values == chrom)
        if not m.any():
            continue
        p1 = chunk["p1"].values[m]
        p2 = chunk["p2"].values[m]

        if p_keep < 1.0:
            sel = rng.random(p1.size) < p_keep
            p1, p2 = p1[sel], p2[sel]
            if p1.size == 0:
                continue
        kept_total += p1.size

        b1 = np.clip(p1 // binsize, 0, nbins - 1)
        b2 = np.clip(p2 // binsize, 0, nbins - 1)
        sep = np.abs(p1.astype(np.int64) - p2.astype(np.int64))
        k = np.abs(b1 - b2)

        far = (sep >= min_sep) & (k >= 1)
        if not far.any():
            continue
        b1, b2, k = b1[far], b2[far], k[far]

        dbin = np.clip(np.digitize(k, dbin_edges) - 1, 0, n_dbin - 1)
        w = 1.0 / E[dbin]

        # Both orientations: each end can be the anchor, the other the partner.
        for anchor, partner in ((b1, b2), (b2, b1)):
            has_win = win_cbr[anchor] >= 0
            if not has_win.any():
                continue
            a = anchor[has_win]
            q = partner[has_win]
            ws = w[has_win]
            ci = win_cbr[a]

            if partner_mode == "pairwise_end":
                # Partner must fall in the paired-end window of THIS CBR.
                good = (q >= pair_lo_bin[ci]) & (q <= pair_hi_bin[ci])
                if not good.any():
                    continue
                a, ci, ws = a[good], ci[good], ws[good]
                ps = seg_of_bin[q[good]]
                ps = np.where(ps >= 0, ps, 0)  # partner is in-segment by design
            else:
                ps = seg_of_bin[q]
                good = (ps >= 0) & (ps != own_left[a]) & (ps != own_right[a])
                if not good.any():
                    continue
                a, ci, ps, ws = a[good], ci[good], ps[good], ws[good]

            flat = ci.astype(np.int64) * n_rel + win_rel[a].astype(np.int64)
            prof_raw += np.bincount(flat, minlength=n_cbr * n_rel)
            prof_oe += np.bincount(flat, weights=ws, minlength=n_cbr * n_rel)
            comp_oe += np.bincount(
                flat * n_seg + ps.astype(np.int64),
                weights=ws,
                minlength=n_cbr * n_rel * n_seg,
            )

    log.info("[%s] pass 2 done (%d cis contacts used)", label, kept_total)
    return label, dict(
        prof_raw=prof_raw.reshape(n_cbr, n_rel),
        prof_oe=prof_oe.reshape(n_cbr, n_rel),
        comp_oe=comp_oe.reshape(n_cbr, n_rel, n_seg),
        kept_total=kept_total,
    )


# ===========================================================================
# Metaprofiles
# ===========================================================================
def metaprofile(profile, valid, rows):
    """
    Mean +/- SEM across the selected CBRs, ignoring clipped (invalid) positions.

    profile : float[n_cbr, n_rel]
    valid   : bool[n_cbr, n_rel]
    rows    : integer index of CBRs to include
    Returns (mean, sem, n_per_bin)
    """
    arr = profile[rows].astype(np.float64).copy()
    msk = valid[rows]
    arr[~msk] = np.nan
    n = np.sum(msk, axis=0)
    # Positions where every CBR was clipped away are all-NaN columns; the empty
    # -slice warnings they raise are expected and are silenced here.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(arr, axis=0)
        sd = np.nanstd(arr, axis=0)
    sem = np.where(n > 0, sd / np.sqrt(np.maximum(n, 1)), np.nan)
    return mean, sem, n


# ===========================================================================
# Plotting
# ===========================================================================
def save_all_formats(fig, base_path, dpi):
    for ext in ("png", "svg", "pdf"):
        fig.savefig(f"{base_path}.{ext}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _decorate(ax, rel_centers, ylabel, title, binsize):
    ax.axvline(0, color="black", linestyle="--", linewidth=1.2)
    ax.axvspan(-binsize / 1000.0, binsize / 1000.0, color="#2CA02C", alpha=0.25, lw=0)
    lo = rel_centers.min() / 1000.0
    hi = rel_centers.max() / 1000.0
    ax.axvspan(lo, 0, color=RETAINED_COLOR, alpha=0.05, lw=0)
    ax.axvspan(0, hi, color=ELIMINATED_COLOR, alpha=0.05, lw=0)
    ax.set_xlim(lo, hi)
    ax.set_xlabel("Relative Position (kb)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)


def plot_metaprofile(rel_centers, means, sems, ns, labels, ylabel, title,
                     base_path, dpi, binsize, ylim=None):
    """Metaprofile with mean line and +/- 1 SEM shading, one line per sample."""
    fig, ax = plt.subplots(figsize=(10, 6.5))
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(labels)))
    x = rel_centers / 1000.0
    for c, lab in zip(colors, labels):
        m, s = means[lab], sems[lab]
        ax.plot(x, m, color=c, lw=1.6, label=lab)
        ax.fill_between(x, m - s, m + s, color=c, alpha=0.22, lw=0)
    _decorate(ax, rel_centers, ylabel, title, binsize)
    if ylim is not None:
        ax.set_ylim(0, ylim)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True)
    save_all_formats(fig, base_path, dpi)


def plot_cbr_count(rel_centers, ns, base_path, dpi, binsize):
    """Companion panel: how many CBRs contribute at each relative position."""
    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.fill_between(rel_centers / 1000.0, 0, ns, color="grey", alpha=0.6, lw=0)
    _decorate(ax, rel_centers, "CBRs contributing", "", binsize)
    save_all_formats(fig, base_path, dpi)


def plot_components(rel_centers, comp, seg_names, ylabel, title, base_path,
                    dpi, binsize, top_n=0):
    """
    Stacked area of per-partner-somatic-chromosome contributions.

    Segments are stacked and coloured in genomic order along chrX (matching the
    ordered, colour-coded chromosome breakdown used for Ascaris).  With top_n > 0
    only the strongest top_n partners are shown individually, still in genomic
    order, and the remainder are pooled as "other".
    """
    totals = comp.sum(axis=0)
    if top_n and top_n < comp.shape[1]:
        keep = np.sort(np.argsort(-totals)[:top_n])
        rest = np.setdiff1d(np.arange(comp.shape[1]), keep)
        mat = np.column_stack([comp[:, keep], comp[:, rest].sum(axis=1)])
        names = [seg_names[i] for i in keep] + ["other"]
    else:
        mat = comp
        names = list(seg_names)

    fig, ax = plt.subplots(figsize=(11, 6.5))
    colors = plt.cm.viridis(np.linspace(0, 0.95, mat.shape[1]))
    ax.stackplot(rel_centers / 1000.0, mat.T, colors=colors, labels=names, lw=0)
    _decorate(ax, rel_centers, ylabel, title, binsize)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), ncol=2,
              frameon=False, fontsize=9)
    save_all_formats(fig, base_path, dpi)


def plot_grid(rel_centers, profiles, valid, cbr, labels, ylabel, base_path,
              dpi, binsize, ncol=12):
    """Grid overview: one small panel per CBR."""
    n = len(cbr)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.4 * ncol, 2.1 * nrow))
    axes = np.atleast_2d(axes)
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(labels)))
    x = rel_centers / 1000.0
    for i in range(nrow * ncol):
        ax = axes.flat[i]
        if i >= n:
            ax.axis("off")
            continue
        for c, lab in zip(colors, labels):
            y = profiles[lab][i].astype(float).copy()
            y[~valid[i]] = np.nan
            ax.plot(x, y, color=c, lw=0.8)
        ax.axvline(0, color="black", ls=":", lw=0.7)
        ax.set_title(cbr["name"].iloc[i], fontsize=8)
        ax.tick_params(labelsize=6)
        if i % ncol == 0:
            ax.set_ylabel(ylabel, fontsize=7)
        if i >= n - ncol:
            ax.set_xlabel("kb", fontsize=7)
    fig.tight_layout()
    save_all_formats(fig, base_path, dpi)


def plot_single_cbr(rel_centers, profiles, valid_row, labels, name, ylabel,
                    base_path, dpi, binsize):
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(labels)))
    x = rel_centers / 1000.0
    for c, lab in zip(colors, labels):
        y = profiles[lab].astype(float).copy()
        y[~valid_row] = np.nan
        ax.plot(x, y, color=c, lw=1.4, label=lab)
    _decorate(ax, rel_centers, ylabel, name, binsize)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    save_all_formats(fig, base_path, dpi)


# ===========================================================================
# CLI
# ===========================================================================
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Cis Hi-C contact metaprofiles around chromosomal "
                    "breakage regions on a single germline chromosome.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # inputs
    p.add_argument("--segments", required=True,
                   help="Retained-segment BED (chrom start end [name]); "
                        "germ_to_soma_mapping.bed")
    p.add_argument("--chrom_sizes", required=True, help="chrom.sizes for the germline assembly")
    p.add_argument("--chrom", default="chrX",
                   help="Germline chromosome to analyse. Default chrX is the "
                        "single Parascaris germline chromosome; set this to match "
                        "the chromosome named in --segments and --chrom_sizes.")
    p.add_argument("--samples", required=True, nargs="+",
                   help="HiC-Pro allValidPairs files (cis contacts required)")
    p.add_argument("--labels", nargs="+", default=None,
                   help="Legend labels, one per sample (default: file basenames)")
    # binning / geometry
    p.add_argument("--binsize", type=int, default=5000, help="Bin size (bp)")
    p.add_argument("--flank_retained", type=int, default=500000,
                   help="Retained-side flank (bp), plotted on the negative axis")
    p.add_argument("--flank_eliminated", type=int, default=100000,
                   help="Eliminated-side flank (bp), plotted on the positive axis")
    p.add_argument("--clip_frac", type=float, default=0.5,
                   help="Fraction of the distance to the neighbouring CBR that a "
                        "window may occupy (0.5 = stop at the midpoint)")
    p.add_argument("--min_cbr_n", type=int, default=10,
                   help="Blank metaprofile bins supported by fewer CBRs than this")
    # partner definition
    p.add_argument("--partner_mode",
                   choices=["inter_segment", "pairwise_end"], default="inter_segment",
                   help="'inter_segment' = Ascaris-style, partner in any other "
                        "somatic segment. 'pairwise_end' restricts partners to the "
                        "partner within +/- pair_window of the CBR at the other "
                        "end of the anchor's own somatic chromosome.")
    p.add_argument("--pair_window", type=int, default=100000,
                   help="Half-width (bp) of the paired-end partner window "
                        "(pairwise_end mode only)")
    p.add_argument("--pair_summary", action="store_true",
                   help="pairwise_end only: write one paired-end O/E scalar per "
                        "CBR per stage (mean over the retained side approaching "
                        "the break) plus a per-region dot plot, for a Figure 6E "
                        "style summary.")
    p.add_argument("--min_window_cov_frac", type=float, default=0.2,
                   help="Flag a CBR as low_coverage if its anchor or partner "
                        "window has marginal coverage below this fraction of the "
                        "median window in any stage. Paired-end contacts need "
                        "coverage at both ends, so one unmappable window forces "
                        "the value to ~0 for technical reasons.")
    p.add_argument("--summary_window", type=int, default=None,
                   help="Retained-side integration window (bp) for --pair_summary "
                        "as [-summary_window, 0). Default: --pair_window.")
    # distance handling
    p.add_argument("--min_sep", type=int, default=1000000,
                   help="Minimum genomic separation (bp) for a contact to count. "
                        "inter_segment: must be below the smallest somatic "
                        "chromosome (2.80 Mb). pairwise_end: must be below "
                        "smallest segment minus pair_window.")
    p.add_argument("--distance_norm", choices=["none", "oe"], default="oe",
                   help="'oe' weights each contact by 1/E(separation)")
    p.add_argument("--n_dbins", type=int, default=100,
                   help="Number of log-spaced separation bins for E(s)")
    p.add_argument("--exclude_signal_from_expected", action="store_true",
                   help="Remove the paired-end contacts from the E(s) estimate "
                        "(pairwise_end mode). DIAGNOSTIC ONLY -- post-PDE the "
                        "only contacts at separation ~= seg_len are the signal "
                        "itself, so removing them leaves E(s) undetermined there "
                        "and inflates O/E for the largest segments. Off by "
                        "default.")
    p.add_argument("--expected_scope",
                   choices=["chromosome", "within_segment"], default="chromosome",
                   help="Scope of the distance-decay expectation E(s). "
                        "'chromosome' uses all bin pairs on the germline "
                        "chromosome. 'within_segment' uses only bin pairs inside "
                        "the same retained segment, which is the correct "
                        "denominator once elimination has separated the segments "
                        "into distinct somatic chromosomes; use it whenever "
                        "comparing O/E across segments of different size.")
    # depth handling
    p.add_argument("--downsample", choices=["none", "auto"], default="auto",
                   help="'auto' subsamples every library to the smallest cis depth "
                        "before counting (CPM alone does not fix sparsity)")
    p.add_argument("--normalize", action="store_true",
                   help="Scale profiles to contacts per million cis contacts")
    p.add_argument("--scale", type=float, default=1e6, help="CPM scale factor")
    p.add_argument("--seed", type=int, default=1, help="RNG seed for downsampling")
    # grouping
    p.add_argument("--split_by", choices=["none", "gap", "size", "side"], default="gap",
                   help="Second-panel split. 'gap' = size of the adjacent "
                        "eliminated block; 'size' = own somatic chromosome length; "
                        "'side' = left vs right end of the somatic chromosome")
    p.add_argument("--gap_threshold", type=int, default=100000)
    p.add_argument("--size_threshold", type=int, default=6000000)
    # outputs
    p.add_argument("--component_sample", default=None,
                   help="Label of the sample used for the component breakdown "
                        "(default: first sample)")
    p.add_argument("--component_top_n", type=int, default=0,
                   help="0 = show all partner segments in genomic order; "
                        "N > 0 = show the N strongest and pool the rest as 'other'")
    p.add_argument("--individual", action="store_true",
                   help="Write per-CBR plots and data tables")
    p.add_argument("--output_dir", default="cbr_profile_output",
                   help="Directory for all outputs; created if absent")
    p.add_argument("--output_table", default="cbr_profile.txt")
    p.add_argument("--output_plot", default="cbr_profile.png")
    p.add_argument("--grid_plot", default="cbrs_grid.png")
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--ylim", type=float, default=None)
    p.add_argument("--chunksize", type=int, default=2000000,
                   help="Rows per pandas read_csv chunk")
    p.add_argument("--threads", type=int, default=1,
                   help="Parallel workers; capped at the number of samples")
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
    for noisy in ("matplotlib", "matplotlib.font_manager", "fontTools",
                  "fontTools.subset", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    # Resolve every input path before changing directory, so that relative paths
    # on the command line keep working.
    args.segments = os.path.abspath(args.segments)
    args.chrom_sizes = os.path.abspath(args.chrom_sizes)
    args.samples = [os.path.abspath(s) for s in args.samples]

    outdir = os.path.abspath(args.output_dir)
    os.makedirs(outdir, exist_ok=True)
    os.chdir(outdir)
    log.info("Working directory: %s", outdir)

    labels = args.labels if args.labels else [
        os.path.basename(s).split(".allValidPairs")[0] for s in args.samples
    ]
    if len(labels) != len(args.samples):
        raise ValueError("--labels must have one entry per --samples")
    for s in args.samples:
        if not os.path.exists(s):
            raise FileNotFoundError(s)

    # ---- geometry -------------------------------------------------------
    chrom_len = load_chrom_size(args.chrom_sizes, args.chrom)
    nbins = int(np.ceil(chrom_len / args.binsize))
    seg = load_segments(args.segments, args.chrom)
    n_seg = len(seg)
    log.info("%s: %d bp, %d bins of %d bp, %d retained segments",
             args.chrom, chrom_len, nbins, args.binsize, n_seg)

    if args.partner_mode == "inter_segment" and \
            args.min_sep >= int(seg["end"].sub(seg["start"]).min()):
        raise ValueError(
            f"--min_sep ({args.min_sep}) is >= the smallest somatic chromosome "
            f"({int(seg['end'].sub(seg['start']).min())} bp); small segments would "
            "become unreachable from adjacent anchors."
        )

    log.info("expectation scope: %s", args.expected_scope)
    if args.expected_scope == "within_segment" and args.partner_mode == "inter_segment":
        log.warning("--expected_scope within_segment normalises INTER-segment "
                    "contacts by an intra-segment expectation; the scopes do not "
                    "match. This is intended for --partner_mode pairwise_end.")

    seg_of_bin, own_left, own_right = build_bin_lookups(seg, nbins, args.binsize)
    cbr = build_cbrs(seg, chrom_len, args.flank_retained, args.flank_eliminated,
                     args.clip_frac)
    cbr = assign_groups(cbr, args.split_by, args.gap_threshold, args.size_threshold)
    win_cbr, win_rel, rel_centers, valid = build_window_index(
        cbr, nbins, args.binsize, args.flank_retained, args.flank_eliminated)
    n_cbr, n_rel = valid.shape
    log.info("%d CBRs; relative axis %d bins from %.0f to %.0f kb",
             n_cbr, n_rel, rel_centers[0] / 1e3, rel_centers[-1] / 1e3)
    cbr.to_csv("segment_geometry.txt", sep="\t", index=False)

    # ---- pairwise partner windows --------------------------------------
    # For each CBR the pairwise partner is the CBR at the OTHER end of the same
    # somatic chromosome: chrNN.L (seg.start) pairs with chrNN.R (seg.end).
    # pair_lo_bin/pair_hi_bin give that partner's bin interval, +/- pair_window.
    pair_lo_bin = np.zeros(n_cbr, dtype=np.int64)
    pair_hi_bin = np.zeros(n_cbr, dtype=np.int64)
    starts = seg["start"].values
    ends = seg["end"].values
    for ci, row in cbr.iterrows():
        partner_pos = ends[row["seg_idx"]] if row["side"] == "L" else starts[row["seg_idx"]]
        lo = max(0, partner_pos - args.pair_window)
        hi = min(chrom_len - 1, partner_pos + args.pair_window)
        pair_lo_bin[ci] = lo // args.binsize
        pair_hi_bin[ci] = hi // args.binsize

    if args.partner_mode == "pairwise_end":
        # The paired end sits seg_len away; a contact to it must survive min_sep.
        # Its own-segment size ranges 2.80-11.86 Mb, so min_sep must clear the
        # smallest segment minus the window, or the smallest chromosome's own
        # pairing is silently dropped.
        smallest = int(seg["end"].sub(seg["start"]).min())
        if args.min_sep >= smallest - args.pair_window:
            raise ValueError(
                f"--min_sep ({args.min_sep}) leaves no room for the pairwise "
                f"partner of the smallest somatic chromosome "
                f"(seg_len {smallest} bp, window {args.pair_window} bp). "
                f"Use --min_sep below {smallest - args.pair_window}."
            )

    # ---- pass 1 ---------------------------------------------------------
    nworkers = max(1, min(args.threads, len(args.samples)))
    exclude_signal = (args.partner_mode == "pairwise_end"
                      and args.exclude_signal_from_expected)
    if exclude_signal:
        log.info("excluding paired-end contacts from the E(s) estimate")
    jobs1 = [(p, l, args.chrom, args.binsize, nbins, args.chunksize,
              seg_of_bin, args.expected_scope,
              win_cbr, pair_lo_bin, pair_hi_bin, exclude_signal)
             for p, l in zip(args.samples, labels)]
    if nworkers > 1:
        with ProcessPoolExecutor(max_workers=nworkers) as ex:
            res1 = dict(ex.map(scan_pass1, jobs1))
    else:
        res1 = dict(scan_pass1(j) for j in jobs1)

    totals = {l: res1[l]["total"] for l in labels}
    for l in labels:
        if totals[l] == 0:
            raise RuntimeError(f"[{l}] no cis {args.chrom} contacts found")
    target = min(totals.values()) if args.downsample == "auto" else None
    p_keep = {l: (1.0 if target is None else min(1.0, target / totals[l])) for l in labels}
    if target is not None:
        log.info("Downsampling all libraries to %d cis contacts: %s",
                 target, {l: round(p_keep[l], 4) for l in labels})

    dbin_edges = make_distance_bins(nbins - 1, args.n_dbins)
    E = {}
    for l in labels:
        if args.distance_norm == "oe":
            # Depth is equalised by --downsample, so E is normalised only for
            # shape (distance decay), not for library size.
            if args.expected_scope == "within_segment":
                npairs_k = occupancy_pair_counts_within(
                    res1[l]["occ"], seg_of_bin, n_seg, nbins - 1)
            else:
                npairs_k = occupancy_pair_counts(res1[l]["occ"], nbins - 1)
            E[l] = expected_per_dbin(res1[l]["dist_hist"], npairs_k,
                                     dbin_edges,
                                     min_k=max(1, args.min_sep // args.binsize))
        else:
            E[l] = np.ones(len(dbin_edges) - 1, dtype=np.float64)

    # ---- pass 2 ---------------------------------------------------------
    jobs2 = [
        (p, l, args.chrom, args.binsize, nbins, args.chunksize, args.min_sep,
         win_cbr, win_rel, seg_of_bin, own_left, own_right,
         n_cbr, n_rel, n_seg, dbin_edges, E[l], p_keep[l], args.seed,
         args.partner_mode, pair_lo_bin, pair_hi_bin)
        for p, l in zip(args.samples, labels)
    ]
    if nworkers > 1:
        with ProcessPoolExecutor(max_workers=nworkers) as ex:
            res2 = dict(ex.map(scan_pass2, jobs2))
    else:
        res2 = dict(scan_pass2(j) for j in jobs2)

    key = "prof_oe" if args.distance_norm == "oe" else "prof_raw"
    profiles = {l: res2[l][key] for l in labels}
    raw_profiles = {l: res2[l]["prof_raw"] for l in labels}
    comps = {l: res2[l]["comp_oe"] for l in labels}
    if args.normalize and args.distance_norm == "oe":
        log.warning("--normalize on top of --distance_norm oe is usually "
                    "redundant; --downsample auto already equalises depth.")
    if args.normalize:
        for l in labels:
            f = args.scale / max(res2[l]["kept_total"], 1)
            profiles[l] = profiles[l] * f
            comps[l] = comps[l] * f

    pairwise = args.partner_mode == "pairwise_end"
    kind = ("Paired-end" if pairwise else "Inter-segment")
    ylabel = (f"{kind} contacts (O/E)" if args.distance_norm == "oe"
              else f"{kind} contacts")
    if args.normalize:
        ylabel += ", CPM"

    # ---- metaprofiles ---------------------------------------------------
    groups = ["all"] + ([] if args.split_by == "none"
                        else sorted(cbr["group"].unique().tolist()))
    table = pd.DataFrame({"rel_center_bp": rel_centers})
    prefix = os.path.splitext(args.output_plot)[0]

    for g in groups:
        rows = (np.arange(n_cbr) if g == "all"
                else np.where(cbr["group"].values == g)[0])
        if rows.size == 0:
            continue
        means, sems, ns = {}, {}, None
        for l in labels:
            m, s, n = metaprofile(profiles[l], valid, rows)
            blank = n < args.min_cbr_n
            m[blank] = np.nan
            s[blank] = np.nan
            means[l], sems[l], ns = m, s, n
            table[f"{g}__{l}__mean"] = m
            table[f"{g}__{l}__sem"] = s
        table[f"{g}__n_cbr"] = ns
        base = prefix if g == "all" else f"{prefix}_{g}"
        title = (f"{kind} interaction profiles centered on CBRs"
                 if g == "all" else f"{kind} interaction profiles: {g}")
        plot_metaprofile(rel_centers, means, sems, ns, labels, ylabel,
                         f"{title}  (n = {rows.size})", base, args.dpi,
                         args.binsize, args.ylim)
        plot_cbr_count(rel_centers, ns, f"{base}_cbr_support", args.dpi, args.binsize)
        log.info("group %s: %d CBRs -> %s.{png,svg,pdf}", g, rows.size, base)

    table.to_csv(args.output_table, sep="\t", index=False, float_format="%.6g")

    # ---- per-CBR pairwise summary --------------------------------------
    # One scalar per CBR per stage: the mean paired-end O/E over the retained
    # side approaching the break, i.e. rel_center in [-summary_window, 0). This
    # is the ramp that peaks at the CBR in pairwise_end mode, and is intended for
    # a Figure 6E-style per-region dot/box plot. Written only in pairwise_end.
    if pairwise and args.pair_summary:
        sw = args.summary_window if args.summary_window is not None else args.pair_window
        in_win = (rel_centers >= -sw) & (rel_centers < 0)
        n_in = int(in_win.sum())
        if n_in == 0:
            log.warning("--pair_summary: no bins in [-%d, 0); widen "
                        "--summary_window. Skipping summary.", sw)
        else:
            # Marginal coverage (contacts with either end in the window, any
            # partner) for the anchor summary window and for the paired-end
            # partner window.  A paired-end contact needs coverage at BOTH ends,
            # so a single unmappable window forces the product to ~0 regardless
            # of biology; these columns make that detectable.
            rel_in = np.flatnonzero(in_win)
            selb = (win_cbr >= 0) & np.isin(win_rel, rel_in)
            acov, pcov = {}, {}
            for l in labels:
                cv = res1[l]["cov"]
                acov[l] = np.bincount(win_cbr[selb], weights=cv[selb],
                                      minlength=n_cbr)
                pcov[l] = np.array([cv[pair_lo_bin[i]: pair_hi_bin[i] + 1].sum()
                                    for i in range(n_cbr)])

            srows = []
            for i, row in cbr.iterrows():
                rec = dict(name=row["name"], somatic=row["somatic"],
                           side=row["side"], group=row["group"],
                           seg_len=row["seg_len"], gap=row["gap"])
                for l in labels:
                    vals = profiles[l][i, in_win].astype(np.float64).copy()
                    rvals = raw_profiles[l][i, in_win].astype(np.float64).copy()
                    vmsk = valid[i, in_win]
                    vals[~vmsk] = np.nan
                    rvals[~vmsk] = np.nan
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", RuntimeWarning)
                        rec[f"{l}__pair_oe"] = np.nanmean(vals)
                        rec[f"{l}__pair_max"] = np.nanmax(vals) if vmsk.any() else np.nan
                        # Raw (un-normalised) contact count. Libraries are already
                        # depth-matched by --downsample, so a per-CBR ratio between
                        # stages is distance-matched by construction (same CBR,
                        # same separation) and needs no E(s) model at all.
                        rec[f"{l}__pair_raw"] = np.nansum(rvals)
                    rec[f"{l}__n_bins"] = int(vmsk.sum())
                    rec[f"{l}__anchor_cov"] = float(acov[l][i])
                    rec[f"{l}__partner_cov"] = float(pcov[l][i])
                srows.append(rec)
            summ = pd.DataFrame(srows)
            # Flag CBRs whose anchor or partner window is far below the typical
            # window in any stage: their paired-end value is coverage-limited.
            low = np.zeros(len(summ), dtype=bool)
            for l in labels:
                for side in ("anchor_cov", "partner_cov"):
                    v = summ[f"{l}__{side}"].values
                    med = np.median(v[v > 0]) if (v > 0).any() else 0.0
                    if med > 0:
                        low |= v < args.min_window_cov_frac * med
            summ["low_coverage"] = low
            if low.any():
                log.warning("%d/%d CBRs flagged low_coverage (window < %.0f%% of "
                            "median in some stage): %s", int(low.sum()), len(summ),
                            100 * args.min_window_cov_frac,
                            ", ".join(summ["name"][low].tolist()))
            summary_path = os.path.splitext(args.output_table)[0] + "_pair_summary.txt"
            summ.to_csv(summary_path, sep="\t", index=False, float_format="%.6g")
            log.info("pairwise summary: %d CBRs x %d stages over [-%d,0) (%d bins) -> %s",
                     len(summ), len(labels), sw, n_in, summary_path)

            # Per-region dot plot: paired-end O/E by stage, coloured by CBR group.
            fig, ax = plt.subplots(figsize=(1.6 * len(labels) + 2, 6))
            rng = np.random.default_rng(0)
            grp_names = sorted(cbr["group"].unique().tolist())
            gcolors = dict(zip(grp_names,
                               plt.cm.viridis(np.linspace(0, 0.85, len(grp_names)))))
            for xi, l in enumerate(labels):
                y = summ[f"{l}__pair_oe"].values.astype(float)
                for gname in grp_names:
                    sel = (summ["group"].values == gname) & np.isfinite(y)
                    if not sel.any():
                        continue
                    jitter = (rng.random(sel.sum()) - 0.5) * 0.28
                    ax.scatter(np.full(sel.sum(), xi) + jitter, y[sel],
                               s=26, alpha=0.7, edgecolor="none",
                               color=gcolors[gname],
                               label=gname if xi == 0 else None)
                good = np.isfinite(y)
                if good.any():
                    med = np.median(y[good])
                    ax.hlines(med, xi - 0.34, xi + 0.34, color="black", lw=2, zorder=5)
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels([f"{l}\n(n={int(summ[f'{l}__pair_oe'].notna().sum())})"
                                for l in labels])
            ax.set_ylabel(ylabel)
            ax.set_title(f"Paired-end interaction per CBR over [-{sw//1000},0) kb")
            ax.set_ylim(bottom=0)
            if len(grp_names) > 1:
                ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
                          frameon=False, title="CBR group")
            base = os.path.splitext(args.output_table)[0] + "_pair_summary"
            save_all_formats(fig, base, args.dpi)
            log.info("pairwise summary dot plot -> %s.{png,svg,pdf}", base)
    elif args.pair_summary and not pairwise:
        log.info("--pair_summary ignored (only meaningful with "
                 "--partner_mode pairwise_end)")

    # ---- components -----------------------------------------------------
    # Meaningful only for inter_segment; in pairwise_end every partner is the
    # anchor's own segment, so the breakdown is degenerate and skipped.
    if pairwise:
        log.info("partner_mode=pairwise_end: skipping component breakdown "
                 "(single partner per CBR)")
    else:
        cs = args.component_sample or labels[0]
        if cs not in comps:
            matches = [l for l in labels if cs in l]
            if not matches:
                raise ValueError(f"--component_sample {cs} matches no label")
            cs = matches[0]
        os.makedirs("components", exist_ok=True)
        for g in groups:
            rows = (np.arange(n_cbr) if g == "all"
                    else np.where(cbr["group"].values == g)[0])
            if rows.size == 0:
                continue
            sub = comps[cs][rows].astype(np.float64).copy()
            sub[~valid[rows]] = np.nan
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                mat = np.nanmean(sub, axis=0)
            mat = np.nan_to_num(mat, nan=0.0)
            plot_components(rel_centers, mat, seg["name"].tolist(), ylabel,
                            f"Partner somatic chromosome contributions ({cs}, {g})",
                            os.path.join("components", f"components_{g}_{cs}"),
                            args.dpi, args.binsize, args.component_top_n)

    # ---- grid + per-CBR -------------------------------------------------
    plot_grid(rel_centers, profiles, valid, cbr, labels, ylabel,
              os.path.splitext(args.grid_plot)[0], args.dpi, args.binsize)

    if args.individual:
        os.makedirs("individual_cbrs", exist_ok=True)
        for i, row in cbr.iterrows():
            per = {l: profiles[l][i] for l in labels}
            plot_single_cbr(rel_centers, per, valid[i], labels, row["name"], ylabel,
                            os.path.join("individual_cbrs", row["name"]),
                            args.dpi, args.binsize)
            df = pd.DataFrame({"rel_center_bp": rel_centers})
            for l in labels:
                y = per[l].astype(float).copy()
                y[~valid[i]] = np.nan
                df[l] = y
            df.to_csv(os.path.join("individual_cbrs", f"{row['name']}_data.txt"),
                      sep="\t", index=False, float_format="%.6g")

    log.info("Done. Outputs in %s", outdir)


if __name__ == "__main__":
    main()
