# cis_cbr_profile

Hi-C contacts around chromosomal breakage regions (CBRs) in a genome whose
retained material is tiled along a single germline chromosome, and the
per-chromosome quantification that follows from it. Used for the *Parascaris
univalens* analysis (Figures 6E and S10A–D).

## Contents

| Script | Produces |
|--------|----------|
| `cis_cbr_profile.py` | Contact metaprofiles centred on every CBR (Fig. 6E, S10A, S10B) |
| `plot_pair_summary.py` | Per-chromosome dot plot with paired Wilcoxon tests (Fig. S10C) |
| `make_null_tests_table.py` | Tests for systematic variation between chromosomes (Fig. S10D) |

The three form one chain. `cis_cbr_profile.py --pair_summary` writes
`cbr_profile_pair_summary.txt`, which is the sole input to both downstream
scripts:

```
cis_cbr_profile.py --partner_mode pairwise_end --pair_summary
        │
        └── cbr_profile_pair_summary.txt
                    ├── plot_pair_summary.py       → Fig. S10C
                    └── make_null_tests_table.py   → Fig. S10D
```

## Overview

This is the cis counterpart of `trans_cbr_profile/`. That analysis counts *trans*
(inter-chromosomal) contacts per bin, which works when the germline karyotype has
many chromosomes. The *Parascaris* germline genome is a single chromosome, so
every contact is cis and the trans definition returns nothing.

The structural equivalent is the **inter-segment** contact: the germline
chromosome is tiled by the retained blocks that become the somatic chromosomes,
separated by interstitial eliminated DNA. A contact counts for an anchor bin if
its partner lands in a retained segment that is not the anchor's own. This is the
exact analogue of "different chromosome" and works identically for pre- and
post-elimination libraries, as long as both are mapped to the germline assembly.

A second mode, `--partner_mode pairwise_end`, restricts partners to the CBR at
the opposite end of the same retained segment — the two CBRs that become the two
ends of one somatic chromosome, a **cognate CBR pair**.

Two complications are handled explicitly:

- **Distance decay.** Unlike true trans contacts, inter-segment cis contacts
  decay with genomic separation. `--min_sep` drops near-diagonal partners, and
  `--distance_norm oe` weights each contact by 1/E(s), with E(s) estimated per
  library from its own cis contacts and bin occupancy.
- **Window collisions.** CBR spacing is uneven, so a fixed flank runs one CBR's
  trace into the next. Each window is clipped independently at `--clip_frac` of
  the distance to the neighbouring CBR. Clipped positions are NaN, not zero, so
  they are excluded from the metaprofile mean, and the number of contributing
  CBRs is tracked per bin.

CBRs are derived from the segment BED rather than annotated separately: each
retained block contributes two, at its left and right boundaries, so N segments
give 2N CBRs (36 → 72 in *Parascaris*). Profiles are oriented so retained DNA is
always on the negative axis and eliminated DNA on the positive axis.

## Dependencies

```
numpy
pandas
matplotlib
scipy        # optional; needed for the statistical tests in the two downstream scripts
```

## Input files

| Argument | Description |
|----------|-------------|
| `--segments` | BED of retained blocks: `chrom  start  end  somatic_name`. One row per prospective somatic chromosome. For *Parascaris*, `data/pu_v3_germ_to_soma_mapping.bed` (36 rows). |
| `--chrom_sizes` | Two-column chrom.sizes for the germline assembly. For *Parascaris*, `data/pu_v3_chrom_sizes.txt`. |
| `--chrom` | Germline chromosome to analyse; must match the name used in both files above. |
| `--samples` | HiC-Pro `allValidPairs` files, one per stage. Columns 2, 3, 5, 6 are read as chr1, pos1, chr2, pos2. |

Hi-C matrices and reads are available from GEO under accession GSE315650
(*Parascaris*) and GSE314626 (*Ascaris*). `allValidPairs` files are large HiC-Pro
intermediates and are not included in this repository; they are regenerated from
the SRA reads with HiC-Pro.

The genome assemblies are deposited separately and are not in this repository:
the *Parascaris univalens* germline assembly used here is v3, which differs from
the previously published v2 (= v1.11) only in the orientation of somatic
chromosome 1. Boundaries between retained and eliminated DNA are unchanged, and
there are no insertions or deletions, so germline coordinates and therefore all
CBR positions are identical between the two versions. Assembly accession:
[ACCESSION].

## Usage

Inter-segment mode (Figures S10A and S10B):

```bash
python cis_cbr_profile.py \
    --segments data/pu_v3_germ_to_soma_mapping.bed \
    --chrom_sizes data/pu_v3_chrom_sizes.txt \
    --chrom chrX \
    --samples data/sample_10hr.allValidPairs \
              data/sample_17hr.allValidPairs \
              data/sample_36hr.allValidPairs \
    --labels 10hr 17hr 36hr \
    --binsize 5000 --flank_retained 500000 --flank_eliminated 100000 \
    --min_sep 1000000 --distance_norm oe --downsample auto \
    --split_by gap --gap_threshold 100000 --min_cbr_n 10 \
    --partner_mode inter_segment \
    --threads 3 \
    --output_dir cbr_intersegment_output
```

Cognate-pair mode (Figure 6E, and the summary table feeding Figure S10C):

```bash
python cis_cbr_profile.py \
    --segments data/pu_v3_germ_to_soma_mapping.bed \
    --chrom_sizes data/pu_v3_chrom_sizes.txt \
    --chrom chrX \
    --samples data/sample_10hr.allValidPairs \
              data/sample_17hr.allValidPairs \
              data/sample_36hr.allValidPairs \
    --labels 10hr 17hr 36hr \
    --binsize 5000 --flank_retained 500000 --flank_eliminated 100000 \
    --min_sep 1000000 --distance_norm oe --downsample auto \
    --partner_mode pairwise_end --pair_window 100000 --pair_summary \
    --split_by size --size_threshold 6000000 --min_cbr_n 10 \
    --threads 3 \
    --output_dir cbr_pairwise_output
```

## Downstream quantification

Both scripts take the `cbr_profile_pair_summary.txt` written by the
`pairwise_end` run above. Both exclude coverage-flagged CBRs and collapse the two
CBRs of a chromosome to a single value, because they derive from largely
overlapping read sets and are not independent observations; n is reported as the
number of chromosomes.

Per-chromosome dot plot (Figure S10C):

```bash
python plot_pair_summary.py \
    --summary cbr_pairwise_output/cbr_profile_pair_summary.txt \
    --stages 10hr 17hr 36hr \
    --metric oe \
    --annotate_p adjacent --test wilcoxon \
    --width 3.125 --fontsize 8 \
    --output_dir figures --output_prefix pair_summary_by_chrom
```

Consecutive stages are compared with a paired Wilcoxon signed-rank test, since
the same chromosomes are measured at every stage. Points are drawn as a beeswarm
so the distribution stays readable at panel width.

Tests for systematic variation between chromosomes (Figure S10D):

```bash
python make_null_tests_table.py \
    --summary cbr_pairwise_output/cbr_profile_pair_summary.txt \
    --stages 10hr 17hr 36hr \
    --ratio 36hr 17hr \
    --post_elimination 36hr \
    --output_dir tables --output_prefix null_tests_table
```

Three predictors — somatic chromosome length, adjacent eliminated block size, and
X-derived vs autosomal identity — are tested against the normalised value at each
stage plus the raw within-chromosome ratio. Columns named in `--post_elimination`
are daggered: once the genome is fragmented, E(s) estimated over germline
coordinates makes single-stage normalised values incomparable across chromosomes
of differing length, so the raw ratio is the appropriate test at that stage. The
footnote explaining this is generated automatically.

Both scripts build their panels at final printed size (`--width`, in inches) so
no rescaling is needed, and leave SVG text editable for Illustrator.

## Key parameters

| Flag | Default | Notes |
|------|---------|-------|
| `--partner_mode` | `inter_segment` | `inter_segment` or `pairwise_end` |
| `--binsize` | 5000 | Bin size in bp |
| `--flank_retained` | 500000 | Extent of the negative (retained) axis |
| `--flank_eliminated` | 100000 | Extent of the positive (eliminated) axis |
| `--clip_frac` | 0.5 | Stop halfway to the neighbouring CBR |
| `--min_sep` | 1000000 | Must stay below the smallest retained segment; validated at startup |
| `--distance_norm` | `oe` | `oe` weights by 1/E(s); `none` uses raw counts |
| `--downsample` | `auto` | Subsample all libraries to the smallest cis depth. CPM alone does not correct sparsity across libraries of different depth. |
| `--min_cbr_n` | 10 | Blank metaprofile bins carried by fewer CBRs |
| `--pair_summary` | off | Per-CBR scalar table and dot plot; `pairwise_end` only |

## Outputs

```
<output_dir>/
├── cbr_profile.txt                    per-bin metaprofile values + CBR counts
├── cbr_profile.{png,svg,pdf}          metaprofile, all CBRs
├── cbr_profile_cbr_support.{png,...}  contributing-CBR count per position
├── cbr_profile_<group>.{png,...}      metaprofile per --split_by group
├── cbr_profile_pair_summary.txt       per-CBR scalar (--pair_summary)
├── cbrs_grid.{png,svg,pdf}            grid overview of all CBRs
├── segment_geometry.txt               derived CBR table: position, orientation,
│                                      clip limits, group assignment
├── components/                        stacked-area partner-segment breakdown
└── individual_cbrs/                   per-CBR traces and data tables
```

All figures are written as PNG, SVG, and PDF. SVG text is left editable
(`svg.fonttype = 'none'`) for Adobe Illustrator.

## Figure mapping

| Figure | Mode | Key flags |
|--------|------|-----------|
| Fig. 6E | `pairwise_end` | `--pair_window 100000 --distance_norm oe --downsample auto` |
| Fig. S10A | `inter_segment` | `--distance_norm oe --downsample auto` |
| Fig. S10B | `inter_segment` | `cbr_support` panel from the same run as S10A |
| Fig. S10C | `pairwise_end` | `--pair_summary`, then `plot_pair_summary.py` |
| Fig. S10D | `pairwise_end` | `--pair_summary`, then `make_null_tests_table.py --post_elimination 36hr` |

Figures 6E and S10A share a common relative axis (−500 kb retained to +100 kb
eliminated) so the two contact classes can be compared directly.
