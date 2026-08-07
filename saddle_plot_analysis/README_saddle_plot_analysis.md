# saddle_plot_analysis

Compartment saddle plots and derived compartmentalization metrics for *Ascaris*
Hi-C data across developmental timepoints spanning programmed DNA elimination
(PDE). Produces Figures 5C and S8A–C.

## Overview

A saddle plot summarises A/B compartmentalization by sorting genomic bins by
eigenvector (PC1) value, binning them into quantiles, and averaging contact
frequency within each quantile pair. Strong compartmentalization gives high
contact in the AA and BB corners and low contact in the AB corners.

From that the script derives:

- **Saddle strength**, (AA + BB) / (AB + BA), the standard scalar summary of how
  well the two compartments are separated
- **Decomposition** into the AA, BB, and AB components separately, which
  distinguishes euchromatic from heterochromatic self-association
- **Compartment strength**, mean |PC1|, an eigenvector-only measure that does not
  depend on the contact matrix
- **Compartment switching** between consecutive stages, classified A-to-B,
  B-to-A, or stable, optionally filtered by a |PC1| threshold so that bins near
  zero (where the compartment call is unreliable) are excluded

Boundary sharpness, organisational entropy, and A/B separation are also computed.
These were exploratory and do not appear in the paper, but are left in because
they share the same data loading and are cheap once the matrices are in memory.

When `--cbr-bed` is supplied, each metric is additionally split into CBR-proximal
and non-CBR bins for comparison.

## Dependencies

```
numpy
pandas
matplotlib
seaborn
scipy
```

## Input files

| Argument | Description |
|----------|-------------|
| `--matrix-dir` | Base directory holding `prepde/` and `postpde/` subfolders of ICE-normalised HiC-Pro sparse matrices |
| `--eigenvector-dir` | Parent directory holding `prepde/` and `postpde/` subfolders of FAN-C eigenvector files (`chrom  start  end  PC1`) |
| `--matrix-pattern` | Matrix filename pattern; `{}` is replaced by the timepoint |
| `--eigenvector-pattern` | Eigenvector filename pattern; `{}` is replaced by the timepoint |
| `--cbr-bed` | Optional BED of chromosome breakage regions for CBR-vs-genome comparisons |

Eigenvector files are small and are included in this repository under
`data/eigenvectors/ascaris/`. Hi-C matrices are large HiC-Pro intermediates and
are not; regenerate them from the SRA reads with HiC-Pro and apply ICE balancing.
Data are at GEO accession GSE314626 (*Ascaris*).

A/B assignment follows the sign of PC1: positive is A (active), negative is B
(inactive).

## Usage

**Figure 5C** — per-timepoint saddle plots only. `--plot-only` stops after the
saddle plots and their strength summary:

```bash
python saddle_plots_basic.py \
    --timepoints teste,ovary,0hr,48hr,60hr,5day,10day \
    --pre-pde-timepoints teste,ovary,0hr,48hr,60hr \
    --post-pde-timepoints 5day,10day \
    --matrix-dir matrix_files_100kb \
    --eigenvector-dir data/eigenvectors/ascaris \
    --max-bins 5000 \
    --global-scale \
    --boundary-linewidth 4.0 --boundary-color yellow \
    --stage-names 'teste:teste,ovary:ovary,0hr:1 cell,48hr:2-4 cell,60hr:4-8 cell,5day:32-64 cell,10day:L1' \
    --plot-only \
    --output-dir output_saddle
```

`--global-scale` puts every timepoint on one colour scale, which is what makes
the panels comparable across stages; without it each is scaled independently.

**Figures S8A–C** — full run. Same command without `--plot-only`, plus the PC1
threshold used for the switching analysis:

```bash
python saddle_plots_basic.py \
    --timepoints teste,ovary,0hr,48hr,60hr,5day,10day \
    --pre-pde-timepoints teste,ovary,0hr,48hr,60hr \
    --post-pde-timepoints 5day,10day \
    --matrix-dir matrix_files_100kb \
    --eigenvector-dir data/eigenvectors/ascaris \
    --max-bins 5000 \
    --pc1-threshold 0.005 \
    --global-scale \
    --cbr-bed data/cbr_v50_500kb_windows_labeled.bed \
    --stage-names 'teste:teste,ovary:ovary,0hr:1 cell,48hr:2-4 cell,60hr:4-8 cell,5day:32-64 cell,10day:L1' \
    --output-dir output_full
```

## Key parameters

| Flag | Default | Notes |
|------|---------|-------|
| `--plot-only` | off | Stop after the saddle plots; skips switching, boundary, entropy, and strength analyses |
| `--global-scale` | off | One colour scale across all timepoints; needed for cross-stage comparison |
| `--max-bins` | 5000 | Bins to analyse; 0 uses all. The saddle computation is O(n²) in Python, so this is the main cost control |
| `--pc1-threshold` | none | Exclude bins with \|PC1\| below this from switching analysis, where the compartment call is unreliable |
| `--cbr-bed` | none | Adds CBR-vs-non-CBR comparison panels to every metric |
| `--boundary-linewidth`, `--boundary-color` | 4.0, yellow | The dashed A/B boundary drawn on each saddle plot |
| `--stage-names` | none | `key:value` pairs mapping timepoint keys to display labels |
| `--output-dir` | `output` | All outputs are written here |

## Outputs

```
<output-dir>/
├── saddle_plot_{prepde,postpde}_{timepoint}.{png,svg}   per-timepoint saddle plots
├── saddle_strength_across_development.{png,svg}         (AA+BB)/(AB+BA) by stage
├── saddle_strength_decomposition.{png,svg}              AA, BB, AB, and overall
├── saddle_strength_summary.csv
├── saddle_strength_detailed.csv
│
│   # full run only (omitted under --plot-only)
├── sequential_compartment_switching.{png,svg}           stacked A-to-B / B-to-A
├── compartment_switching_{t1}_vs_{t2}.{png,svg}         per-transition detail
├── compartment_strength.{png,svg}                       mean |PC1| by stage
├── ab_separation.{png,svg}                              mean A minus mean B
├── boundary_strength.{png,svg}                          mean |grad PC1| at boundaries
├── compartment_entropy.{png,svg}                        transitions per bin
├── *_cbr_comparison.{png,svg}                           CBR vs non-CBR, when --cbr-bed given
└── *_summary.csv                                        one per analysis
```

SVG text is left editable (`svg.fonttype = 'none'`) for Adobe Illustrator.

## Figure mapping

| Figure | Output file | Invocation |
|--------|-------------|------------|
| Fig. 5C | `saddle_plot_{prepde,postpde}_{timepoint}.svg` | `--plot-only --global-scale` |
| Fig. S8A | `compartment_strength.svg` | full run |
| Fig. S8B | `saddle_strength_decomposition.svg` | full run |
| Fig. S8C | `sequential_compartment_switching.svg` | full run, `--pc1-threshold 0.005` |

Both figures were assembled and labelled in Adobe Illustrator; this script
produces the individual panels.
