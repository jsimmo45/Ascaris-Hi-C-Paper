# cbr_heatmap

Generate Hi-C contact heatmaps organized by chromosome break regions (CBRs)
in *Ascaris*, with PCA analysis of trans-chromosomal interaction profiles.

## Overview

`cbr_heatmap.py` builds a CBR-by-CBR contact matrix from HiC-Pro sparse
matrices, with regions sorted by type (terminal CBRs first, then internal)
and within each type by chromosome number.  Within-chromosome contacts can be
masked to isolate trans-chromosomal interactions.

Produces:
- **Contact heatmap** — CBR × CBR matrix with custom white→red→black colormap,
  blue lines separating terminal from internal regions
- **PCA plot** — terminal vs. internal CBR profiles with 95% confidence
  ellipses
- **Trans interaction statistics** — dot plots comparing terminal-terminal,
  internal-internal, and terminal-internal contact strengths
- **Saddle enrichment score** — quantifies whether same-type CBR pairs
  interact more than cross-type pairs
- **Multi-sample comparison** — bar chart comparing interaction categories
  across developmental stages

## Dependencies

```
numpy
pandas
matplotlib
seaborn
scipy
```

## Input files

| File | In repo? | Description |
|------|----------|-------------|
| `data/cbr_v50_200kb_split_internal.bed` | Yes | CBR windows BED (chr start end name type) |
| `data/AG_10kb_hicpro_bins.bed` | Yes | HiC-Pro bins file for 10kb resolution |
| `data/AG_20kb_hicpro_bins.bed` | Yes | HiC-Pro bins file for 20kb resolution |
| ICE-normalized matrices | No | HiC-Pro sparse matrices; regenerate from GEO [GSE314626](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE314626). |

## Usage

### Single sample

```bash
python cbr_heatmap.py \
    --bed data/cbr_v50_200kb_split_internal.bed \
    --bins data/AG_10kb_hicpro_bins.bed \
    --matrix matrix/10kb/48hr_10kb_iced.matrix \
    --output results/48hr \
    --bin-size 10000 \
    --vmaxes 'Vmax300:300' \
    --colormap white_to_red \
    --cpm-normalize \
    --mask-chromosomes \
    --trans-vmax 2000 \
    --dpi 300
```

### Multi-sample comparison (after running individual samples)

```bash
python cbr_heatmap.py \
    --comparison-mode \
    --comparison-dir results/hic_plots_10kb_mask_chroms \
    --dpi 300
```

## Parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--bed` | required | CBR BED file (chr start end name type) |
| `--bins` | required | HiC-Pro bins BED file |
| `--matrix` | required | HiC-Pro ICE-normalized sparse matrix |
| `--output` | required | Output path prefix |
| `--bin-size` | required | Matrix bin size in bp |
| `--vmaxes` | `auto:auto` | Heatmap vmax specs (name:value, comma-separated) |
| `--colormap` | `white_to_red` | Colormap name |
| `--cpm-normalize` | off | Apply CPM normalization |
| `--mask-chromosomes` | off | Mask within-chromosome contacts (show trans only) |
| `--trans-vmax` | 2000 | Y-axis max for trans-stats plot |
| `--dpi` | 300 | Output resolution |
| `--comparison-mode` | off | Generate multi-sample comparison from existing results |

## Outputs

Per sample (in organized subdirectories):
- `*_{vmax_name}.png/.pdf` — contact heatmap
- `*_pca.png/.pdf` — PCA of trans interaction profiles
- `*_trans_stats.png/.pdf` — dot plot of interaction strengths by category
- `*_saddle_score.txt` — enrichment score

Comparison:
- `trans_interaction_comparison.png/.pdf` — bar chart across stages

## Figure mapping

| Figure | Sample | Flags | Notes |
|--------|--------|-------|-------|
| Fig. 1D, 1F | 48hr | `--cpm-normalize --mask-chromosomes --vmaxes Vmax300:300` | 10kb + 20kb |
