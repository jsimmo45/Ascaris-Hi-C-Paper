# Ascaris and Parascaris Hi-C Analysis

Code and reference files for Simmons JR, Xue T, McCord RP, Wang J.
*Spatial genome organization in nematodes with programmed DNA elimination.*
Genome Research (2026).

Hi-C analysis of chromatin interactions during programmed DNA elimination (PDE)
in *Ascaris suum* and *Parascaris univalens*.

## Data availability

Hi-C data generated in this study are deposited in NCBI GEO under accession
numbers **GSE314626** (*Ascaris*) and **GSE315650** (*Parascaris*).

The scripts here operate on ICE-normalized sparse matrices and `allValidPairs`
files produced by [HiC-Pro](https://github.com/nservant/HiC-Pro). Both are large
intermediates and are not included in this repository. To regenerate them, run
HiC-Pro with the appropriate reference genome and apply ICE balancing.
Eigenvector files (from [FAN-C](https://fan-c.readthedocs.io/)
`fanc compartments`) are small enough to include and are under
`data/eigenvectors/`.

### Reference assemblies

| Assembly | Used for | Accession |
|----------|----------|-----------|
| *Ascaris suum* germline v50 | All *Ascaris* analyses. Improved in this study by Hi-C-guided manual curation and PacBio extension of chromosome termini (Figure S1). |
| *Parascaris univalens* germline v3 | *Parascaris* analyses. |

*Parascaris* v3 differs from the previously published v2 (= v1.11) only in the
orientation of somatic chromosome 1. Boundaries between retained and eliminated
DNA are unchanged and there are no insertions or deletions, so germline
coordinates are identical between the two versions. Analyses performed in
germline coordinates give the same result with either; the orientation matters
only where somatic chromosome 1 is displayed in somatic coordinates.

## Repository structure

```
data/                                    Shared reference files
├── 20000/, 40000/                       HiC-Pro bin mappings
├── interaction_regions/                 Region pair BED files
├── eigenvectors/                        EV1 files (Ascaris + Parascaris)
│   ├── ascaris/prepde/, postpde/
│   └── parascaris/prepde/, postpde/
├── new_ends_post_pde/                   Break region BEDs at multiple window sizes
├── AG_v50_eliminated_strict.bed         Eliminated regions
├── AG_v50_chrom_sizes.txt               Chromosome sizes
├── AG_v50_chrom.bed                     Chromosome regions BED
├── AG_10kb_hicpro_bins.bed              HiC-Pro abs.bed (10 kb)
├── AG_20kb_hicpro_bins.bed              HiC-Pro abs.bed (20 kb)
├── as_v50_prepde_to_postpde_coordinates.bed
├── pu_v2_germ_to_soma_mapping.bed
├── pu_v2_eliminated.bed
├── pu_v3_germ_to_soma_mapping.bed       Germline-to-somatic mapping, v3
├── pu_v3_chrom_sizes.txt                Chromosome sizes, v3
├── as_to_pu_chrom_order.txt
├── flip_or_not.txt
├── cbr_v50_500kb_windows_labeled.bed
├── cbr_v50_500kb_windows_cbr.bed
├── cbr_v50_200kb_split_internal.bed
├── chrom_order_pre.txt, chrom_order_post.txt
└── combined_whole_genome_100000bp_normalized.txt

interaction_scoring/                     Interaction frequency quantification
hic_triangle_plots/                      Triangular Hi-C heatmaps across development
ascarid_eigenvector_comparison/          Ascaris vs Parascaris EV1 comparison
developmental_eigenvector_chromosome_stacked/
                                         Ascaris EV1 across developmental stages
saddle_plot_analysis/                    Compartment saddle plots and strength
multiomics_ab_compartments/              Multi-omics A/B compartment comparison
clustering_heatmap/                      Hierarchical clustering of interaction patterns
distance_decay/                          Distance decay curves (FAN-C expected values)
somatic_chromosome_end_analysis/         Multi-omics at new chromosome ends vs internal breaks
plot_insulation/                         Per-chromosome insulation score plots + boxplot
insulation_genome/                       Genome-wide stacked insulation score plots
eigenvector_genome/                      Genome-wide stacked eigenvector 1 plots
trans_cbr_profile/                       Trans interaction profiles at CBRs
cis_cbr_profile/                         Cis interaction profiles at CBRs on a single
                                         germline chromosome, plus per-chromosome
                                         quantification and null tests
cbr_heatmap/                             CBR-by-CBR contact heatmap + PCA
hic_subregion/                           Custom subregion Hi-C heatmaps
```

Each analysis directory contains its own `README.md` with usage examples,
parameter descriptions, and a figure mapping table.

## Dependencies

All scripts are Python 3 and use standard scientific packages:

```
numpy
pandas
matplotlib
seaborn
scipy
```

Upstream tools used to generate the inputs (not required to run these scripts):

- [HiC-Pro](https://github.com/nservant/HiC-Pro) — read mapping, matrix
  generation, ICE normalization
- [FAN-C](https://fan-c.readthedocs.io/) — eigenvectors, insulation scores,
  expected values

## License

MIT
