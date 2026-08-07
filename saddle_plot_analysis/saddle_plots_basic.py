#!/usr/bin/env python3
"""
saddle_plots_basic.py
=====================
Compartment saddle plot analysis for Ascaris suum Hi-C data across
developmental timepoints spanning programmed DNA elimination (PDE).

Computes saddle plots (contact enrichment stratified by eigenvector quantile),
saddle strength metrics (AA, BB, AB compartment interaction strengths), and
optionally performs compartment switching, boundary sharpness, entropy, and
compartment strength analyses.

Outputs (all to --output-dir):
  - Per-timepoint saddle plots (PNG + SVG)
  - Saddle strength across development (line plot + CSV)
  - Saddle strength decomposition (AA, BB, AB components)
  - Compartment switching analysis between consecutive timepoints
  - Boundary sharpness and entropy analyses
  - CBR (chromosome break region) comparisons when --cbr-bed is provided

Dependencies:
  numpy, pandas, matplotlib, seaborn, scipy

Example usage:
  python saddle_plots_basic.py \\
      --timepoints teste,ovary,0hr,48hr,60hr,5day,10day \\
      --pre-pde-timepoints teste,ovary,0hr,48hr,60hr \\
      --post-pde-timepoints 5day,10day \\
      --matrix-dir matrix_files_100kb \\
      --eigenvector-dir .. \\
      --matrix-pattern 'as_{}_iced_100kb.matrix' \\
      --eigenvector-pattern 'as_{}_iced_100kb.matrix.eigenvector' \\
      --max-bins 5000 \\
      --pc1-threshold 0.005 \\
      --global-scale \\
      --boundary-linewidth 4.0 \\
      --boundary-color yellow \\
      --cbr-bed data/cbr_v50_500kb_windows_labeled.bed \\
      --stage-names 'teste:teste,ovary:ovary,0hr:1 cell,48hr:2-4 cell,60hr:4-8 cell,5day:32-64 cell,10day:L1' \\
      --output-dir output/

Input files:
  Hi-C matrices: ICE-normalized sparse matrices (HiC-Pro format) in
      --matrix-dir/{prepde,postpde}/ with filenames matching --matrix-pattern.
  Eigenvector files: FANC-format (chr  start  end  PC1) in
      --eigenvector-dir/{prepde,postpde}/eigenvectors/.
  --cbr-bed (optional): BED file of chromosome break regions for
      CBR-vs-genome comparisons.
  --chromosome-mapping (optional): BED mapping pre-PDE to post-PDE
      chromosome coordinates.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import mannwhitneyu
import os
from matplotlib.colors import LogNorm
import re
import sys
import argparse

# Set publication-quality font defaults
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 14
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['axes.titlesize'] = 22
plt.rcParams['xtick.labelsize'] = 18
plt.rcParams['ytick.labelsize'] = 18
plt.rcParams['legend.fontsize'] = 18
plt.rcParams['figure.titlesize'] = 24
plt.rcParams['svg.fonttype'] = 'none'  # editable text in SVG for Illustrator

#################################################
#         ARGUMENT PARSING                      #
#################################################

def parse_args():
    parser = argparse.ArgumentParser(
        description='Saddle plot and compartment analysis for Ascaris Hi-C data')

    # Timepoints
    parser.add_argument('--timepoints', default='teste,ovary,0hr,48hr,60hr,5day,10day',
                        help='Comma-separated timepoints in developmental order')
    parser.add_argument('--pre-pde-timepoints', default='teste,ovary,0hr,48hr,60hr',
                        help='Comma-separated pre-PDE timepoints')
    parser.add_argument('--post-pde-timepoints', default='5day,10day',
                        help='Comma-separated post-PDE timepoints')

    # Directories
    parser.add_argument('--matrix-dir', default='matrix_files_100kb',
                        help='Base directory for Hi-C matrices')
    parser.add_argument('--eigenvector-dir', default='data/eigenvectors/ascaris',
                        help='Parent directory for eigenvector subfolders')
    parser.add_argument('--subfolder-pre', default='prepde',
                        help='Subfolder name for pre-PDE matrices')
    parser.add_argument('--subfolder-post', default='postpde',
                        help='Subfolder name for post-PDE matrices')
    parser.add_argument('--eigen-subfolder-pre', default='prepde',
                        help='Path to pre-PDE eigenvector files')
    parser.add_argument('--eigen-subfolder-post', default='postpde',
                        help='Path to post-PDE eigenvector files')

    # File patterns
    parser.add_argument('--matrix-pattern', default='as_{}_iced_100kb.matrix',
                        help='Matrix filename pattern ({} = timepoint)')
    parser.add_argument('--eigenvector-pattern', default='as_{}_iced_100kb.matrix.eigenvector',
                        help='Eigenvector filename pattern ({} = timepoint)')
    parser.add_argument('--post-pde-matrix-pattern', default=None,
                        help='Post-PDE matrix pattern (default: same as --matrix-pattern)')
    parser.add_argument('--post-pde-eigenvector-pattern', default=None,
                        help='Post-PDE eigenvector pattern (default: same as --eigenvector-pattern)')

    # Analysis parameters
    parser.add_argument('--max-bins', type=int, default=5000,
                        help='Max bins to analyze (default: 5000, 0 = all)')
    parser.add_argument('--pc1-threshold', type=float, default=None,
                        help='PC1 threshold for filtering compartment calls')
    parser.add_argument('--cbr-bed', default=None,
                        help='BED file with chromosome break regions')
    parser.add_argument('--chromosome-mapping', default=None,
                        help='BED file mapping pre-PDE to post-PDE coordinates')
    parser.add_argument('--internal-break-window', type=int, default=1000000,
                        help='Window size around internal breaks in bp')

    # Plot options
    parser.add_argument('--plot-only', action='store_true',
                        help='Only generate saddle plots (skip switching/boundary analyses)')
    parser.add_argument('--global-scale', action='store_true',
                        help='Use same color scale for all saddle plots')
    parser.add_argument('--boundary-linewidth', type=float, default=4.0,
                        help='Line width for A/B boundary lines on saddle plots')
    parser.add_argument('--boundary-color', default='yellow',
                        help='Color for A/B boundary lines on saddle plots')

    # Display names
    parser.add_argument('--stage-names', default='',
                        help='Map timepoints to display names (key1:val1,key2:val2,...)')

    # Output
    parser.add_argument('--output-dir', default='output',
                        help='Output directory')

    return parser.parse_args()


# Parse arguments
args = parse_args()

# Unpack into module-level variables used by the rest of the script
PLOT_ONLY = args.plot_only
GLOBAL_SCALE = args.global_scale
BOUNDARY_LINEWIDTH = args.boundary_linewidth
BOUNDARY_COLOR = args.boundary_color

timepoints = [t.strip() for t in args.timepoints.split(',')]
pre_pde_timepoints = [t.strip() for t in args.pre_pde_timepoints.split(',')]
post_pde_timepoints = [t.strip() for t in args.post_pde_timepoints.split(',')]

matrix_dir = args.matrix_dir
eigenvector_dir = args.eigenvector_dir
subfolder_pre = args.subfolder_pre
subfolder_post = args.subfolder_post
eigen_subfolder_pre = args.eigen_subfolder_pre
eigen_subfolder_post = args.eigen_subfolder_post

matrix_pattern = args.matrix_pattern
eigenvector_pattern = args.eigenvector_pattern
post_pde_matrix_pattern = args.post_pde_matrix_pattern or matrix_pattern
post_pde_eigenvector_pattern = args.post_pde_eigenvector_pattern or eigenvector_pattern

max_bins = args.max_bins if args.max_bins > 0 else None
pc1_threshold = args.pc1_threshold

output_dir = args.output_dir
cbr_bed_file = args.cbr_bed
chromosome_mapping_file = args.chromosome_mapping
internal_break_window = args.internal_break_window

# Parse display names
STAGE_NAMES = {}
if args.stage_names:
    for pair in args.stage_names.split(','):
        if ':' in pair:
            key, val = pair.split(':', 1)
            STAGE_NAMES[key] = val

print("=" * 70)
print("CONFIGURATION")
print("=" * 70)
print(f"\nPlotting Options:")
print(f"  PLOT_ONLY: {PLOT_ONLY}")
print(f"  GLOBAL_SCALE: {GLOBAL_SCALE}")
print(f"  BOUNDARY_LINEWIDTH: {BOUNDARY_LINEWIDTH}")
print(f"  BOUNDARY_COLOR: {BOUNDARY_COLOR}")

print(f"\nTimepoints:")
print(f"  All: {', '.join(timepoints)}")
print(f"  Pre-PDE: {', '.join(pre_pde_timepoints)}")
print(f"  Post-PDE: {', '.join(post_pde_timepoints)}")

print(f"\nDirectories:")
print(f"  Matrix dir: {matrix_dir}")
print(f"  Eigenvector dir: {eigenvector_dir}")
print(f"  Pre-PDE subfolder: {subfolder_pre}")
print(f"  Post-PDE subfolder: {subfolder_post}")

print(f"\nFile Patterns:")
print(f"  Matrix: {matrix_pattern}")
print(f"  Eigenvector: {eigenvector_pattern}")

print(f"\nAnalysis Parameters:")
print(f"  Max bins: {max_bins}")
print(f"  PC1 threshold: {pc1_threshold}")
print(f"  CBR file: {cbr_bed_file}")
print(f"  Internal break window: {internal_break_window}")
print("=" * 70)

def get_display_name(timepoint):
    """Convert internal timepoint name to display name"""
    return STAGE_NAMES.get(timepoint, timepoint)

#################################################
#           CORE ANALYSIS FUNCTIONS             #
#################################################

def get_file_paths(timepoint, matrix_dir, eigenvector_dir, matrix_pattern, eigenvector_pattern, 
                   post_pde_matrix_pattern, post_pde_eigenvector_pattern, 
                   subfolder_pre, subfolder_post, eigen_subfolder_pre, eigen_subfolder_post,
                   pre_pde_timepoints, post_pde_timepoints):
    """
    Get the correct file paths for a given timepoint based on whether it's pre or post-PDE
    """
    if timepoint in pre_pde_timepoints:
        matrix_path = os.path.join(matrix_dir, subfolder_pre, matrix_pattern.format(timepoint))
        eigen_path = os.path.join(eigenvector_dir, eigen_subfolder_pre, eigenvector_pattern.format(timepoint))
    elif timepoint in post_pde_timepoints:
        matrix_path = os.path.join(matrix_dir, subfolder_post, post_pde_matrix_pattern.format(timepoint))
        eigen_path = os.path.join(eigenvector_dir, eigen_subfolder_post, post_pde_eigenvector_pattern.format(timepoint))
    else:
        # Default to pre-PDE pattern
        matrix_path = os.path.join(matrix_dir, subfolder_pre, matrix_pattern.format(timepoint))
        eigen_path = os.path.join(eigenvector_dir, eigen_subfolder_pre, eigenvector_pattern.format(timepoint))
    
    return matrix_path, eigen_path

def load_sparse_matrix(file_path):
    """
    Load a sparse matrix in 3-column format (bin1, bin2, value)
    Returns a pandas DataFrame with the matrix data
    """
    print(f"Loading matrix from {file_path}")
    if not os.path.exists(file_path):
        print(f"ERROR: Matrix file not found: {file_path}")
        return None
    
    matrix_df = pd.read_csv(file_path, sep='\t', header=None, 
                           names=['bin1', 'bin2', 'value'])
    print(f"  Loaded {len(matrix_df)} interactions")
    return matrix_df

def load_eigenvector(file_path):
    """
    CORRECTED: Load eigenvector data with 4-column format:
    chr   start   end   PC1_value
    
    Assigns A/B compartments based on PC1 sign (positive = A, negative = B)
    """
    print(f"Loading eigenvector from {file_path}")
    
    if not os.path.exists(file_path):
        print(f"ERROR: Eigenvector file not found: {file_path}")
        return None
    
    try:
        # Load the file - assume tab-separated with no header
        eigen_df = pd.read_csv(file_path, sep='\t', header=None)
        
        # CORRECTED: Handle 4-column format
        if eigen_df.shape[1] == 4:
            eigen_df.columns = ['chrom', 'start', 'end', 'PC1']
            print(f"  Loaded 4-column format (chr, start, end, PC1)")
        elif eigen_df.shape[1] == 5:
            eigen_df.columns = ['chrom', 'start', 'end', 'compartment', 'PC1']
            print(f"  Loaded 5-column format with compartment labels")
        elif eigen_df.shape[1] == 6:
            eigen_df.columns = ['chrom', 'start', 'end', 'compartment', 'PC1', 'extra']
            print(f"  Loaded 6-column format")
        else:
            print(f"ERROR: Unexpected number of columns: {eigen_df.shape[1]}")
            print(f"  First few rows:\n{eigen_df.head()}")
            return None
        
        # Convert PC1 to numeric if it's not already
        eigen_df['PC1'] = pd.to_numeric(eigen_df['PC1'], errors='coerce')
        
        # CORRECTED: Assign compartment labels based on PC1 sign if not present
        if 'compartment' not in eigen_df.columns:
            eigen_df['compartment'] = eigen_df['PC1'].apply(lambda x: 'A' if x > 0 else 'B')
            print(f"  Assigned A/B compartments based on PC1 sign")
        
        # Create a numeric bin ID
        eigen_df['bin'] = np.arange(1, len(eigen_df) + 1)
        
        # Add a binary compartment indicator (1 for A, 0 for B)
        eigen_df['comp_binary'] = (eigen_df['compartment'] == 'A').astype(int)
        
        print(f"  Loaded {len(eigen_df)} bins")
        print(f"  PC1 range: [{eigen_df['PC1'].min():.4f}, {eigen_df['PC1'].max():.4f}]")
        print(f"  A compartment bins: {(eigen_df['compartment'] == 'A').sum()}")
        print(f"  B compartment bins: {(eigen_df['compartment'] == 'B').sum()}")
        
        return eigen_df
    
    except Exception as e:
        print(f"ERROR loading eigenvector file: {e}")
        import traceback
        traceback.print_exc()
        return None

def sparse_to_dense(sparse_df, max_bin=None, min_bin=1):
    """
    Convert sparse matrix to dense format
    Optional: filter to specific bin range
    """
    if sparse_df is None or len(sparse_df) == 0:
        print("ERROR: Cannot convert empty sparse matrix to dense")
        return None
    
    if max_bin is None:
        max_bin = max(sparse_df['bin1'].max(), sparse_df['bin2'].max())
    
    print(f"  Converting sparse to dense matrix (bins {min_bin}-{max_bin})")
    
    # Filter to bins in range
    sparse_df = sparse_df[(sparse_df['bin1'] >= min_bin) & (sparse_df['bin1'] <= max_bin) &
                          (sparse_df['bin2'] >= min_bin) & (sparse_df['bin2'] <= max_bin)]
    
    # Create empty dense matrix
    n_bins = max_bin - min_bin + 1
    dense_matrix = np.zeros((n_bins, n_bins))
    
    # Fill matrix (adjust bin IDs if min_bin > 1)
    for _, row in sparse_df.iterrows():
        i, j = int(row['bin1'] - min_bin), int(row['bin2'] - min_bin)
        if 0 <= i < n_bins and 0 <= j < n_bins:
            dense_matrix[i, j] = row['value']
            # Make symmetric
            if i != j:
                dense_matrix[j, i] = row['value']
    
    print(f"  Created {n_bins}x{n_bins} dense matrix")
    return dense_matrix

def create_saddle_plot(matrix, eigenvector, n_bins=20, percentile_range=(1, 99)):
    """
    Create a saddle plot from a dense matrix and eigenvector values
    Uses quantile-based binning to ensure even distribution
    
    Returns:
    saddle_matrix: numpy 2D array - the saddle plot matrix
    bin_edges: numpy 1D array - the bin edges for PC1 values
    saddle_strength: float - compartmentalization strength (AA+BB)/(AB+BA)
    aa_strength: float - A-to-A interaction strength
    bb_strength: float - B-to-B interaction strength
    ab_strength: float - A-to-B interaction strength
    """
    # Get percentile range of eigenvector to avoid outliers
    low_p, high_p = percentile_range
    ev_min, ev_max = np.percentile(eigenvector, [low_p, high_p])
    
    # Filter bins by eigenvector value
    valid_bins = np.logical_and(eigenvector >= ev_min, eigenvector <= ev_max)
    valid_eigenvector = eigenvector[valid_bins]
    
    # Create quantile-based bins for even distribution
    quantiles = np.linspace(0, 100, n_bins+1)
    bin_edges = np.percentile(valid_eigenvector, quantiles)
    
    # Ensure unique bin edges
    bin_edges = np.unique(bin_edges)
    n_bins = len(bin_edges) - 1
    
    # Digitize eigenvector values into bins
    digitized = np.zeros_like(eigenvector, dtype=int)
    digitized[valid_bins] = np.digitize(eigenvector[valid_bins], bin_edges) - 1
    digitized = np.clip(digitized, 0, n_bins-1)
    
    # Initialize saddle matrix
    saddle_matrix = np.zeros((n_bins, n_bins))
    count_matrix = np.zeros((n_bins, n_bins))
    
    # Aggregate interactions by eigenvector quantiles
    n = len(eigenvector)
    for i in range(n):
        if not valid_bins[i]:
            continue
        for j in range(i, n):
            if not valid_bins[j]:
                continue
            
            bin_i, bin_j = digitized[i], digitized[j]
            saddle_matrix[bin_i, bin_j] += matrix[i, j]
            count_matrix[bin_i, bin_j] += 1
            
            if i != j:  # Fill symmetric part
                saddle_matrix[bin_j, bin_i] += matrix[i, j]
                count_matrix[bin_j, bin_i] += 1
    
    # Normalize by counts
    mask = count_matrix > 0
    normalized_matrix = np.full_like(saddle_matrix, np.nan, dtype=float)
    normalized_matrix[mask] = saddle_matrix[mask] / count_matrix[mask]
    
    # Replace NaN with small value for visualization
    min_value = np.nanmin(normalized_matrix) / 10 if not np.all(np.isnan(normalized_matrix)) else 1e-6
    normalized_matrix = np.nan_to_num(normalized_matrix, nan=min_value)
    
    # Calculate saddle strength components: (AA + BB) / (AB + BA)
    # Find the boundary between A (negative PC1) and B (positive PC1) compartments
    zero_idx = np.searchsorted(bin_edges, 0)
    
    if 0 < zero_idx < n_bins:
        # B compartment (negative PC1): bins 0 to zero_idx-1
        # A compartment (positive PC1): bins zero_idx to n_bins-1
        
        # AA interactions (upper right quadrant)
        aa_region = normalized_matrix[zero_idx:, zero_idx:]
        aa_strength = np.nanmean(aa_region) if aa_region.size > 0 else np.nan
        
        # BB interactions (lower left quadrant)
        bb_region = normalized_matrix[:zero_idx, :zero_idx]
        bb_strength = np.nanmean(bb_region) if bb_region.size > 0 else np.nan
        
        # AB interactions (upper left quadrant)
        ab_region = normalized_matrix[zero_idx:, :zero_idx]
        ab_mean = np.nanmean(ab_region) if ab_region.size > 0 else 0
        
        # BA interactions (lower right quadrant) - should be same as AB due to symmetry
        ba_region = normalized_matrix[:zero_idx, zero_idx:]
        ba_mean = np.nanmean(ba_region) if ba_region.size > 0 else 0
        
        # AB strength (average of AB and BA)
        ab_strength = (ab_mean + ba_mean) / 2
        
        # Calculate overall saddle strength
        within_compartment = aa_strength + bb_strength
        between_compartment = ab_mean + ba_mean
        
        if between_compartment > 0:
            saddle_strength = within_compartment / between_compartment
        else:
            saddle_strength = np.nan
    else:
        saddle_strength = np.nan
        aa_strength = np.nan
        bb_strength = np.nan
        ab_strength = np.nan
    
    return normalized_matrix, bin_edges, saddle_strength, aa_strength, bb_strength, ab_strength

def plot_saddle_plot(saddle_matrix, bin_edges, timepoint, output_dir="output", subfolder="", 
                     vmin=None, vmax=None, boundary_linewidth=3.0, boundary_color='black'):
    """
    Create a saddle plot with PC1 values on axes
    Saves in PNG and SVG formats for publication
    
    Parameters:
    -----------
    saddle_matrix : numpy array
        The saddle plot matrix to visualize
    bin_edges : numpy array
        PC1 bin edges
    timepoint : str
        Name of the timepoint
    output_dir : str
        Output directory
    subfolder : str
        Subfolder name for output files
    vmin : float, optional
        Minimum value for color scale (for global scaling)
    vmax : float, optional
        Maximum value for color scale (for global scaling)
    boundary_linewidth : float
        Line width for A/B boundary lines
    boundary_color : str
        Color for A/B boundary lines
    """
    os.makedirs(output_dir, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(11, 10))
    
    # Calculate bin centers for axis labels
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Plot with log normalization
    if vmin is not None and vmax is not None:
        im = ax.imshow(saddle_matrix, cmap='coolwarm', norm=LogNorm(vmin=vmin, vmax=vmax), aspect='auto')
    else:
        im = ax.imshow(saddle_matrix, cmap='coolwarm', norm=LogNorm(), aspect='auto')
    
    # Set title with display name
    display_name = get_display_name(timepoint)
    ax.set_title(display_name, fontsize=26, fontweight='bold', pad=20)
    ax.set_xlabel('Eigenvector (PC1)', fontsize=22, fontweight='bold', labelpad=12)
    ax.set_ylabel('Eigenvector (PC1)', fontsize=22, fontweight='bold', labelpad=12)
    
    # Add colorbar with larger labels
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Contact enrichment', fontsize=20, fontweight='bold', labelpad=18)
    cbar.ax.tick_params(labelsize=18)
    
    # Set tick marks to show PC1 values
    step = max(1, len(bin_centers) // 8)
    tick_positions = np.arange(0, len(bin_centers), step)
    tick_labels = [f"{bin_centers[i]:.3f}" for i in tick_positions]
    
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=18, fontweight='bold')
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(tick_labels, fontsize=18, fontweight='bold')
    
    # Add lines at PC1=0 to indicate A/B boundary with custom linewidth and color
    zero_idx = np.searchsorted(bin_centers, 0)
    if 0 < zero_idx < len(bin_centers):
        ax.axhline(y=zero_idx-0.5, color=boundary_color, linestyle='--', alpha=0.8, linewidth=boundary_linewidth)
        ax.axvline(x=zero_idx-0.5, color=boundary_color, linestyle='--', alpha=0.8, linewidth=boundary_linewidth)
    
    plt.tight_layout()
    
    # Save in both formats
    base_name = f'saddle_plot_{subfolder}_{timepoint}' if subfolder else f'saddle_plot_{timepoint}'
    plt.savefig(f'{output_dir}/{base_name}.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/{base_name}.svg', format='svg', bbox_inches='tight')
    
    plt.close()

def analyze_compartment_boundaries(eigen_data, timepoints, window_size=5, cbr_df=None, output_dir="output"):
    """
    Analyze compartment boundary sharpness using PC1 gradient
    Creates CBR vs rest of genome comparison plot
    """
    print("\n=== Analyzing compartment boundary sharpness ===")
    os.makedirs(output_dir, exist_ok=True)
    
    boundary_strength_by_tp = {}
    boundary_strength_cbr_by_tp = {}
    boundary_strength_non_cbr_by_tp = {}
    
    # Prepare CBR bin IDs if available
    cbr_bins = set()
    if cbr_df is not None:
        for tp in timepoints:
            if tp in eigen_data:
                eigen_df = eigen_data[tp]
                for _, cbr in cbr_df.iterrows():
                    overlapping_bins = eigen_df[(eigen_df['chrom'] == cbr['chrom']) & 
                                               (eigen_df['end'] > cbr['start']) & 
                                               (eigen_df['start'] < cbr['end'])]
                    cbr_bins.update(overlapping_bins['bin'].tolist())
        print(f"  Found {len(cbr_bins)} unique CBR bins")
    
    for tp in timepoints:
        if tp not in eigen_data:
            continue
            
        df = eigen_data[tp]
        boundary_strengths = []
        cbr_boundary_strengths = []
        non_cbr_boundary_strengths = []
        
        # Detect compartment boundaries
        for i in range(1, len(df)-1):
            if df.iloc[i]['compartment'] != df.iloc[i-1]['compartment']:
                # Calculate gradient at boundary
                pc1_values = df.iloc[max(0, i-window_size):min(len(df), i+window_size+1)]['PC1'].values
                if len(pc1_values) > 1:
                    gradient = np.abs(np.gradient(pc1_values)).mean()
                    boundary_strengths.append(gradient)
                    
                    # Check if in CBR
                    if cbr_bins and df.iloc[i]['bin'] in cbr_bins:
                        cbr_boundary_strengths.append(gradient)
                    elif cbr_bins:
                        non_cbr_boundary_strengths.append(gradient)
        
        if boundary_strengths:
            boundary_strength_by_tp[tp] = np.mean(boundary_strengths)
            print(f"  {tp}: {len(boundary_strengths)} boundaries, mean strength = {boundary_strength_by_tp[tp]:.6f}")
        
        if cbr_boundary_strengths:
            boundary_strength_cbr_by_tp[tp] = np.mean(cbr_boundary_strengths)
        if non_cbr_boundary_strengths:
            boundary_strength_non_cbr_by_tp[tp] = np.mean(non_cbr_boundary_strengths)
    
    # Plot genome-wide boundary strength
    if boundary_strength_by_tp:
        fig, ax = plt.subplots(figsize=(12, 7))
        
        all_timepoints = [tp for tp in timepoints if tp in boundary_strength_by_tp]
        strengths = [boundary_strength_by_tp[tp] for tp in all_timepoints]
        
        bars = ax.bar(range(len(all_timepoints)), strengths, color='darkblue', alpha=0.7, edgecolor='black', linewidth=2)
        
        # Add values as text
        for i, (tp, strength) in enumerate(zip(all_timepoints, strengths)):
            ax.text(i, strength + strength*0.02, f"{strength:.6f}", ha='center', va='bottom', fontweight='bold', fontsize=14)
        
        ax.set_ylabel('Boundary Strength (mean |∇PC1|)', fontsize=18, fontweight='bold', labelpad=10)
        ax.set_title('Compartment Boundary Sharpness Across Development', fontsize=20, fontweight='bold', pad=15)
        
        # Use display names
        display_labels = [get_display_name(tp) for tp in all_timepoints]
        ax.set_xticks(range(len(all_timepoints)))
        ax.set_xticklabels(display_labels, rotation=30, ha='right', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/boundary_strength.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'{output_dir}/boundary_strength.svg', format='svg', bbox_inches='tight')
        plt.close()
    
    # Plot CBR vs non-CBR comparison
    if boundary_strength_cbr_by_tp and boundary_strength_non_cbr_by_tp:
        fig, ax = plt.subplots(figsize=(14, 7))
        
        common_tps = [tp for tp in timepoints 
                     if tp in boundary_strength_cbr_by_tp and tp in boundary_strength_non_cbr_by_tp]
        
        if common_tps:
            x = np.arange(len(common_tps))
            width = 0.35
            
            cbr_values = [boundary_strength_cbr_by_tp[tp] for tp in common_tps]
            non_cbr_values = [boundary_strength_non_cbr_by_tp[tp] for tp in common_tps]
            
            ax.bar(x - width/2, cbr_values, width, label='CBR regions', color='red', alpha=0.7, edgecolor='black', linewidth=1.5)
            ax.bar(x + width/2, non_cbr_values, width, label='Non-CBR regions', color='blue', alpha=0.7, edgecolor='black', linewidth=1.5)
            
            # Add value annotations
            for i, (cbr, non_cbr) in enumerate(zip(cbr_values, non_cbr_values)):
                ax.text(i - width/2, cbr + cbr*0.02, f"{cbr:.6f}", ha='center', va='bottom', fontsize=12, fontweight='bold')
                ax.text(i + width/2, non_cbr + non_cbr*0.02, f"{non_cbr:.6f}", ha='center', va='bottom', fontsize=12, fontweight='bold')
            
            ax.set_ylabel('Boundary Strength (mean |∇PC1|)', fontsize=18, fontweight='bold', labelpad=10)
            ax.set_title('Compartment Boundary Strength: CBR vs. Non-CBR Regions', 
                        fontsize=20, fontweight='bold', pad=15)
            
            # Use display names
            display_labels = [get_display_name(tp) for tp in common_tps]
            ax.set_xticks(x)
            ax.set_xticklabels(display_labels, rotation=30, ha='right', fontsize=14, fontweight='bold')
            ax.legend(fontsize=16)
            ax.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            plt.savefig(f'{output_dir}/boundary_strength_cbr_comparison.png', dpi=300, bbox_inches='tight')
            plt.savefig(f'{output_dir}/boundary_strength_cbr_comparison.svg', format='svg', bbox_inches='tight')
            plt.close()
    
    # Save summary
    summary_df = pd.DataFrame({
        'Timepoint': list(boundary_strength_by_tp.keys()),
        'Stage_Name': [get_display_name(tp) for tp in boundary_strength_by_tp.keys()],
        'Boundary_Strength': list(boundary_strength_by_tp.values())
    })
    
    if boundary_strength_cbr_by_tp:
        cbr_data = pd.DataFrame({
            'Timepoint': list(boundary_strength_cbr_by_tp.keys()),
            'CBR_Boundary_Strength': list(boundary_strength_cbr_by_tp.values())
        })
        summary_df = summary_df.merge(cbr_data, on='Timepoint', how='outer')
    
    if boundary_strength_non_cbr_by_tp:
        non_cbr_data = pd.DataFrame({
            'Timepoint': list(boundary_strength_non_cbr_by_tp.keys()),
            'NonCBR_Boundary_Strength': list(boundary_strength_non_cbr_by_tp.values())
        })
        summary_df = summary_df.merge(non_cbr_data, on='Timepoint', how='outer')
    
    summary_df.to_csv(f'{output_dir}/boundary_analysis_summary.csv', index=False)
    print(f"  Saved boundary analysis to {output_dir}/boundary_analysis_summary.csv")
    
    return boundary_strength_by_tp, boundary_strength_cbr_by_tp, boundary_strength_non_cbr_by_tp

def analyze_compartment_entropy(eigen_data, timepoints, cbr_df=None, output_dir="output"):
    """
    Analyze compartment organization entropy (measure of disorder/chaos)
    Higher entropy = more switching between A and B compartments
    """
    print("\n=== Analyzing compartment organization entropy ===")
    os.makedirs(output_dir, exist_ok=True)
    
    entropy_by_tp = {}
    entropy_cbr_by_tp = {}
    entropy_non_cbr_by_tp = {}
    
    # Prepare CBR bins
    cbr_bins = set()
    if cbr_df is not None:
        for tp in timepoints:
            if tp in eigen_data:
                eigen_df = eigen_data[tp]
                for _, cbr in cbr_df.iterrows():
                    overlapping_bins = eigen_df[(eigen_df['chrom'] == cbr['chrom']) & 
                                               (eigen_df['end'] > cbr['start']) & 
                                               (eigen_df['start'] < cbr['end'])]
                    cbr_bins.update(overlapping_bins['bin'].tolist())
        print(f"  Found {len(cbr_bins)} unique CBR bins")
    
    for tp in timepoints:
        if tp not in eigen_data:
            continue
            
        df = eigen_data[tp]
        
        # Count transitions for genome-wide entropy
        transitions = 0
        for i in range(1, len(df)):
            if df.iloc[i]['compartment'] != df.iloc[i-1]['compartment']:
                transitions += 1
        
        # Calculate entropy: transitions per bin
        entropy = transitions / len(df) if len(df) > 0 else 0
        entropy_by_tp[tp] = entropy
        
        # CBR and non-CBR entropy
        if cbr_bins:
            cbr_mask = df['bin'].isin(cbr_bins)
            
            # CBR entropy
            if cbr_mask.sum() > 1:
                cbr_df_subset = df[cbr_mask].reset_index(drop=True)
                cbr_transitions = 0
                for i in range(1, len(cbr_df_subset)):
                    if cbr_df_subset.iloc[i]['compartment'] != cbr_df_subset.iloc[i-1]['compartment']:
                        cbr_transitions += 1
                entropy_cbr_by_tp[tp] = cbr_transitions / len(cbr_df_subset)
            
            # Non-CBR entropy
            if (~cbr_mask).sum() > 1:
                non_cbr_df_subset = df[~cbr_mask].reset_index(drop=True)
                non_cbr_transitions = 0
                for i in range(1, len(non_cbr_df_subset)):
                    if non_cbr_df_subset.iloc[i]['compartment'] != non_cbr_df_subset.iloc[i-1]['compartment']:
                        non_cbr_transitions += 1
                entropy_non_cbr_by_tp[tp] = non_cbr_transitions / len(non_cbr_df_subset)
        
        print(f"  {tp}: Entropy = {entropy:.6f} (transitions/bin)")
    
    # Plot genome-wide entropy
    if entropy_by_tp:
        fig, ax = plt.subplots(figsize=(12, 7))
        
        all_timepoints = [tp for tp in timepoints if tp in entropy_by_tp]
        entropies = [entropy_by_tp[tp] for tp in all_timepoints]
        
        bars = ax.bar(range(len(all_timepoints)), entropies, color='purple', alpha=0.7, edgecolor='black', linewidth=2)
        
        # Add values as text
        for i, (tp, entropy) in enumerate(zip(all_timepoints, entropies)):
            ax.text(i, entropy + entropy*0.02, f"{entropy:.6f}", ha='center', va='bottom', fontweight='bold', fontsize=14)
        
        ax.set_ylabel('Compartment Entropy (transitions/bin)', fontsize=18, fontweight='bold', labelpad=10)
        ax.set_title('Compartment Organization Entropy Across Development', fontsize=20, fontweight='bold', pad=15)
        
        # Use display names
        display_labels = [get_display_name(tp) for tp in all_timepoints]
        ax.set_xticks(range(len(all_timepoints)))
        ax.set_xticklabels(display_labels, rotation=30, ha='right', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/compartment_entropy.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'{output_dir}/compartment_entropy.svg', format='svg', bbox_inches='tight')
        plt.close()
    
    # Plot CBR vs non-CBR comparison
    if entropy_cbr_by_tp and entropy_non_cbr_by_tp:
        fig, ax = plt.subplots(figsize=(14, 7))
        
        common_tps = [tp for tp in timepoints 
                     if tp in entropy_cbr_by_tp and tp in entropy_non_cbr_by_tp]
        
        if common_tps:
            x = np.arange(len(common_tps))
            width = 0.35
            
            cbr_values = [entropy_cbr_by_tp[tp] for tp in common_tps]
            non_cbr_values = [entropy_non_cbr_by_tp[tp] for tp in common_tps]
            
            ax.bar(x - width/2, cbr_values, width, label='CBR regions', color='red', alpha=0.7, edgecolor='black', linewidth=1.5)
            ax.bar(x + width/2, non_cbr_values, width, label='Non-CBR regions', color='blue', alpha=0.7, edgecolor='black', linewidth=1.5)
            
            # Add value annotations
            for i, (cbr, non_cbr) in enumerate(zip(cbr_values, non_cbr_values)):
                ax.text(i - width/2, cbr + cbr*0.02, f"{cbr:.6f}", ha='center', va='bottom', fontsize=12, fontweight='bold')
                ax.text(i + width/2, non_cbr + non_cbr*0.02, f"{non_cbr:.6f}", ha='center', va='bottom', fontsize=12, fontweight='bold')
            
            ax.set_ylabel('Compartment Entropy (transitions/bin)', fontsize=18, fontweight='bold', labelpad=10)
            ax.set_title('Compartment Organization Entropy: CBR vs. Non-CBR Regions', 
                        fontsize=20, fontweight='bold', pad=15)
            
            # Use display names
            display_labels = [get_display_name(tp) for tp in common_tps]
            ax.set_xticks(x)
            ax.set_xticklabels(display_labels, rotation=30, ha='right', fontsize=14, fontweight='bold')
            ax.legend(fontsize=16)
            ax.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            plt.savefig(f'{output_dir}/compartment_entropy_cbr_comparison.png', dpi=300, bbox_inches='tight')
            plt.savefig(f'{output_dir}/compartment_entropy_cbr_comparison.svg', format='svg', bbox_inches='tight')
            plt.close()
    
    # Save summary
    summary_df = pd.DataFrame({
        'Timepoint': list(entropy_by_tp.keys()),
        'Stage_Name': [get_display_name(tp) for tp in entropy_by_tp.keys()],
        'Genome_Entropy': list(entropy_by_tp.values())
    })
    
    if entropy_cbr_by_tp:
        cbr_data = pd.DataFrame({
            'Timepoint': list(entropy_cbr_by_tp.keys()),
            'CBR_Entropy': list(entropy_cbr_by_tp.values())
        })
        summary_df = summary_df.merge(cbr_data, on='Timepoint', how='outer')
    
    if entropy_non_cbr_by_tp:
        non_cbr_data = pd.DataFrame({
            'Timepoint': list(entropy_non_cbr_by_tp.keys()),
            'NonCBR_Entropy': list(entropy_non_cbr_by_tp.values())
        })
        summary_df = summary_df.merge(non_cbr_data, on='Timepoint', how='outer')
    
    summary_df.to_csv(f'{output_dir}/entropy_analysis_summary.csv', index=False)
    print(f"  Saved entropy summary to {output_dir}/entropy_analysis_summary.csv")
    
    return entropy_by_tp, entropy_cbr_by_tp, entropy_non_cbr_by_tp

def analyze_compartment_strength(eigen_data, timepoints, cbr_df=None, output_dir="output"):
    """
    Analyze compartment strength and A/B separation
    
    Compartment strength = mean absolute PC1 (how far from zero)
    A/B separation = difference between mean A and mean B compartments
    """
    print("\n=== Analyzing compartment strength and A/B separation ===")
    os.makedirs(output_dir, exist_ok=True)
    
    strength_by_tp = {}
    separation_by_tp = {}
    strength_cbr_by_tp = {}
    strength_non_cbr_by_tp = {}
    separation_cbr_by_tp = {}
    separation_non_cbr_by_tp = {}
    
    # Prepare CBR bins
    cbr_bins = set()
    if cbr_df is not None:
        for tp in timepoints:
            if tp in eigen_data:
                eigen_df = eigen_data[tp]
                for _, cbr in cbr_df.iterrows():
                    overlapping_bins = eigen_df[(eigen_df['chrom'] == cbr['chrom']) & 
                                               (eigen_df['end'] > cbr['start']) & 
                                               (eigen_df['start'] < cbr['end'])]
                    cbr_bins.update(overlapping_bins['bin'].tolist())
        print(f"  Found {len(cbr_bins)} unique CBR bins")
    
    for tp in timepoints:
        if tp not in eigen_data:
            continue
            
        df = eigen_data[tp]
        pc1_values = df['PC1'].values
        
        # Genome-wide metrics
        # Compartment strength: mean absolute PC1
        strength = np.mean(np.abs(pc1_values))
        strength_by_tp[tp] = strength
        
        # A/B separation: difference between A and B compartment means
        a_compartment = df[df['compartment'] == 'A']['PC1'].values
        b_compartment = df[df['compartment'] == 'B']['PC1'].values
        
        if len(a_compartment) > 0 and len(b_compartment) > 0:
            separation = np.mean(a_compartment) - np.mean(b_compartment)
            separation_by_tp[tp] = separation
        
        # CBR and non-CBR metrics
        if cbr_bins:
            cbr_mask = df['bin'].isin(cbr_bins)
            
            # CBR regions
            if cbr_mask.sum() > 0:
                cbr_pc1 = df.loc[cbr_mask, 'PC1'].values
                strength_cbr_by_tp[tp] = np.mean(np.abs(cbr_pc1))
                
                cbr_a = df.loc[cbr_mask & (df['compartment'] == 'A'), 'PC1'].values
                cbr_b = df.loc[cbr_mask & (df['compartment'] == 'B'), 'PC1'].values
                if len(cbr_a) > 0 and len(cbr_b) > 0:
                    separation_cbr_by_tp[tp] = np.mean(cbr_a) - np.mean(cbr_b)
            
            # Non-CBR regions
            non_cbr_mask = ~cbr_mask
            if non_cbr_mask.sum() > 0:
                non_cbr_pc1 = df.loc[non_cbr_mask, 'PC1'].values
                strength_non_cbr_by_tp[tp] = np.mean(np.abs(non_cbr_pc1))
                
                non_cbr_a = df.loc[non_cbr_mask & (df['compartment'] == 'A'), 'PC1'].values
                non_cbr_b = df.loc[non_cbr_mask & (df['compartment'] == 'B'), 'PC1'].values
                if len(non_cbr_a) > 0 and len(non_cbr_b) > 0:
                    separation_non_cbr_by_tp[tp] = np.mean(non_cbr_a) - np.mean(non_cbr_b)
        
        print(f"  {tp}: Strength = {strength:.4f}, A/B Separation = {separation_by_tp.get(tp, 0):.4f}")
    
    # Plot 1: Compartment Strength
    if strength_by_tp:
        fig, ax = plt.subplots(figsize=(12, 7))
        
        all_timepoints = [tp for tp in timepoints if tp in strength_by_tp]
        strengths = [strength_by_tp[tp] for tp in all_timepoints]
        
        bars = ax.bar(range(len(all_timepoints)), strengths, color='darkgreen', alpha=0.7, edgecolor='black', linewidth=2)
        
        # Add values as text
        for i, (tp, strength) in enumerate(zip(all_timepoints, strengths)):
            ax.text(i, strength + 0.002, f"{strength:.4f}", ha='center', va='bottom', fontweight='bold', fontsize=14)
        
        ax.set_ylabel('Compartment Strength (mean |PC1|)', fontsize=18, fontweight='bold', labelpad=10)
        ax.set_title('Compartment Strength Across Development', fontsize=20, fontweight='bold', pad=15)
        
        # Use display names
        display_labels = [get_display_name(tp) for tp in all_timepoints]
        ax.set_xticks(range(len(all_timepoints)))
        ax.set_xticklabels(display_labels, rotation=30, ha='right', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/compartment_strength.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'{output_dir}/compartment_strength.svg', format='svg', bbox_inches='tight')
        plt.close()
    
    # Plot 2: A/B Separation
    if separation_by_tp:
        fig, ax = plt.subplots(figsize=(12, 7))
        
        all_timepoints = [tp for tp in timepoints if tp in separation_by_tp]
        separations = [separation_by_tp[tp] for tp in all_timepoints]
        
        bars = ax.bar(range(len(all_timepoints)), separations, color='darkorange', alpha=0.7, edgecolor='black', linewidth=2)
        
        # Add values as text
        for i, (tp, sep) in enumerate(zip(all_timepoints, separations)):
            text_y = sep + 0.002 if sep > 0 else sep - 0.002
            va = 'bottom' if sep > 0 else 'top'
            ax.text(i, text_y, f"{sep:.4f}", ha='center', va=va, fontweight='bold', fontsize=14)
        
        ax.axhline(0, color='black', linestyle='--', alpha=0.5, linewidth=1)
        ax.set_ylabel('A/B Separation (mean A - mean B PC1)', fontsize=18, fontweight='bold', labelpad=10)
        ax.set_title('A/B Compartment Separation Across Development', fontsize=20, fontweight='bold', pad=15)
        
        # Use display names
        display_labels = [get_display_name(tp) for tp in all_timepoints]
        ax.set_xticks(range(len(all_timepoints)))
        ax.set_xticklabels(display_labels, rotation=30, ha='right', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/ab_separation.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'{output_dir}/ab_separation.svg', format='svg', bbox_inches='tight')
        plt.close()
    
    # Plot 3: CBR comparison for Compartment Strength
    if strength_cbr_by_tp and strength_non_cbr_by_tp:
        fig, ax = plt.subplots(figsize=(14, 7))
        
        common_tps = [tp for tp in timepoints 
                     if tp in strength_cbr_by_tp and tp in strength_non_cbr_by_tp]
        
        if common_tps:
            x = np.arange(len(common_tps))
            width = 0.35
            
            cbr_values = [strength_cbr_by_tp[tp] for tp in common_tps]
            non_cbr_values = [strength_non_cbr_by_tp[tp] for tp in common_tps]
            
            ax.bar(x - width/2, cbr_values, width, label='CBR regions', color='red', alpha=0.7, edgecolor='black', linewidth=1.5)
            ax.bar(x + width/2, non_cbr_values, width, label='Non-CBR regions', color='blue', alpha=0.7, edgecolor='black', linewidth=1.5)
            
            # Add value annotations
            for i, (cbr, non_cbr) in enumerate(zip(cbr_values, non_cbr_values)):
                ax.text(i - width/2, cbr + 0.002, f"{cbr:.4f}", ha='center', va='bottom', fontsize=12, fontweight='bold')
                ax.text(i + width/2, non_cbr + 0.002, f"{non_cbr:.4f}", ha='center', va='bottom', fontsize=12, fontweight='bold')
                
                # Difference annotation
                diff = cbr - non_cbr
                pct_diff = (diff / non_cbr) * 100 if non_cbr != 0 else 0
                max_val = max(cbr, non_cbr)
                
                ax.text(i, max_val + 0.008, f"Δ: {diff:+.4f}\n({pct_diff:+.1f}%)", 
                       ha='center', va='bottom', fontsize=10, fontweight='bold',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.6))
            
            ax.set_ylabel('Compartment Strength (mean |PC1|)', fontsize=18, fontweight='bold', labelpad=10)
            ax.set_title('Compartment Strength: CBR vs. Non-CBR Regions', 
                        fontsize=20, fontweight='bold', pad=15)
            
            # Use display names
            display_labels = [get_display_name(tp) for tp in common_tps]
            ax.set_xticks(x)
            ax.set_xticklabels(display_labels, rotation=30, ha='right', fontsize=14, fontweight='bold')
            ax.legend(fontsize=16)
            ax.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            plt.savefig(f'{output_dir}/compartment_strength_cbr_comparison.png', dpi=300, bbox_inches='tight')
            plt.savefig(f'{output_dir}/compartment_strength_cbr_comparison.svg', format='svg', bbox_inches='tight')
            plt.close()
    
    # Plot 4: CBR comparison for A/B Separation
    if separation_cbr_by_tp and separation_non_cbr_by_tp:
        fig, ax = plt.subplots(figsize=(14, 7))
        
        common_tps = [tp for tp in timepoints 
                     if tp in separation_cbr_by_tp and tp in separation_non_cbr_by_tp]
        
        if common_tps:
            x = np.arange(len(common_tps))
            width = 0.35
            
            cbr_values = [separation_cbr_by_tp[tp] for tp in common_tps]
            non_cbr_values = [separation_non_cbr_by_tp[tp] for tp in common_tps]
            
            ax.bar(x - width/2, cbr_values, width, label='CBR regions', color='red', alpha=0.7, edgecolor='black', linewidth=1.5)
            ax.bar(x + width/2, non_cbr_values, width, label='Non-CBR regions', color='blue', alpha=0.7, edgecolor='black', linewidth=1.5)
            
            # Add value annotations
            for i, (cbr, non_cbr) in enumerate(zip(cbr_values, non_cbr_values)):
                text_y_cbr = cbr + 0.002 if cbr > 0 else cbr - 0.002
                va_cbr = 'bottom' if cbr > 0 else 'top'
                ax.text(i - width/2, text_y_cbr, f"{cbr:.4f}", ha='center', va=va_cbr, fontsize=12, fontweight='bold')
                
                text_y_non_cbr = non_cbr + 0.002 if non_cbr > 0 else non_cbr - 0.002
                va_non_cbr = 'bottom' if non_cbr > 0 else 'top'
                ax.text(i + width/2, text_y_non_cbr, f"{non_cbr:.4f}", ha='center', va=va_non_cbr, fontsize=12, fontweight='bold')
                
                # Difference annotation
                diff = cbr - non_cbr
                pct_diff = (diff / abs(non_cbr)) * 100 if non_cbr != 0 else 0
                max_val = max(abs(cbr), abs(non_cbr))
                
                ax.text(i, max_val + 0.008, f"Δ: {diff:+.4f}\n({pct_diff:+.1f}%)", 
                       ha='center', va='bottom', fontsize=10, fontweight='bold',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.6))
            
            ax.axhline(0, color='black', linestyle='--', alpha=0.5, linewidth=1)
            ax.set_ylabel('A/B Separation (mean A - mean B PC1)', fontsize=18, fontweight='bold', labelpad=10)
            ax.set_title('A/B Separation: CBR vs. Non-CBR Regions', 
                        fontsize=20, fontweight='bold', pad=15)
            
            # Use display names
            display_labels = [get_display_name(tp) for tp in common_tps]
            ax.set_xticks(x)
            ax.set_xticklabels(display_labels, rotation=30, ha='right', fontsize=14, fontweight='bold')
            ax.legend(fontsize=16)
            ax.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            plt.savefig(f'{output_dir}/ab_separation_cbr_comparison.png', dpi=300, bbox_inches='tight')
            plt.savefig(f'{output_dir}/ab_separation_cbr_comparison.svg', format='svg', bbox_inches='tight')
            plt.close()
    
    # Save summary
    summary_df = pd.DataFrame({
        'Timepoint': list(strength_by_tp.keys()),
        'Stage_Name': [get_display_name(tp) for tp in strength_by_tp.keys()],
        'Compartment_Strength': list(strength_by_tp.values())
    })
    
    if separation_by_tp:
        sep_data = pd.DataFrame({
            'Timepoint': list(separation_by_tp.keys()),
            'AB_Separation': list(separation_by_tp.values())
        })
        summary_df = summary_df.merge(sep_data, on='Timepoint', how='outer')
    
    if strength_cbr_by_tp:
        cbr_strength = pd.DataFrame({
            'Timepoint': list(strength_cbr_by_tp.keys()),
            'CBR_Compartment_Strength': list(strength_cbr_by_tp.values())
        })
        summary_df = summary_df.merge(cbr_strength, on='Timepoint', how='outer')
    
    if strength_non_cbr_by_tp:
        non_cbr_strength = pd.DataFrame({
            'Timepoint': list(strength_non_cbr_by_tp.keys()),
            'NonCBR_Compartment_Strength': list(strength_non_cbr_by_tp.values())
        })
        summary_df = summary_df.merge(non_cbr_strength, on='Timepoint', how='outer')
    
    if separation_cbr_by_tp:
        cbr_sep = pd.DataFrame({
            'Timepoint': list(separation_cbr_by_tp.keys()),
            'CBR_AB_Separation': list(separation_cbr_by_tp.values())
        })
        summary_df = summary_df.merge(cbr_sep, on='Timepoint', how='outer')
    
    if separation_non_cbr_by_tp:
        non_cbr_sep = pd.DataFrame({
            'Timepoint': list(separation_non_cbr_by_tp.keys()),
            'NonCBR_AB_Separation': list(separation_non_cbr_by_tp.values())
        })
        summary_df = summary_df.merge(non_cbr_sep, on='Timepoint', how='outer')
    
    summary_df.to_csv(f'{output_dir}/compartment_strength_summary.csv', index=False)
    print(f"  Saved compartment strength summary to {output_dir}/compartment_strength_summary.csv")
    
    return strength_by_tp, separation_by_tp

def analyze_compartment_switching(eigenvector_df1, eigenvector_df2, timepoint1, timepoint2, pc1_threshold=None):
    """
    Analyze compartment switching between two timepoints
    """
    # Ensure both dataframes have the same number of bins
    min_len = min(len(eigenvector_df1), len(eigenvector_df2))
    df1 = eigenvector_df1.iloc[:min_len].copy()
    df2 = eigenvector_df2.iloc[:min_len].copy()
    
    # Apply PC1 threshold filtering if specified
    if pc1_threshold is not None:
        reliable_mask1 = abs(df1['PC1']) > pc1_threshold
        reliable_mask2 = abs(df2['PC1']) > pc1_threshold
        
        total_bins = len(df1)
        reliable_bins1 = reliable_mask1.sum()
        reliable_bins2 = reliable_mask2.sum()
        
        print(f"  PC1 threshold (|PC1| > {pc1_threshold}):")
        print(f"    {timepoint1}: {reliable_bins1}/{total_bins} bins ({reliable_bins1/total_bins*100:.1f}%)")
        print(f"    {timepoint2}: {reliable_bins2}/{total_bins} bins ({reliable_bins2/total_bins*100:.1f}%)")
        
        reliable_in_both = reliable_mask1 & reliable_mask2
        df1 = df1[reliable_in_both].reset_index(drop=True)
        df2 = df2[reliable_in_both].reset_index(drop=True)
        
        if len(df1) < 100:
            print(f"  WARNING: Only {len(df1)} bins remain after filtering")
    
    # Create switching DataFrame
    switching_df = pd.DataFrame({
        'bin': df1['bin'],
        'chrom': df1['chrom'],
        'start': df1['start'],
        'end': df1['end'],
        'PC1_t1': df1['PC1'],
        'PC1_t2': df2['PC1'],
        'compartment_t1': df1['compartment'],
        'compartment_t2': df2['compartment']
    })
    
    # Calculate switching
    switching_df['switched'] = switching_df['compartment_t1'] != switching_df['compartment_t2']
    switching_df['switch_type'] = 'stable'
    
    mask_a_to_b = (switching_df['compartment_t1'] == 'A') & (switching_df['compartment_t2'] == 'B')
    mask_b_to_a = (switching_df['compartment_t1'] == 'B') & (switching_df['compartment_t2'] == 'A')
    switching_df.loc[mask_a_to_b, 'switch_type'] = 'A to B'
    switching_df.loc[mask_b_to_a, 'switch_type'] = 'B to A'
    
    switching_df['switch_magnitude'] = np.abs(switching_df['PC1_t1'] - switching_df['PC1_t2'])
    
    # Statistics
    total_bins = len(switching_df)
    if total_bins == 0:
        print(f"  ERROR: No bins to analyze")
        return None
    
    switched_bins = switching_df['switched'].sum()
    switch_pct = (switched_bins / total_bins) * 100
    a_to_b = (switching_df['switch_type'] == 'A to B').sum()
    b_to_a = (switching_df['switch_type'] == 'B to A').sum()
    
    print(f"  Switching: {timepoint1} → {timepoint2}")
    print(f"    Total bins: {total_bins}")
    print(f"    Switched: {switched_bins} ({switch_pct:.2f}%)")
    print(f"    A→B: {a_to_b} ({(a_to_b/total_bins)*100:.2f}%)")
    print(f"    B→A: {b_to_a} ({(b_to_a/total_bins)*100:.2f}%)")
    
    return switching_df

def create_switching_plots(switching_df, timepoint1, timepoint2, output_dir="output"):
    """
    Create visualization plots for compartment switching
    CORRECTED: A-to-B as red, B-to-A as blue, stable as grey
    """
    os.makedirs(output_dir, exist_ok=True)
    
    fig = plt.figure(figsize=(18, 6))
    
    # Get display names
    display_name1 = get_display_name(timepoint1)
    display_name2 = get_display_name(timepoint2)
    
    # Plot 1: PC1 scatterplot with CORRECTED colors
    ax1 = plt.subplot(1, 3, 1)
    
    # Separate points by switch type
    stable_mask = switching_df['switch_type'] == 'stable'
    a_to_b_mask = switching_df['switch_type'] == 'A to B'
    b_to_a_mask = switching_df['switch_type'] == 'B to A'
    
    # Plot in order: stable first (grey), then switching (colored on top)
    ax1.scatter(switching_df.loc[stable_mask, 'PC1_t1'], 
               switching_df.loc[stable_mask, 'PC1_t2'], 
               c='grey', alpha=0.3, s=5, label='Stable')
    ax1.scatter(switching_df.loc[a_to_b_mask, 'PC1_t1'], 
               switching_df.loc[a_to_b_mask, 'PC1_t2'], 
               c='red', alpha=0.6, s=5, label='A → B')
    ax1.scatter(switching_df.loc[b_to_a_mask, 'PC1_t1'], 
               switching_df.loc[b_to_a_mask, 'PC1_t2'], 
               c='blue', alpha=0.6, s=5, label='B → A')
    
    ax1.axhline(0, color='black', linestyle='--', alpha=0.5, linewidth=2)
    ax1.axvline(0, color='black', linestyle='--', alpha=0.5, linewidth=2)
    ax1.set_xlabel(f'{display_name1} PC1', fontsize=16, fontweight='bold', labelpad=8)
    ax1.set_ylabel(f'{display_name2} PC1', fontsize=16, fontweight='bold', labelpad=8)
    ax1.set_title('Compartment Score Comparison', fontsize=18, fontweight='bold', pad=10)
    ax1.legend(fontsize=14, loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(labelsize=13)
    
    # Plot 2: Pie chart with CORRECTED colors
    ax2 = plt.subplot(1, 3, 2)
    switch_counts = switching_df['switch_type'].value_counts()
    colors_pie = {'stable': 'grey', 'A to B': 'red', 'B to A': 'blue'}
    pie_colors = [colors_pie.get(label, 'grey') for label in switch_counts.index]
    wedges, texts, autotexts = ax2.pie(switch_counts, labels=switch_counts.index, autopct='%1.1f%%', 
                                        colors=pie_colors, textprops={'fontsize': 14, 'fontweight': 'bold'})
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(14)
    ax2.set_title('Compartment Switching', fontsize=18, fontweight='bold', pad=10)
    
    # Plot 3: Switching magnitude with CORRECTED colors
    ax3 = plt.subplot(1, 3, 3)
    switched_magnitudes = switching_df.loc[switching_df['switched'], 'switch_magnitude']
    
    if len(switched_magnitudes) > 0:
        a_to_b_mask = switching_df['switch_type'] == 'A to B'
        b_to_a_mask = switching_df['switch_type'] == 'B to A'
        
        a_to_b_mag = switching_df.loc[a_to_b_mask, 'switch_magnitude']
        b_to_a_mag = switching_df.loc[b_to_a_mask, 'switch_magnitude']
        
        bins = min(50, max(20, len(switched_magnitudes) // 20))
        
        if len(a_to_b_mag) > 0:
            ax3.hist(a_to_b_mag, bins=bins, color='red', alpha=0.7, label='A → B', edgecolor='black')
        if len(b_to_a_mag) > 0:
            ax3.hist(b_to_a_mag, bins=bins, color='blue', alpha=0.6, label='B → A', edgecolor='black')
        
        ax3.legend(fontsize=14, loc='best')
    else:
        ax3.text(0.5, 0.5, "No switches detected", ha='center', va='center', 
                transform=ax3.transAxes, fontsize=16, fontweight='bold')
    
    ax3.set_xlabel('Switch magnitude (|ΔPC1|)', fontsize=16, fontweight='bold', labelpad=8)
    ax3.set_ylabel('Count', fontsize=16, fontweight='bold', labelpad=8)
    ax3.set_title('Switching Magnitude', fontsize=18, fontweight='bold', pad=10)
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.tick_params(labelsize=13)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/compartment_switching_{timepoint1}_vs_{timepoint2}.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/compartment_switching_{timepoint1}_vs_{timepoint2}.svg', format='svg', bbox_inches='tight')
    plt.close()

def create_sequential_summary_plot(sequential_results, pc1_threshold=None, output_dir="output"):
    """
    Create summary visualization for sequential timepoint comparisons
    CORRECTED: A-to-B as red, B-to-A as blue
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if not sequential_results:
        print("  No sequential results to plot")
        return
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    pair_labels = list(sequential_results.keys())
    a_to_b_pct = []
    b_to_a_pct = []
    stable_pct = []
    switched_pct = []
    
    for pair, df in sequential_results.items():
        total = len(df)
        switched = df['switched'].sum()
        switched_pct.append(switched / total * 100)
        
        a_to_b = (df['switch_type'] == 'A to B').sum() / total * 100
        b_to_a = (df['switch_type'] == 'B to A').sum() / total * 100
        stable = (df['switch_type'] == 'stable').sum() / total * 100
        
        a_to_b_pct.append(a_to_b)
        b_to_a_pct.append(b_to_a)
        stable_pct.append(stable)
    
    x = np.arange(len(pair_labels))
    width = 0.7
    
    # CORRECTED: grey for stable, red for A→B, blue for B→A
    ax.bar(x, stable_pct, width, label='Stable', color='grey', edgecolor='black', linewidth=1.5)
    ax.bar(x, a_to_b_pct, width, bottom=stable_pct, label='A → B', color='red', edgecolor='black', linewidth=1.5)
    ax.bar(x, b_to_a_pct, width, bottom=[a+b for a,b in zip(stable_pct, a_to_b_pct)],
           label='B → A', color='blue', edgecolor='black', linewidth=1.5)
    
    # Create display labels using stage names
    display_labels = []
    for pair in pair_labels:
        tp1, tp2 = pair.split('_to_')
        display_labels.append(f"{get_display_name(tp1)} → {get_display_name(tp2)}")
    
    # Add percentage annotations
    for i, switch in enumerate(switched_pct):
        ax.text(i, 103, f"{switch:.1f}%", ha='center', va='bottom', fontweight='bold', fontsize=14)
    
    ax.set_ylabel('Percentage of bins', fontsize=18, fontweight='bold', labelpad=10)
    title = f"Compartment Switching Across Development"
    if pc1_threshold:
        title += f" (|PC1| > {pc1_threshold})"
    ax.set_title(title, fontsize=20, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(display_labels, rotation=30, ha='right', fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=16)
    ax.grid(True, alpha=0.3, axis='y')
    ax.tick_params(labelsize=14)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/sequential_compartment_switching.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/sequential_compartment_switching.svg', format='svg', bbox_inches='tight')
    plt.close()
    
    # Save CSV with display names
    summary_df = pd.DataFrame({
        'Transition': display_labels,
        'Stable_Pct': stable_pct,
        'A_to_B_Pct': a_to_b_pct,
        'B_to_A_Pct': b_to_a_pct,
        'Total_Switched_Pct': switched_pct
    })
    summary_df.to_csv(f'{output_dir}/sequential_switching_summary.csv', index=False)
    print(f"  Saved switching summary to {output_dir}/sequential_switching_summary.csv")

#################################################
#               MAIN EXECUTION                  #
#################################################
if __name__ == "__main__":
    print("\n" + "="*70)
    print("Enhanced Saddle Plot Analysis for Ascaris suum PDE")
    print("="*70)
    print(f"\nTimepoints: {', '.join([f'{tp} ({get_display_name(tp)})' for tp in timepoints])}")
    print(f"Pre-PDE: {', '.join(pre_pde_timepoints)}")
    print(f"Post-PDE: {', '.join(post_pde_timepoints)}")
    if pc1_threshold:
        print(f"PC1 filtering threshold: |PC1| > {pc1_threshold}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load all data
    print("\n" + "="*70)
    print("LOADING DATA")
    print("="*70)
    
    all_matrix_data = {}
    all_eigen_data = {}
    
    for tp in timepoints:
        print(f"\n--- Loading {tp} ({get_display_name(tp)}) ---")
        
        # Get file paths
        matrix_path, eigen_path = get_file_paths(
            tp, matrix_dir, eigenvector_dir, 
            matrix_pattern, eigenvector_pattern,
            post_pde_matrix_pattern, post_pde_eigenvector_pattern,
            subfolder_pre, subfolder_post,
            eigen_subfolder_pre, eigen_subfolder_post,
            pre_pde_timepoints, post_pde_timepoints
        )
        
        # Load eigenvector
        eigen_df = load_eigenvector(eigen_path)
        if eigen_df is not None:
            all_eigen_data[tp] = eigen_df
        else:
            print(f"  WARNING: Could not load eigenvector for {tp}")
            continue
        
        # Load matrix
        sparse_matrix = load_sparse_matrix(matrix_path)
        if sparse_matrix is not None:
            # Convert to dense
            eigen_df = all_eigen_data[tp]
            use_max_bins = min(max_bins, len(eigen_df)) if max_bins else len(eigen_df)
            
            dense_matrix = sparse_to_dense(sparse_matrix, max_bin=use_max_bins)
            if dense_matrix is not None:
                all_matrix_data[tp] = dense_matrix
            else:
                print(f"  WARNING: Could not convert to dense matrix for {tp}")
        else:
            print(f"  WARNING: Could not load matrix for {tp}")
    
    print(f"\n✓ Successfully loaded data for {len(all_matrix_data)} timepoints")
    
    # Generate saddle plots
    print("\n" + "="*70)
    print("STEP 1: Generating Saddle Plots and Computing Strength")
    print("="*70)
    
    saddle_results = {}
    
    # First pass: compute all saddle plots to find global min/max if needed
    if GLOBAL_SCALE:
        print("\nComputing global scale for all saddle plots...")
        all_saddle_matrices = []
        for tp in timepoints:
            if tp in all_matrix_data and tp in all_eigen_data:
                dense_matrix = all_matrix_data[tp]
                eigen_df = all_eigen_data[tp]
                eigenvector = eigen_df['PC1'].values[:len(dense_matrix)]
                
                saddle_matrix, _, _, _, _, _ = create_saddle_plot(dense_matrix, eigenvector)
                all_saddle_matrices.append(saddle_matrix)
        
        if all_saddle_matrices:
            # Find global min and max across all matrices (excluding zeros/nan)
            valid_values = []
            for sm in all_saddle_matrices:
                valid = sm[sm > 0]  # Only positive values for log scale
                if len(valid) > 0:
                    valid_values.extend(valid.flatten())
            
            if valid_values:
                global_vmin = np.min(valid_values)
                global_vmax = np.max(valid_values)
                print(f"  Global scale range: {global_vmin:.6f} to {global_vmax:.6f}")
            else:
                global_vmin = None
                global_vmax = None
                print("  WARNING: Could not determine global scale")
        else:
            global_vmin = None
            global_vmax = None
    else:
        global_vmin = None
        global_vmax = None
    
    # Second pass: plot with appropriate scaling
    for tp in timepoints:
        if tp not in all_matrix_data or tp not in all_eigen_data:
            print(f"\nSkipping {tp} ({get_display_name(tp)}): data not available")
            continue
        
        print(f"\n--- Processing {tp} ({get_display_name(tp)}) ---")
        
        dense_matrix = all_matrix_data[tp]
        eigen_df = all_eigen_data[tp]
        
        # Ensure matrix and eigenvector dimensions match
        eigenvector = eigen_df['PC1'].values[:len(dense_matrix)]
        
        # Create saddle plot
        saddle_matrix, bin_edges, saddle_strength, aa_strength, bb_strength, ab_strength = \
            create_saddle_plot(dense_matrix, eigenvector)
        
        # Save results
        saddle_results[tp] = {
            'saddle_matrix': saddle_matrix,
            'bin_edges': bin_edges,
            'saddle_strength': saddle_strength,
            'aa_strength': aa_strength,
            'bb_strength': bb_strength,
            'ab_strength': ab_strength
        }
        
        # Determine subfolder
        subfolder = "postpde" if tp in post_pde_timepoints else "prepde"
        
        # Plot and save with global or local scaling
        plot_saddle_plot(
            saddle_matrix, bin_edges, tp, 
            output_dir=output_dir, 
            subfolder=subfolder,
            vmin=global_vmin, 
            vmax=global_vmax,
            boundary_linewidth=BOUNDARY_LINEWIDTH,
            boundary_color=BOUNDARY_COLOR
        )
        
        print(f"  Saddle strength: {saddle_strength:.4f}")
        print(f"  AA strength: {aa_strength:.4f}")
        print(f"  BB strength: {bb_strength:.4f}")
        print(f"  AB strength: {ab_strength:.4f}")
        print(f"  ✓ Saved saddle plot for {tp}")
    
    # Save saddle strength summary
    if saddle_results:
        summary_df = pd.DataFrame({
            'Timepoint': list(saddle_results.keys()),
            'Stage_Name': [get_display_name(tp) for tp in saddle_results.keys()],
            'Saddle_Strength': [saddle_results[tp]['saddle_strength'] for tp in saddle_results.keys()],
            'AA_Strength': [saddle_results[tp]['aa_strength'] for tp in saddle_results.keys()],
            'BB_Strength': [saddle_results[tp]['bb_strength'] for tp in saddle_results.keys()],
            'AB_Strength': [saddle_results[tp]['ab_strength'] for tp in saddle_results.keys()]
        })
        summary_df.to_csv(f'{output_dir}/saddle_strength_summary.csv', index=False)
        print(f"\n✓ Saved saddle strength summary")
        
        # Plot saddle strength across development
        valid_tps = [tp for tp in timepoints if tp in saddle_results and 
                    not np.isnan(saddle_results[tp]['saddle_strength'])]
        
        if valid_tps:
            strengths = [saddle_results[tp]['saddle_strength'] for tp in valid_tps]
            display_labels = [get_display_name(tp) for tp in valid_tps]
            
            fig, ax = plt.subplots(figsize=(12, 7))
            x_pos = range(len(valid_tps))
            
            ax.plot(x_pos, strengths, 'o-', color='darkblue', linewidth=3, markersize=12)
            
            # Add values as text
            for i, (strength, label) in enumerate(zip(strengths, display_labels)):
                ax.text(i, strength + 0.05, f"{strength:.3f}", ha='center', va='bottom', 
                       fontweight='bold', fontsize=14)
            
            ax.set_ylabel('Saddle Strength', fontsize=20, fontweight='bold', labelpad=12)
            ax.set_xlabel('Developmental Stage', fontsize=20, fontweight='bold', labelpad=12)
            ax.set_title('Compartmentalization Strength Across Development\n(Gold Standard: (AA+BB)/(AB+BA))', 
                        fontsize=22, fontweight='bold', pad=15)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(display_labels, rotation=30, ha='right', fontsize=16, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=16)
            
            plt.tight_layout()
            plt.savefig(f'{output_dir}/saddle_strength_across_development.png', dpi=300, bbox_inches='tight')
            plt.savefig(f'{output_dir}/saddle_strength_across_development.svg', format='svg', bbox_inches='tight')
            plt.close()
            
            print(f"  Saved saddle strength plot")
        
        # Plot saddle strength decomposition
        valid_tps_comp = [tp for tp in timepoints if tp in saddle_results and 
                         not np.isnan(saddle_results[tp]['aa_strength']) and
                         not np.isnan(saddle_results[tp]['bb_strength']) and
                         not np.isnan(saddle_results[tp]['ab_strength'])]
        
        if valid_tps_comp:
            aa_strengths = [saddle_results[tp]['aa_strength'] for tp in valid_tps_comp]
            bb_strengths = [saddle_results[tp]['bb_strength'] for tp in valid_tps_comp]
            ab_strengths = [saddle_results[tp]['ab_strength'] for tp in valid_tps_comp]
            overall_strengths = [saddle_results[tp]['saddle_strength'] for tp in valid_tps_comp]
            
            display_labels = [get_display_name(tp) for tp in valid_tps_comp]
            x_pos = range(len(valid_tps_comp))
            
            # Create 2x2 subplot
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('Saddle Strength Decomposition Across Development', 
                        fontsize=24, fontweight='bold', y=0.995)
            
            # Plot 1: AA Strength (A-to-A interactions)
            ax1 = axes[0, 0]
            ax1.plot(x_pos, aa_strengths, 'o-', color='darkorange', linewidth=3, markersize=10)
            for i, val in enumerate(aa_strengths):
                ax1.text(i, val + 0.02, f"{val:.3f}", ha='center', va='bottom', 
                        fontweight='bold', fontsize=14)
            ax1.set_ylabel('AA Strength', fontsize=18, fontweight='bold', labelpad=10)
            ax1.set_title('A-to-A Interactions (Euchromatin)', fontsize=20, fontweight='bold', pad=10)
            ax1.set_xticks(x_pos)
            ax1.set_xticklabels(display_labels, rotation=30, ha='right', fontsize=14, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            ax1.tick_params(labelsize=14)
            
            # Plot 2: BB Strength (B-to-B interactions)
            ax2 = axes[0, 1]
            ax2.plot(x_pos, bb_strengths, 'o-', color='darkgreen', linewidth=3, markersize=10)
            for i, val in enumerate(bb_strengths):
                ax2.text(i, val + 0.02, f"{val:.3f}", ha='center', va='bottom', 
                        fontweight='bold', fontsize=14)
            ax2.set_ylabel('BB Strength', fontsize=18, fontweight='bold', labelpad=10)
            ax2.set_title('B-to-B Interactions (Heterochromatin)', fontsize=20, fontweight='bold', pad=10)
            ax2.set_xticks(x_pos)
            ax2.set_xticklabels(display_labels, rotation=30, ha='right', fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            ax2.tick_params(labelsize=14)
            
            # Plot 3: AB Strength (between compartments)
            ax3 = axes[1, 0]
            ax3.plot(x_pos, ab_strengths, 'o-', color='darkred', linewidth=3, markersize=10)
            for i, val in enumerate(ab_strengths):
                ax3.text(i, val + 0.02, f"{val:.3f}", ha='center', va='bottom', 
                        fontweight='bold', fontsize=14)
            ax3.set_ylabel('AB Strength', fontsize=18, fontweight='bold', labelpad=10)
            ax3.set_title('A-to-B Interactions (Between Compartments)', fontsize=20, fontweight='bold', pad=10)
            ax3.set_xticks(x_pos)
            ax3.set_xticklabels(display_labels, rotation=30, ha='right', fontsize=14, fontweight='bold')
            ax3.grid(True, alpha=0.3)
            ax3.tick_params(labelsize=14)
            
            # Plot 4: Overall Saddle Strength
            ax4 = axes[1, 1]
            ax4.plot(x_pos, overall_strengths, 'o-', color='darkblue', linewidth=3, markersize=10)
            for i, val in enumerate(overall_strengths):
                ax4.text(i, val + 0.05, f"{val:.3f}", ha='center', va='bottom', 
                        fontweight='bold', fontsize=14)
            ax4.set_ylabel('Saddle Strength', fontsize=18, fontweight='bold', labelpad=10)
            ax4.set_title('Overall: (AA+BB)/(AB+BA)', fontsize=20, fontweight='bold', pad=10)
            ax4.set_xticks(x_pos)
            ax4.set_xticklabels(display_labels, rotation=30, ha='right', fontsize=14, fontweight='bold')
            ax4.grid(True, alpha=0.3)
            ax4.tick_params(labelsize=14)
            
            plt.tight_layout()
            plt.savefig(f'{output_dir}/saddle_strength_decomposition.png', dpi=300, bbox_inches='tight')
            plt.savefig(f'{output_dir}/saddle_strength_decomposition.svg', format='svg', bbox_inches='tight')
            plt.close()
            
            # Save detailed CSV
            detailed_df = pd.DataFrame({
                'Timepoint': valid_tps_comp,
                'Stage_Name': display_labels,
                'AA_Strength': aa_strengths,
                'BB_Strength': bb_strengths,
                'AB_Strength': ab_strengths,
                'Overall_Saddle_Strength': overall_strengths
            })
            detailed_df.to_csv(f'{output_dir}/saddle_strength_detailed.csv', index=False)
            
            print(f"  Saved saddle strength decomposition plot and detailed CSV")
        else:
            print("  No valid component data for decomposition plot")
    
    print(f"\n✓ Completed saddle strength analysis")
    
    # Exit early if PLOT_ONLY is True
    if PLOT_ONLY:
        print("\n" + "="*70)
        print("PLOT_ONLY MODE: Skipping additional analyses")
        print("="*70)
        print(f"\nResults saved to '{output_dir}/' directory:")
        print("  - Saddle plots (PNG & SVG)")
        print("  - Saddle strength analysis")
        print("  - Saddle strength decomposition")
        print("\n" + "="*70)
        sys.exit(0)
    
    # Continue with full analysis if PLOT_ONLY is False
    
    # 2. Sequential compartment switching
    print("\n" + "="*70)
    print("STEP 2: Sequential Compartment Switching Analysis")
    print("="*70)
    
    sequential_results = {}
    target_transitions = [
        ("0hr", "48hr"),
        ("48hr", "60hr"),
        ("60hr", "5day"),
        ("5day", "10day")
    ]
    
    for tp1, tp2 in target_transitions:
        if tp1 not in all_eigen_data or tp2 not in all_eigen_data:
            print(f"\nSkipping {tp1} ({get_display_name(tp1)}) → {tp2} ({get_display_name(tp2)}): data not available")
            continue
        
        print(f"\n--- Analyzing {tp1} ({get_display_name(tp1)}) → {tp2} ({get_display_name(tp2)}) ---")
        switching_df = analyze_compartment_switching(
            all_eigen_data[tp1], all_eigen_data[tp2], tp1, tp2, pc1_threshold
        )
        
        if switching_df is not None:
            sequential_results[f"{tp1}_to_{tp2}"] = switching_df
            create_switching_plots(switching_df, tp1, tp2, output_dir=output_dir)
            print(f"✓ Completed {tp1} → {tp2}")
    
    if sequential_results:
        create_sequential_summary_plot(sequential_results, pc1_threshold, output_dir=output_dir)
        print(f"\n✓ Completed sequential switching analysis")
    
    # 3. CBR analysis (if available)
    cbr_df = None
    if cbr_bed_file and os.path.exists(cbr_bed_file):
        print("\n" + "="*70)
        print("STEP 3: CBR Analysis")
        print("="*70)
        
        cbr_df = pd.read_csv(cbr_bed_file, sep='\t', header=None)
        if cbr_df.shape[1] >= 3:
            cbr_df.columns = ['chrom', 'start', 'end'] + list(cbr_df.columns[3:])
            print(f"Loaded {len(cbr_df)} CBR regions")
        else:
            print("WARNING: CBR file has < 3 columns")
            cbr_df = None
    
    # 4. Boundary analysis
    if all_eigen_data:
        print("\n" + "="*70)
        print("STEP 4: Compartment Boundary Analysis")
        print("="*70)
        analyze_compartment_boundaries(all_eigen_data, timepoints, cbr_df=cbr_df, output_dir=output_dir)
        print("✓ Completed boundary analysis")
    
    # 5. Entropy analysis
    if all_eigen_data:
        print("\n" + "="*70)
        print("STEP 5: Entropy Analysis")
        print("="*70)
        analyze_compartment_entropy(all_eigen_data, timepoints, cbr_df=cbr_df, output_dir=output_dir)
        print("✓ Completed entropy analysis")
    
    # 6. Compartment Strength and A/B Separation
    if all_eigen_data:
        print("\n" + "="*70)
        print("STEP 6: Compartment Strength and A/B Separation Analysis")
        print("="*70)
        analyze_compartment_strength(all_eigen_data, timepoints, cbr_df=cbr_df, output_dir=output_dir)
        print("✓ Completed compartment strength analysis")
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print(f"\nResults saved to '{output_dir}/' directory:")
    print("  - Saddle plots (PNG & SVG)")
    print("  - Saddle strength analysis (gold standard metric)")
    print("  - Saddle strength decomposition (AA, BB, AB components)")
    print("  - Switching analysis plots")
    print("  - Boundary strength analysis")
    print("  - Entropy analysis")
    print("  - Compartment strength analysis (mean |PC1|)")
    print("  - A/B separation analysis")
    print("  - Summary CSV files")
    print("\n" + "="*70)
