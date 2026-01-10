#!/usr/bin/env python3


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# IEEE PAPER CONFIGURATION
# ============================================================================

IEEE_CONFIG = {
    # IEEE two-column format specifications
    'column_width_inches': 3.5,      # Width of single column in inches
    'figure_height_inches': 2.5,     # Height for single plot
    
    # Font sizes following IEEE guidelines
    'font_size_title': 9,
    'font_size_labels': 8,
    'font_size_ticks': 7,
    'font_size_legend': 7,
    
    # DPI for publication quality
    'dpi': 300,
    
    # Plot style
    'grid_alpha': 0.3,
    'grid_linestyle': '--',
    'marker_size': 20,
    'marker_alpha': 0.6,
    'marker_edge_width': 0.3,
    'trend_line_width': 1.5,
    'trend_line_style': '--',
    'trend_line_alpha': 0.8,
    
    # Colors (colorblind-friendly)
    'scatter_color': '#2E86AB',      # Blue
    'trend_color': '#A23B72',        # Magenta
    'edge_color': '#000000',         # Black
    
    # Margins and spacing
    'tight_layout': True,
    'pad_inches': 0.05,
}

# ============================================================================
# INPUT FILES CONFIGURATION
# ============================================================================

INPUT_CONFIG = {
    # Latency file (contains multiple application latencies)
    'latency_file': 'client_latency.csv',
    
    # Kernel metric file (single metric extracted)
    'kernel_metric_file': 'dstate_io_count.csv',
    
    # Column names in latency file (will create one plot per latency column)
    'latency_columns': [
        'Social_P99_ms',
        'Hotel_P99_ms', 
        'Feedback_P99_ms'
    ],
    
    # Timestamp column names
    'latency_timestamp_col': 'Time',
    'kernel_timestamp_col': 'timestamp',
    
    # Kernel metric column name (auto-detected if None)
    'kernel_metric_col': None,  # Will use first non-timestamp column
    
    # Time alignment tolerance (seconds)
    'time_tolerance_seconds': 2,
}

# ============================================================================
# PLOT CONFIGURATION
# ============================================================================

PLOT_CONFIG = {
    # Plot 1: Social Network P99 Latency
    'Social_P99_ms': {
        'x_label': 'D-State I/O Count',
        'y_label': 'Social Network P99 Latency (ms)',
        'title': 'Social Network: D-State I/O vs Latency',
        'output_filename': 'social_dstate_correlation.pdf',
        'show_correlation_text': True,
        'correlation_position': 'upper left',  # 'upper left', 'upper right', 'lower left', 'lower right'
    },
    
    # Plot 2: Hotel Reservation P99 Latency  
    'Hotel_P99_ms': {
        'x_label': 'D-State I/O Count',
        'y_label': 'Hotel Reservation P99 Latency (ms)',
        'title': 'Hotel Reservation: D-State I/O vs Latency',
        'output_filename': 'hotel_dstate_correlation.pdf',
        'show_correlation_text': True,
        'correlation_position': 'upper left',
    },
    
    # Plot 3: Feedback Service P99 Latency
    'Feedback_P99_ms': {
        'x_label': 'D-State I/O Count',
        'y_label': 'Feedback Service P99 Latency (ms)',
        'title': 'Feedback Service: D-State I/O vs Latency',
        'output_filename': 'feedback_dstate_correlation.pdf',
        'show_correlation_text': True,
        'correlation_position': 'upper left',
    },
}

# Output directory for plots
OUTPUT_DIR = './'

# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def set_ieee_style():
    """Configure matplotlib for IEEE publication style."""
    plt.rcParams.update({
        # Font settings
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
        'font.size': IEEE_CONFIG['font_size_labels'],
        
        # Figure settings
        'figure.dpi': IEEE_CONFIG['dpi'],
        'savefig.dpi': IEEE_CONFIG['dpi'],
        'savefig.bbox': 'tight',
        'savefig.pad_inches': IEEE_CONFIG['pad_inches'],
        
        # Line and marker settings
        'lines.linewidth': IEEE_CONFIG['trend_line_width'],
        'lines.markersize': np.sqrt(IEEE_CONFIG['marker_size']),
        
        # Axes settings
        'axes.linewidth': 0.8,
        'axes.labelsize': IEEE_CONFIG['font_size_labels'],
        'axes.titlesize': IEEE_CONFIG['font_size_title'],
        'axes.grid': True,
        'axes.axisbelow': True,
        
        # Tick settings
        'xtick.labelsize': IEEE_CONFIG['font_size_ticks'],
        'ytick.labelsize': IEEE_CONFIG['font_size_ticks'],
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        
        # Legend settings
        'legend.fontsize': IEEE_CONFIG['font_size_legend'],
        'legend.frameon': True,
        'legend.framealpha': 0.9,
        'legend.edgecolor': 'black',
        'legend.fancybox': False,
        
        # Grid settings
        'grid.alpha': IEEE_CONFIG['grid_alpha'],
        'grid.linestyle': IEEE_CONFIG['grid_linestyle'],
        'grid.linewidth': 0.5,
    })


def load_and_align_data(latency_file, kernel_file, latency_ts_col, kernel_ts_col, 
                        kernel_metric_col=None, tolerance_seconds=2):
    """
    Load and align latency and kernel metric data based on timestamps.
    
    Returns:
        DataFrame with aligned data
    """
    # Load latency data
    latency_df = pd.read_csv(latency_file)
    latency_df[latency_ts_col] = pd.to_datetime(latency_df[latency_ts_col])
    
    # Load kernel metric data
    kernel_df = pd.read_csv(kernel_file)
    kernel_df[kernel_ts_col] = pd.to_datetime(kernel_df[kernel_ts_col])
    
    # Auto-detect kernel metric column if not specified
    if kernel_metric_col is None:
        kernel_metric_col = [col for col in kernel_df.columns if col != kernel_ts_col][0]
    
    # Rename for merging
    latency_df = latency_df.rename(columns={latency_ts_col: 'timestamp'})
    kernel_df = kernel_df.rename(columns={kernel_ts_col: 'timestamp'})
    
    # Merge on nearest timestamp
    aligned = pd.merge_asof(
        latency_df.sort_values('timestamp'),
        kernel_df.sort_values('timestamp'),
        on='timestamp',
        direction='nearest',
        tolerance=pd.Timedelta(seconds=tolerance_seconds)
    )
    
    # Remove rows with missing data
    aligned = aligned.dropna()
    
    return aligned, kernel_metric_col


def calculate_spearman(x, y):
    
    correlation, p_value = spearmanr(x, y)
    return correlation, p_value


def create_correlation_plot(x_data, y_data, plot_config, ieee_config, output_path):
    
    # Calculate correlation
    corr, p_value = calculate_spearman(x_data, y_data)
    
    # Create figure with IEEE column width
    fig, ax = plt.subplots(
        figsize=(ieee_config['column_width_inches'], 
                ieee_config['figure_height_inches'])
    )
    
    # Scatter plot
    ax.scatter(
        x_data, 
        y_data,
        s=ieee_config['marker_size'],
        alpha=ieee_config['marker_alpha'],
        color=ieee_config['scatter_color'],
        edgecolors=ieee_config['edge_color'],
        linewidth=ieee_config['marker_edge_width'],
        zorder=3
    )
    
    # Add trend line (linear regression for visualization)
    z = np.polyfit(x_data, y_data, 1)
    p = np.poly1d(z)
    x_trend = np.linspace(x_data.min(), x_data.max(), 100)
    ax.plot(
        x_trend, 
        p(x_trend),
        color=ieee_config['trend_color'],
        linestyle=ieee_config['trend_line_style'],
        linewidth=ieee_config['trend_line_width'],
        alpha=ieee_config['trend_line_alpha'],
        label='Linear fit',
        zorder=2
    )
    
    # Labels and title
    ax.set_xlabel(plot_config['x_label'], fontsize=ieee_config['font_size_labels'])
    ax.set_ylabel(plot_config['y_label'], fontsize=ieee_config['font_size_labels'])
    # ax.set_title(plot_config['title'], fontsize=ieee_config['font_size_title'], 
    #             fontweight='bold', pad=8)
    
    # Grid
    ax.grid(True, alpha=ieee_config['grid_alpha'], 
           linestyle=ieee_config['grid_linestyle'], zorder=1)
    
    # Add correlation text if requested
    if plot_config.get('show_correlation_text', True):
        # Format p-value
        if p_value < 0.001:
            p_text = "p < 0.001"
        elif p_value < 0.01:
            p_text = f"p = {p_value:.3f}"
        else:
            p_text = f"p = {p_value:.2f}"
        
        # Create text
        corr_text = f"$\\rho$ = {corr:.3f}\n{p_text}\nn = {len(x_data)}"
        
        # Position text
        position = plot_config.get('correlation_position', 'upper left')
        if position == 'upper left':
            x_pos, y_pos = 0.05, 0.95
            va = 'top'
            ha = 'left'
        elif position == 'upper right':
            x_pos, y_pos = 0.95, 0.95
            va = 'top'
            ha = 'right'
        elif position == 'lower left':
            x_pos, y_pos = 0.05, 0.05
            va = 'bottom'
            ha = 'left'
        else:  # lower right
            x_pos, y_pos = 0.95, 0.05
            va = 'bottom'
            ha = 'right'
        
        ax.text(
            x_pos, y_pos, corr_text,
            transform=ax.transAxes,
            fontsize=ieee_config['font_size_legend'],
            verticalalignment=va,
            horizontalalignment=ha,
            bbox=dict(boxstyle='round', facecolor='white', 
                     alpha=0.9, edgecolor='black', linewidth=0.5)
        )
    
    # Tight layout
    if ieee_config['tight_layout']:
        plt.tight_layout()
    
    # Save plot
    plt.savefig(output_path, dpi=ieee_config['dpi'], bbox_inches='tight', 
               pad_inches=ieee_config['pad_inches'])
    plt.close()
    
    return corr, p_value


def generate_all_plots():
    """
    Main function to generate all correlation plots.
    """
    print("=" * 70)
    print("IEEE CORRELATION PLOT GENERATOR")
    print("=" * 70)
    
    # Set IEEE style
    set_ieee_style()
    print("\n✓ IEEE style configured")
    
    # Create output directory
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(exist_ok=True)
    print(f"✓ Output directory: {output_dir}")
    
    # Load and align data
    print("\nLoading and aligning data...")
    aligned_df, kernel_metric_col = load_and_align_data(
        INPUT_CONFIG['latency_file'],
        INPUT_CONFIG['kernel_metric_file'],
        INPUT_CONFIG['latency_timestamp_col'],
        INPUT_CONFIG['kernel_timestamp_col'],
        INPUT_CONFIG['kernel_metric_col'],
        INPUT_CONFIG['time_tolerance_seconds']
    )
    
    print(f"✓ Loaded {len(aligned_df)} aligned data points")
    print(f"✓ Kernel metric: {kernel_metric_col}")
    
    # Generate plots for each latency column
    print("\nGenerating plots...")
    print("-" * 70)
    
    results = []
    
    for latency_col in INPUT_CONFIG['latency_columns']:
        if latency_col not in aligned_df.columns:
            print(f"⚠ Warning: {latency_col} not found in data, skipping...")
            continue
        
        if latency_col not in PLOT_CONFIG:
            print(f"⚠ Warning: No plot configuration for {latency_col}, skipping...")
            continue
        
        plot_cfg = PLOT_CONFIG[latency_col]
        output_path = output_dir / plot_cfg['output_filename']
        
        # Create plot
        corr, p_value = create_correlation_plot(
            aligned_df[kernel_metric_col],
            aligned_df[latency_col],
            plot_cfg,
            IEEE_CONFIG,
            output_path
        )
        
        results.append({
            'Application': latency_col.replace('_P99_ms', ''),
            'Correlation': corr,
            'P-value': p_value,
            'N': len(aligned_df),
            'Output': plot_cfg['output_filename']
        })
        
        # Print results
        print(f"\n{latency_col}:")
        print(f"  Spearman ρ: {corr:+.4f}")
        print(f"  P-value: {p_value:.2e}")
        print(f"  Samples: {len(aligned_df)}")
        print(f"  ✓ Saved: {output_path}")
    
    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    results_df = pd.DataFrame(results)
    print("\n" + results_df.to_string(index=False))
    
    # Save summary
    summary_path = output_dir / 'correlation_summary.csv'
    results_df.to_csv(summary_path, index=False)
    print(f"\n✓ Summary saved: {summary_path}")
    
    print("\n" + "=" * 70)
    print("ALL PLOTS GENERATED SUCCESSFULLY")
    print("=" * 70)
    print(f"\nPlots saved in: {output_dir}")
    print("\nReady for inclusion in IEEE two-column format paper!")


if __name__ == "__main__":
    generate_all_plots()