"""Visualization for spatial transcriptomics multiscale analysis.

Generates publication-quality figures for:
- Tissue maps with cells colored by type
- Per-type VMR fingerprints with null envelopes
- Cross-type co-occurrence heatmaps and curves
- Regional comparison overlays
- Universality figure (SMLM + dendrites + tissue)
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Publication style defaults
STYLE = {
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 8,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'font.family': 'sans-serif',
}


def apply_style():
    """Apply publication style to matplotlib."""
    plt.rcParams.update(STYLE)


def get_type_colors(types, cmap='tab20'):
    """Assign distinct colors to cell types.

    Parameters
    ----------
    types : list of str
    cmap : str
        Colormap name.

    Returns
    -------
    dict of str -> color
    """
    cm = plt.get_cmap(cmap)
    colors = {}
    for i, t in enumerate(types):
        colors[t] = cm(i % cm.N / cm.N)
    return colors


# ── Figure 1: Tissue Maps ──────────────────────────────────────────────


def plot_tissue_maps(
    region_data,
    type_level='class',
    max_types=20,
    point_size=0.3,
    figsize=(18, 5),
    output_path=None,
):
    """Spatial maps of cells colored by type for multiple regions.

    Parameters
    ----------
    region_data : dict of str -> (np.ndarray, np.ndarray)
        Region name -> (positions (N,2), labels (N,)).
    type_level : str
        For figure title.
    max_types : int
        Max types to show in legend. Others grouped as 'Other'.
    point_size : float
    figsize : tuple
    output_path : str or Path, optional
    """
    apply_style()
    n_regions = len(region_data)
    fig, axes = plt.subplots(1, n_regions, figsize=figsize)
    if n_regions == 1:
        axes = [axes]

    # Collect all types across regions for consistent coloring
    all_types = set()
    for pos, lab in region_data.values():
        all_types.update(np.unique(lab))
    all_types = sorted(all_types)

    # Keep top types by frequency, group rest as 'Other'
    type_counts = {}
    for pos, lab in region_data.values():
        for t in lab:
            type_counts[t] = type_counts.get(t, 0) + 1
    top_types = sorted(type_counts, key=type_counts.get, reverse=True)[:max_types]
    colors = get_type_colors(top_types)
    colors['Other'] = (0.8, 0.8, 0.8, 0.5)

    for ax, (region_name, (positions, labels)) in zip(axes, region_data.items()):
        # Map labels to colors
        plot_labels = np.array([l if l in top_types else 'Other' for l in labels])

        # Plot 'Other' first (background)
        other_mask = plot_labels == 'Other'
        if other_mask.any():
            ax.scatter(positions[other_mask, 0], positions[other_mask, 1],
                       c=[colors['Other']], s=point_size, alpha=0.3,
                       rasterized=True)

        # Plot each type
        for t in top_types:
            mask = plot_labels == t
            if mask.any():
                ax.scatter(positions[mask, 0], positions[mask, 1],
                           c=[colors[t]], s=point_size, alpha=0.6,
                           label=t, rasterized=True)

        ax.set_title(f'{region_name}\n({len(positions):,} cells)')
        ax.set_xlabel('x (μm)')
        ax.set_ylabel('y (μm)')
        ax.set_aspect('equal')
        ax.tick_params(axis='both', which='both', length=0)

    # Shared legend
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc='center left',
               bbox_to_anchor=(1.0, 0.5), fontsize=7, markerscale=5)

    fig.suptitle(f'Cell Type Organization ({type_level} level)', y=1.02)
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, bbox_inches='tight')
        logger.info(f'Saved {output_path}')
    return fig


# ── Figure 2: Per-Type Fingerprints ────────────────────────────────────


def plot_per_type_fingerprints(
    fingerprints,
    max_types=12,
    ncols=4,
    figsize=None,
    output_path=None,
):
    """VMR curves per cell type with null envelopes.

    Parameters
    ----------
    fingerprints : dict of str -> TypeFingerprint
    max_types : int
        Max types to plot (sorted by significance).
    ncols : int
    figsize : tuple, optional
    output_path : str or Path, optional
    """
    apply_style()

    # Sort by effect size (integrated excess), take top
    sorted_fps = sorted(fingerprints.values(),
                        key=lambda fp: fp.integrated_excess, reverse=True)
    to_plot = sorted_fps[:max_types]

    nrows = (len(to_plot) + ncols - 1) // ncols
    if figsize is None:
        figsize = (4 * ncols, 3.5 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.atleast_2d(axes)

    for idx, fp in enumerate(to_plot):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]

        scales = np.array(fp.scales_um)
        vmr = np.array(fp.vmr_curve)
        null_mean = np.array(fp.mock_vmr_mean)
        null_std = np.array(fp.mock_vmr_std)

        # Normalize by null mean for interpretable comparison
        safe_null = np.where(null_mean > 0, null_mean, 1.0)
        vmr_ratio = vmr / safe_null
        null_upper = (null_mean + 2 * null_std) / safe_null
        null_lower = np.maximum(0, (null_mean - 2 * null_std)) / safe_null

        # Null envelope (2σ)
        ax.fill_between(scales, null_lower, null_upper,
                        alpha=0.25, color='gray', label='Null ±2σ')
        ax.axhline(1.0, color='gray', linewidth=0.8, linestyle='--',
                   label='Null mean')

        # Real VMR ratio curve
        color = 'tab:red' if fp.is_significant else 'tab:blue'
        ax.plot(scales, vmr_ratio, '-o', color=color, markersize=3,
                linewidth=1.5, label='Observed')

        # Mark peak scale
        peak_idx = int(np.argmax(vmr_ratio - 1.0))
        ax.axvline(scales[peak_idx], color=color, alpha=0.3, linestyle=':')

        ratio_str = f'peak={fp.peak_vmr_ratio:.1f}x'
        ax.set_title(f'{fp.cell_type}\n(n={fp.n_cells:,}, {ratio_str})',
                     fontsize=9)
        ax.set_xscale('log')
        ax.set_xlabel('Scale (μm)')
        ax.set_ylabel('VMR / null')

        if idx == 0:
            ax.legend(fontsize=6, loc='upper left')

    # Hide unused axes
    for idx in range(len(to_plot), nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].set_visible(False)

    fig.suptitle('Per-Type Multiscale Fingerprints', y=1.01)
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, bbox_inches='tight')
        logger.info(f'Saved {output_path}')
    return fig


# ── Figure 3: Cross-Type Heatmap ──────────────────────────────────────


def plot_cross_type_heatmap(
    corr_matrix,
    figsize=(10, 8),
    output_path=None,
):
    """Heatmap of co-occurrence correlation at peak scale.

    Parameters
    ----------
    corr_matrix : pd.DataFrame
        Symmetric correlation matrix from cooccurrence_to_matrix().
    figsize : tuple
    output_path : str or Path, optional
    """
    apply_style()
    fig, ax = plt.subplots(figsize=figsize)

    vmax = max(abs(corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].min()),
               abs(corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].max()))
    vmax = max(vmax, 0.1)

    im = ax.imshow(corr_matrix.values, cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                   aspect='equal')
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Pearson correlation (peak scale)')

    ax.set_xticks(range(len(corr_matrix.columns)))
    ax.set_xticklabels(corr_matrix.columns, rotation=45, ha='right', fontsize=7)
    ax.set_yticks(range(len(corr_matrix.index)))
    ax.set_yticklabels(corr_matrix.index, fontsize=7)

    ax.set_title('Cross-Type Co-occurrence (Peak Scale)')
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, bbox_inches='tight')
        logger.info(f'Saved {output_path}')
    return fig


# ── Figure 4: Co-occurrence Curves ────────────────────────────────────


def plot_cooccurrence_curves(
    results,
    pairs=None,
    max_pairs=9,
    ncols=3,
    figsize=None,
    output_path=None,
):
    """Co-occurrence curves for selected pairs across scales.

    Parameters
    ----------
    results : dict of (str, str) -> CoOccurrenceResult
    pairs : list of (str, str), optional
        Specific pairs to plot. Default: top by |peak_correlation|.
    max_pairs : int
    ncols : int
    figsize : tuple, optional
    output_path : str or Path, optional
    """
    apply_style()

    if pairs is None:
        sorted_results = sorted(results.values(),
                                key=lambda r: abs(r.peak_correlation),
                                reverse=True)
        to_plot = sorted_results[:max_pairs]
    else:
        to_plot = [results[p] for p in pairs if p in results]

    nrows = (len(to_plot) + ncols - 1) // ncols
    if figsize is None:
        figsize = (4 * ncols, 3 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.atleast_2d(axes)

    for idx, res in enumerate(to_plot):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]

        scales = np.array(res.scales_um)
        corr = np.array(res.correlation_curve)
        null_mean = np.array(res.null_corr_mean)
        null_std = np.array(res.null_corr_std)

        ax.fill_between(scales, null_mean - 2 * null_std,
                        null_mean + 2 * null_std,
                        alpha=0.2, color='gray')
        ax.plot(scales, null_mean, '--', color='gray', linewidth=0.8)
        ax.plot(scales, corr, '-o', color='tab:purple', markersize=3,
                linewidth=1.5)

        # Highlight significant scales
        sig = np.array(res.significant_scales)
        if sig.any():
            ax.scatter(scales[sig], corr[sig], color='red', s=20, zorder=5)

        ax.axhline(0, color='black', alpha=0.3, linewidth=0.5)
        ax.set_xscale('log')
        ax.set_title(f'{res.type_a} × {res.type_b}', fontsize=9)
        ax.set_xlabel('Scale (μm)')
        ax.set_ylabel('Correlation')

    for idx in range(len(to_plot), nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].set_visible(False)

    fig.suptitle('Cross-Type Co-occurrence Across Scales', y=1.01)
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, bbox_inches='tight')
        logger.info(f'Saved {output_path}')
    return fig


# ── Figure 5: Regional Comparison ─────────────────────────────────────


def plot_regional_comparison(
    region_curves,
    figsize=(8, 5),
    output_path=None,
):
    """Overlay of regional VMR fingerprints.

    Parameters
    ----------
    region_curves : dict of str -> dict
        From compute_regional_fingerprints().
    figsize : tuple
    output_path : str or Path, optional
    """
    apply_style()
    fig, ax = plt.subplots(figsize=figsize)

    region_colors = {
        'isocortex': 'tab:blue',
        'hippocampus': 'tab:green',
        'thalamus': 'tab:orange',
    }

    for region, data in region_curves.items():
        color = region_colors.get(region.lower(), None)
        scales = np.array(data['scales_um'])
        vmr = np.array(data['vmr'])
        null_mean = np.array(data['null_vmr_mean'])
        null_std = np.array(data['null_vmr_std'])

        # Normalize: VMR ratio = real / null_mean
        # This removes the trivial scaling with bin size and isolates
        # excess clustering beyond CSR
        ratio = np.where(null_mean > 0, vmr / null_mean, 1.0)
        null_ratio_upper = np.where(
            null_mean > 0,
            (null_mean + 2 * null_std) / null_mean,
            1.0,
        )
        null_ratio_lower = np.where(
            null_mean > 0,
            np.maximum(0, (null_mean - 2 * null_std) / null_mean),
            1.0,
        )

        ax.fill_between(scales, null_ratio_lower, null_ratio_upper,
                        alpha=0.15, color=color)
        ax.plot(scales, ratio, '-o', color=color, markersize=4,
                linewidth=2, label=f'{region} ({data["n_cells"]:,} cells)')

    ax.axhline(1.0, color='black', alpha=0.4, linewidth=1,
               linestyle='--', label='CSR expectation')
    ax.set_xscale('log')
    ax.set_xlabel('Scale (μm)')
    ax.set_ylabel('VMR ratio (observed / CSR null)')
    ax.set_title('Regional Multiscale Organization')
    ax.legend()
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, bbox_inches='tight')
        logger.info(f'Saved {output_path}')
    return fig


# ── Figure 6: Characteristic Scales ──────────────────────────────────


def plot_characteristic_scales(
    fingerprints_by_region,
    figsize=(10, 6),
    output_path=None,
):
    """Characteristic clustering scale per type, by region.

    Parameters
    ----------
    fingerprints_by_region : dict of str -> dict of str -> TypeFingerprint
        Region -> cell_type -> TypeFingerprint.
    figsize : tuple
    output_path : str or Path, optional
    """
    apply_style()
    fig, ax = plt.subplots(figsize=figsize)

    region_colors = {
        'isocortex': 'tab:blue',
        'hippocampus': 'tab:green',
        'thalamus': 'tab:orange',
    }

    all_types = set()
    for fps in fingerprints_by_region.values():
        all_types.update(fps.keys())
    all_types = sorted(all_types)

    y_positions = {t: i for i, t in enumerate(all_types)}
    offsets = np.linspace(-0.2, 0.2, len(fingerprints_by_region))

    for (region, fps), offset in zip(fingerprints_by_region.items(), offsets):
        color = region_colors.get(region.lower(), 'gray')
        for ct, fp in fps.items():
            if fp.is_significant:
                y = y_positions[ct] + offset
                ax.scatter(fp.peak_scale_um, y, color=color, s=30,
                           alpha=0.8, edgecolors='black', linewidths=0.3)

        # Invisible for legend
        ax.scatter([], [], color=color, s=30, label=region)

    ax.set_yticks(range(len(all_types)))
    ax.set_yticklabels(all_types, fontsize=7)
    ax.set_xscale('log')
    ax.set_xlabel('Characteristic Clustering Scale (μm)')
    ax.set_title('Peak Clustering Scale per Cell Type')
    ax.legend()
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, bbox_inches='tight')
        logger.info(f'Saved {output_path}')
    return fig


# ── Figure 7: Label-Shuffle Control ──────────────────────────────────


def plot_label_shuffle_control(
    fingerprints,
    alpha=0.05,
    figsize=(8, 5),
    output_path=None,
):
    """Real vs shuffled clustering fractions.

    Shows the fraction of cell types that are significantly clustered
    under real labels vs label-shuffle expectation.

    Parameters
    ----------
    fingerprints : dict of str -> TypeFingerprint
    alpha : float
        Significance threshold.
    figsize : tuple
    output_path : str or Path, optional
    """
    apply_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    p_values = [fp.p_value for fp in fingerprints.values()]
    n_types = len(p_values)
    n_sig = sum(1 for p in p_values if p < alpha)
    expected_sig = alpha * n_types

    # Bar chart: observed vs expected
    ax1.bar(['Observed', 'Expected\n(5% FPR)'],
            [n_sig, expected_sig],
            color=['tab:red', 'gray'], alpha=0.8)
    ax1.set_ylabel('Number of significant types')
    ax1.set_title(f'Significant Types (α={alpha})')
    for i, v in enumerate([n_sig, expected_sig]):
        ax1.text(i, v + 0.3, f'{v:.1f}', ha='center', fontsize=10)

    # P-value histogram
    ax2.hist(p_values, bins=20, range=(0, 1), color='tab:blue',
             alpha=0.7, edgecolor='black', linewidth=0.5)
    ax2.axhline(n_types / 20, color='red', linestyle='--',
                label=f'Uniform ({n_types/20:.1f})')
    ax2.axvline(alpha, color='black', linestyle=':', alpha=0.5)
    ax2.set_xlabel('p-value')
    ax2.set_ylabel('Count')
    ax2.set_title('P-value Distribution')
    ax2.legend()

    fig.suptitle('Label-Shuffle Control', y=1.02)
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, bbox_inches='tight')
        logger.info(f'Saved {output_path}')
    return fig


# ── Figure 8: Grid Jitter Robustness ─────────────────────────────────


def plot_grid_jitter_robustness(
    jitter_results,
    top_types=6,
    ncols=3,
    figsize=None,
    output_path=None,
):
    """Show VMR ratio curves across grid jitter trials for stability.

    For each cell type, overlays the VMR/null ratio curve from each
    jitter trial. Tight bundles = robust, spread = grid-sensitive.

    Parameters
    ----------
    jitter_results : list of dict of str -> TypeFingerprint
        Fingerprints from each jitter trial.
    top_types : int
        Number of cell types to show (sorted by mean effect size).
    ncols : int
    figsize : tuple, optional
    output_path : str or Path, optional
    """
    apply_style()

    # Collect all types present in all trials
    all_types = set(jitter_results[0].keys())
    for jr in jitter_results[1:]:
        all_types &= set(jr.keys())

    # Rank by mean integrated_excess across trials
    type_scores = {}
    for ct in all_types:
        scores = [jr[ct].integrated_excess for jr in jitter_results if ct in jr]
        type_scores[ct] = np.mean(scores)
    ranked = sorted(type_scores, key=type_scores.get, reverse=True)[:top_types]

    nrows = (len(ranked) + ncols - 1) // ncols
    if figsize is None:
        figsize = (4.5 * ncols, 3.5 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.atleast_2d(axes)

    for idx, ct in enumerate(ranked):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]

        # Overlay all jitter trials
        for trial_idx, jr in enumerate(jitter_results):
            if ct not in jr:
                continue
            fp = jr[ct]
            scales = np.array(fp.scales_um)
            ratio = np.array(fp.vmr_ratio_curve)
            alpha = 0.4 if trial_idx > 0 else 0.9
            lw = 1.0 if trial_idx > 0 else 2.0
            ax.plot(scales, ratio, '-', color='tab:blue', alpha=alpha,
                    linewidth=lw)

        ax.axhline(1.0, color='gray', linestyle='--', linewidth=0.8)
        ax.set_xscale('log')
        ax.set_xlabel('Scale (μm)')
        ax.set_ylabel('VMR / null')

        # Compute CV of peak_vmr_ratio across trials
        peaks = [jr[ct].peak_vmr_ratio for jr in jitter_results if ct in jr]
        cv = np.std(peaks) / np.mean(peaks) if np.mean(peaks) > 0 else 0
        ax.set_title(f'{ct}\n(CV={cv:.2f}, n={len(peaks)} trials)', fontsize=9)

    for idx in range(len(ranked), nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].set_visible(False)

    fig.suptitle('Grid Jitter Robustness (VMR ratio across offset trials)',
                 y=1.01)
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, bbox_inches='tight')
        logger.info(f'Saved {output_path}')
    return fig


# ── Figure 9: Top Types by Effect Size ───────────────────────────────


def plot_top_types_by_effect(
    fingerprints_by_region,
    top_n=10,
    figsize=(10, 6),
    output_path=None,
):
    """Horizontal bar chart of top cell types ranked by effect size.

    Parameters
    ----------
    fingerprints_by_region : dict of str -> dict of str -> TypeFingerprint
    top_n : int
    figsize : tuple
    output_path : str or Path, optional
    """
    apply_style()

    region_colors = {
        'isocortex': 'tab:blue',
        'hippocampus': 'tab:green',
        'thalamus': 'tab:orange',
    }

    # Collect all fingerprints with region labels
    entries = []
    for region, fps in fingerprints_by_region.items():
        for ct, fp in fps.items():
            entries.append({
                'label': f'{ct} ({region[:4]})',
                'peak_vmr_ratio': fp.peak_vmr_ratio,
                'integrated_excess': fp.integrated_excess,
                'region': region,
            })

    # Sort by integrated_excess
    entries.sort(key=lambda e: e['integrated_excess'], reverse=True)
    entries = entries[:top_n]

    fig, ax = plt.subplots(figsize=figsize)
    y_pos = range(len(entries))
    colors = [region_colors.get(e['region'], 'gray') for e in entries]
    values = [e['integrated_excess'] for e in entries]
    labels = [e['label'] for e in entries]

    ax.barh(y_pos, values, color=colors, alpha=0.8, edgecolor='black',
            linewidth=0.3)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel('Integrated Excess (area under VMR ratio − 1)')
    ax.set_title(f'Top {top_n} Cell Types by Clustering Effect Size')

    # Add peak ratio annotations
    for i, e in enumerate(entries):
        ax.text(values[i] + 0.02 * max(values), i,
                f'{e["peak_vmr_ratio"]:.1f}x', va='center', fontsize=7)

    # Legend for regions
    for region, color in region_colors.items():
        if any(e['region'] == region for e in entries):
            ax.barh([], [], color=color, label=region)
    ax.legend(loc='lower right', fontsize=8)

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, bbox_inches='tight')
        logger.info(f'Saved {output_path}')
    return fig


# ── Figure 10: Curated Cross-Type Pairs ──────────────────────────────


def plot_curated_cooccurrence(
    results,
    figsize=(14, 4),
    output_path=None,
):
    """Three-panel curated co-occurrence: strongest positive, strongest
    negative, and best scale-switching pair.

    Parameters
    ----------
    results : dict of (str, str) -> CoOccurrenceResult
    figsize : tuple
    output_path : str or Path, optional
    """
    apply_style()

    if not results:
        return None

    # Find best candidates
    best_pos = max(results.values(), key=lambda r: r.max_positive_correlation)
    best_neg = min(results.values(), key=lambda r: r.max_negative_correlation)

    # Best scale-switcher: has sign change and highest excess
    switchers = [r for r in results.values() if r.sign_change_scale is not None]
    if switchers:
        best_switch = max(switchers, key=lambda r: r.excess_over_null)
    else:
        # Fallback to highest excess overall
        best_switch = max(results.values(), key=lambda r: r.excess_over_null)

    panels = [
        ('Strongest co-localization', best_pos, 'tab:green'),
        ('Strongest segregation', best_neg, 'tab:red'),
        ('Scale-switching', best_switch, 'tab:purple'),
    ]

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    for ax, (title, res, color) in zip(axes, panels):
        scales = np.array(res.scales_um)
        corr = np.array(res.correlation_curve)
        null_mean = np.array(res.null_corr_mean)
        null_std = np.array(res.null_corr_std)

        ax.fill_between(scales, null_mean - 2 * null_std,
                        null_mean + 2 * null_std,
                        alpha=0.2, color='gray')
        ax.plot(scales, null_mean, '--', color='gray', linewidth=0.8)
        ax.plot(scales, corr, '-o', color=color, markersize=4, linewidth=2)

        sig = np.array(res.significant_scales)
        if sig.any():
            ax.scatter(scales[sig], corr[sig], color='black', s=25, zorder=5,
                       marker='*')

        ax.axhline(0, color='black', alpha=0.3, linewidth=0.5)
        ax.set_xscale('log')
        ax.set_xlabel('Scale (μm)')
        ax.set_ylabel('Correlation')
        ax.set_title(f'{title}\n{res.type_a} × {res.type_b}', fontsize=9)

    fig.suptitle('Curated Cross-Type Co-occurrence Patterns', y=1.02)
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, bbox_inches='tight')
        logger.info(f'Saved {output_path}')
    return fig


# ── Figure 11: Universal Comparison ──────────────────────────────────


def plot_universal_comparison(
    tissue_curves=None,
    smlm_curves=None,
    dendrite_curves=None,
    figsize=(14, 4),
    output_path=None,
):
    """SMLM + dendrites + tissue fingerprints side by side.

    The key figure: VMR fingerprints from SMLM (nm), dendrites (μm), and
    tissue (mm) on the same normalized axis.

    Parameters
    ----------
    tissue_curves : dict, optional
        {'scales_um': [...], 'vmr': [...], 'null_mean': [...], 'null_std': [...]}
    smlm_curves : dict, optional
        {'scales_nm': [...], 'vmr': [...], ...}
    dendrite_curves : dict, optional
        {'scales_um': [...], 'vmr': [...], ...}
    figsize : tuple
    output_path : str or Path, optional
    """
    apply_style()

    panels = []
    if smlm_curves is not None:
        panels.append(('SMLM\n(protein nanoclusters)', smlm_curves, 'nm'))
    if dendrite_curves is not None:
        panels.append(('Dendrites\n(synapse clustering)', dendrite_curves, 'μm'))
    if tissue_curves is not None:
        panels.append(('Tissue\n(cell type organization)', tissue_curves, 'μm'))

    if not panels:
        # Placeholder with just tissue if nothing provided
        panels = [('Tissue\n(cell type organization)', {
            'scales_um': [10, 30, 100, 300, 1000],
            'vmr': [1.5, 3.0, 5.0, 2.0, 1.2],
        }, 'μm')]

    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(figsize[0] / 3 * n, figsize[1]))
    if n == 1:
        axes = [axes]

    colors = ['tab:blue', 'tab:green', 'tab:red']

    for ax, (title, curves, unit), color in zip(axes, panels, colors[:n]):
        if unit == 'nm':
            scales = np.array(curves.get('scales_nm', curves.get('scales_um', [])))
        else:
            scales = np.array(curves.get('scales_um', []))
        vmr = np.array(curves.get('vmr', []))

        ax.plot(scales, vmr, '-o', color=color, markersize=4, linewidth=2)

        if 'null_mean' in curves:
            null_m = np.array(curves['null_mean'])
            null_s = np.array(curves.get('null_std', np.zeros_like(null_m)))
            ax.fill_between(scales, null_m - 2 * null_s, null_m + 2 * null_s,
                            alpha=0.15, color='gray')
            ax.plot(scales, null_m, '--', color='gray', linewidth=0.8)

        ax.axhline(1.0, color='black', alpha=0.2, linewidth=0.5)
        ax.set_xscale('log')
        ax.set_xlabel(f'Scale ({unit})')
        ax.set_ylabel('VMR')
        ax.set_title(title, fontsize=11)

    fig.suptitle('Universal Multiscale Spatial Fingerprints', y=1.05,
                 fontsize=13, fontweight='bold')
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, bbox_inches='tight')
        logger.info(f'Saved {output_path}')
    return fig
