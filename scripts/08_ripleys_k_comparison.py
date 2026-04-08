#!/usr/bin/env python3
"""Ripley's K/L comparison with VMR fingerprints.

Computes Ripley's K and Besag's L functions for a subset of cell types
and compares characteristic scales with VMR peak scales. This serves as
a supplementary validation that VMR fingerprints capture the same spatial
structure detected by the standard point-process statistic.
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import KDTree

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

from spatial_statistics import ScaleRange, TissueMultiscaleTest

logger = logging.getLogger(__name__)


def compute_ripleys_k(positions, radii, area, n_null=20, rng=None):
    """Compute Ripley's K and Besag's L for a set of 2D positions.

    Parameters
    ----------
    positions : (N, 2) array
    radii : array of distances at which to evaluate K
    area : total area of the study region (um^2)
    n_null : number of CSR null simulations for CI
    rng : numpy Generator

    Returns
    -------
    dict with 'radii', 'K_obs', 'L_obs', 'K_null_mean', 'L_null_mean',
         'L_null_lo', 'L_null_hi'
    """
    if rng is None:
        rng = np.random.default_rng(42)

    n = len(positions)
    if n < 10:
        return None

    # Subsample if too many points (K is O(N^2))
    max_pts = 5000
    if n > max_pts:
        idx = rng.choice(n, max_pts, replace=False)
        positions = positions[idx]
        n = max_pts

    tree = KDTree(positions)

    # Observed K(r)
    K_obs = np.zeros(len(radii))
    for i, r in enumerate(radii):
        # Count pairs within distance r
        pairs = tree.query_pairs(r)
        K_obs[i] = area * 2 * len(pairs) / (n * (n - 1))

    # Besag's L: L(r) = sqrt(K(r)/pi) - r
    L_obs = np.sqrt(K_obs / np.pi) - radii

    # CSR null: uniform random in bounding box
    x_min, y_min = positions.min(axis=0)
    x_max, y_max = positions.max(axis=0)

    L_nulls = np.zeros((n_null, len(radii)))
    for j in range(n_null):
        null_pos = np.column_stack([
            rng.uniform(x_min, x_max, n),
            rng.uniform(y_min, y_max, n),
        ])
        null_tree = KDTree(null_pos)
        K_null = np.zeros(len(radii))
        for i, r in enumerate(radii):
            pairs = null_tree.query_pairs(r)
            K_null[i] = area * 2 * len(pairs) / (n * (n - 1))
        L_nulls[j] = np.sqrt(K_null / np.pi) - radii

    return {
        'radii_um': radii.tolist(),
        'K_obs': K_obs.tolist(),
        'L_obs': L_obs.tolist(),
        'K_null_mean': K_obs.tolist(),  # placeholder
        'L_null_mean': L_nulls.mean(axis=0).tolist(),
        'L_null_lo': np.percentile(L_nulls, 2.5, axis=0).tolist(),
        'L_null_hi': np.percentile(L_nulls, 97.5, axis=0).tolist(),
    }


def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s: %(message)s',
    )

    data_dir = Path('data')
    results_dir = Path('results')
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load isocortex data
    parquet_path = data_dir / 'isocortex_cells.parquet'
    cells = pd.read_parquet(parquet_path)
    logger.info(f'Loaded {len(cells):,} isocortex cells')

    # Select representative cell types (mix of high/low effect sizes)
    target_types = [
        '01 IT-ET Glut',      # major excitatory
        '02 NP-CT-L6b Glut',  # deep layer
        '06 CTX-CGE GABA',    # CGE interneurons
        '07 CTX-MGE GABA',    # MGE interneurons
        '30 Astro-Epen',      # glial (sub-Poisson)
        '31 OPC-Oligo',       # oligodendrocytes
        '33 Vascular',        # vascular
        '34 Immune',          # immune (sub-Poisson)
    ]

    # Filter to available types
    available = cells['class'].unique()
    target_types = [t for t in target_types if t in available]
    logger.info(f'Analyzing {len(target_types)} types: {target_types}')

    # Positions in um
    positions_all = cells[['x', 'y']].values.astype(np.float64)
    if positions_all[:, 0].max() < 20:
        positions_all *= 1000.0

    # Compute VMR fingerprints for these types
    tester = TissueMultiscaleTest.from_positions(positions_all, grid_size=256)
    roi_size = float(tester.roi_size_um.max())
    scale_range = ScaleRange.for_tissue(
        min_um=10, max_um=min(1000, roi_size * 0.4),
        n_scales=12, roi_size_um=roi_size, grid_size=256,
    )

    # Radii for Ripley's K (match VMR scale range)
    radii = np.array(scale_range.cell_sizes_um)

    area = float(tester.roi_size_um[0] * tester.roi_size_um[1])
    rng = np.random.default_rng(42)

    results = []

    for cell_type in target_types:
        logger.info(f'Processing {cell_type}...')
        type_mask = cells['class'].values == cell_type
        type_pos = positions_all[type_mask]
        n_type = len(type_pos)

        # VMR fingerprint
        vmr_curve = tester.compute_curves(type_pos, scale_range)
        vmr_values = np.array(vmr_curve['vmr'])
        vmr_scales = np.array(vmr_curve['scales_um'])
        vmr_peak_scale = float(vmr_scales[np.argmax(vmr_values)])

        # Ripley's K/L
        rk = compute_ripleys_k(type_pos, radii, area, n_null=20, rng=rng)
        if rk is None:
            continue

        # Find characteristic L scale (max L above null)
        L_excess = np.array(rk['L_obs']) - np.array(rk['L_null_mean'])
        if np.any(L_excess > 0):
            L_peak_idx = np.argmax(L_excess)
            L_peak_scale = float(radii[L_peak_idx])
        else:
            L_peak_scale = float('nan')

        result = {
            'cell_type': cell_type,
            'n_cells': n_type,
            'vmr_peak_scale_um': vmr_peak_scale,
            'vmr_peak_value': float(np.max(vmr_values)),
            'L_peak_scale_um': L_peak_scale,
            'L_peak_excess': float(np.max(L_excess)) if np.any(L_excess > 0) else 0.0,
            'vmr_curve': vmr_values.tolist(),
            'vmr_scales': vmr_scales.tolist(),
            'ripleys_L': rk['L_obs'],
            'ripleys_L_null_mean': rk['L_null_mean'],
            'ripleys_L_null_lo': rk['L_null_lo'],
            'ripleys_L_null_hi': rk['L_null_hi'],
            'radii_um': rk['radii_um'],
        }
        results.append(result)

        logger.info(
            f'  VMR peak: {vmr_peak_scale:.0f} um (VMR={np.max(vmr_values):.2f}), '
            f'L peak: {L_peak_scale:.0f} um (excess={np.max(L_excess):.1f})'
        )

    # Correlation between VMR and L peak scales
    vmr_peaks = [r['vmr_peak_scale_um'] for r in results if not np.isnan(r['L_peak_scale_um'])]
    L_peaks = [r['L_peak_scale_um'] for r in results if not np.isnan(r['L_peak_scale_um'])]

    if len(vmr_peaks) >= 3:
        from scipy.stats import spearmanr
        rho, p = spearmanr(vmr_peaks, L_peaks)
        correlation = {'spearman_rho': float(rho), 'p_value': float(p), 'n': len(vmr_peaks)}
    else:
        correlation = {'spearman_rho': float('nan'), 'p_value': float('nan'), 'n': len(vmr_peaks)}

    out_path = results_dir / 'ripleys_k_comparison.json'
    with open(out_path, 'w') as f:
        json.dump({
            'analysis': 'ripleys_k_vs_vmr',
            'region': 'isocortex',
            'n_types': len(results),
            'correlation': correlation,
            'results': results,
        }, f, indent=2)
    logger.info(f'Saved to {out_path}')
    logger.info(f'VMR vs L peak scale correlation: rho={correlation["spearman_rho"]:.3f}, p={correlation["p_value"]:.3f}')


if __name__ == '__main__':
    main()
