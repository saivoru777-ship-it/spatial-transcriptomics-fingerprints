#!/usr/bin/env python3
"""3D VMR fingerprints for thalamus — compare with 2D results.

Tests whether thalamic nuclear signatures change when using 3D gridding
instead of 2D projection. Thalamic nuclei are heavily z-laminated, so
2D-only analysis may miss or distort nuclear organization patterns.

Focus: thalamic reticular nucleus (RT) as the most z-laminated test case.
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

from spatial_statistics import (ScaleRange, TissueMultiscaleTest,
                                TissueMultiscaleTest3D,
                                chi_squared_with_covariance)

logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s: %(message)s',
    )

    data_dir = Path('data')
    results_dir = Path('results')

    # Load thalamus data
    cells = pd.read_parquet(data_dir / 'thalamus_cells.parquet')
    logger.info(f'Loaded {len(cells):,} thalamus cells')

    # Coordinates in um
    positions_2d = cells[['x', 'y']].values.astype(np.float64) * 1000.0
    positions_3d = cells[['x', 'y', 'z']].values.astype(np.float64) * 1000.0

    labels = cells['class'].values.astype(str)
    rng = np.random.default_rng(42)

    # Get unique types
    unique_types, type_counts = np.unique(labels, return_counts=True)
    valid_types = unique_types[type_counts >= 50]
    logger.info(f'{len(valid_types)} types with >= 50 cells')

    # Set up 2D and 3D testers
    tester_2d = TissueMultiscaleTest.from_positions(positions_2d, grid_size=256)
    tester_3d = TissueMultiscaleTest3D.from_positions(positions_3d, grid_size=64)

    roi_2d = float(tester_2d.roi_size_um.max())
    roi_3d = float(tester_3d.roi_size_um.max())

    scales_2d = ScaleRange.for_tissue(
        min_um=10, max_um=min(1000, roi_2d * 0.4),
        n_scales=12, roi_size_um=roi_2d, grid_size=256,
    )
    scales_3d = ScaleRange.for_tissue(
        min_um=10, max_um=min(1000, roi_3d * 0.4),
        n_scales=12, roi_size_um=roi_3d, grid_size=64,
    )

    # Generate label shuffles
    n_mocks = 50
    shuffled_list = [rng.permutation(labels) for _ in range(n_mocks)]

    results = []

    for cell_type in valid_types:
        type_mask = labels == cell_type
        type_2d = positions_2d[type_mask]
        type_3d = positions_3d[type_mask]
        n_type = int(type_mask.sum())

        # 2D curves
        real_2d = tester_2d.compute_curves(type_2d, scales_2d)
        mock_vmrs_2d = []
        for sl in shuffled_list:
            mp = positions_2d[sl == cell_type]
            if len(mp) >= 5:
                mc = tester_2d.compute_curves(mp, scales_2d)
                mock_vmrs_2d.append(mc['vmr'])

        # 3D curves
        real_3d = tester_3d.compute_curves(type_3d, scales_3d)
        mock_vmrs_3d = []
        for sl in shuffled_list:
            mp = positions_3d[sl == cell_type]
            if len(mp) >= 5:
                mc = tester_3d.compute_curves(mp, scales_3d)
                mock_vmrs_3d.append(mc['vmr'])

        if len(mock_vmrs_2d) < 5 or len(mock_vmrs_3d) < 5:
            continue

        test_2d = chi_squared_with_covariance(
            np.array(real_2d['vmr']), np.array(mock_vmrs_2d))
        test_3d = chi_squared_with_covariance(
            np.array(real_3d['vmr']), np.array(mock_vmrs_3d))

        # VMR ratios
        mock_mean_2d = np.array(mock_vmrs_2d).mean(axis=0)
        mock_mean_3d = np.array(mock_vmrs_3d).mean(axis=0)
        peak_ratio_2d = max(r / m if m > 0 else 1.0
                            for r, m in zip(real_2d['vmr'], mock_mean_2d))
        peak_ratio_3d = max(r / m if m > 0 else 1.0
                            for r, m in zip(real_3d['vmr'], mock_mean_3d))

        result = {
            'cell_type': cell_type,
            'n_cells': n_type,
            'chi2_2d': float(test_2d['chi_squared']),
            'p_2d': float(test_2d['p_value']),
            'chi2_3d': float(test_3d['chi_squared']),
            'p_3d': float(test_3d['p_value']),
            'peak_vmr_ratio_2d': float(peak_ratio_2d),
            'peak_vmr_ratio_3d': float(peak_ratio_3d),
            'ratio_change': float(peak_ratio_3d / peak_ratio_2d)
                if peak_ratio_2d > 0 else 1.0,
        }
        results.append(result)

        logger.info(
            f'{cell_type}: 2D peak={peak_ratio_2d:.2f} (p={test_2d["p_value"]:.2e}), '
            f'3D peak={peak_ratio_3d:.2f} (p={test_3d["p_value"]:.2e}), '
            f'ratio change={result["ratio_change"]:.2f}'
        )

    # Save
    out = {
        'analysis': '2D vs 3D VMR fingerprints for thalamus',
        'grid_2d': 256, 'grid_3d': 64,
        'n_mocks': n_mocks,
        'results': results,
    }

    out_path = results_dir / 'thalamus_2d_vs_3d.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    logger.info(f'Saved to {out_path}')

    # Summary
    if results:
        ratios = [r['ratio_change'] for r in results]
        logger.info(f'\n--- 2D vs 3D Summary ---')
        logger.info(f'Mean ratio change (3D/2D): {np.mean(ratios):.2f}')
        logger.info(f'Median ratio change: {np.median(ratios):.2f}')
        sig_change = sum(1 for r in results
                         if abs(r['ratio_change'] - 1.0) > 0.20)
        logger.info(f'Types with >20% change: {sig_change}/{len(results)}')


if __name__ == '__main__':
    main()
