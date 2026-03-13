#!/usr/bin/env python3
"""Compute cross-type co-localization fingerprints.

Usage:
    python scripts/03_cross_type_colocalization.py [--data-dir data]

For each region, computes pairwise co-occurrence correlations across scales.
Identifies scale-specific co-localization and segregation patterns. (~30 min)
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loading import load_region_data, prepare_analysis_data
from src.cell_type_clustering import (
    compute_cross_type_cooccurrence,
    compute_regional_fingerprints,
    cooccurrence_to_matrix,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(name)s %(levelname)s: %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)

REGIONS = ['isocortex', 'hippocampus', 'thalamus']


def cooccurrence_to_json(results):
    """Serialize co-occurrence results to JSON."""
    out = {}
    for (ta, tb), res in results.items():
        key = f'{ta}__x__{tb}'
        out[key] = {
            'type_a': res.type_a,
            'type_b': res.type_b,
            'scales_um': res.scales_um,
            'correlation_curve': res.correlation_curve,
            'null_corr_mean': res.null_corr_mean,
            'null_corr_std': res.null_corr_std,
            'peak_scale_um': res.peak_scale_um,
            'peak_correlation': res.peak_correlation,
            'significant_scales': res.significant_scales,
        }
    return out


def main():
    parser = argparse.ArgumentParser(
        description='Compute cross-type co-localization'
    )
    parser.add_argument('--data-dir', default='data')
    parser.add_argument('--results-dir', default='results')
    parser.add_argument('--n-mocks', type=int, default=50)
    parser.add_argument('--min-cells', type=int, default=50)
    parser.add_argument('--grid-size', type=int, default=256)
    parser.add_argument('--regions', nargs='+', default=REGIONS)
    parser.add_argument('--type-level', default='class',
                        choices=['class', 'subclass', 'supertype'])
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)

    # ── Cross-type co-occurrence per region ──
    for region in args.regions:
        parquet_path = data_dir / f'{region}_cells.parquet'
        if not parquet_path.exists():
            logger.warning(f'{parquet_path} not found, skipping')
            continue

        logger.info('='*60)
        logger.info(f'Cross-type co-occurrence: {region.upper()}')
        logger.info('='*60)

        region_df = load_region_data(parquet_path)
        positions, labels = prepare_analysis_data(region_df, args.type_level)

        t0 = time.time()
        results = compute_cross_type_cooccurrence(
            positions, labels,
            n_mocks=args.n_mocks,
            min_cells=args.min_cells,
            grid_size=args.grid_size,
            rng=rng,
        )
        elapsed = time.time() - t0
        logger.info(f'Computed {len(results)} pairs in {elapsed:.1f}s')

        # Save co-occurrence results
        json_path = results_dir / f'{region}_cooccurrence.json'
        with open(json_path, 'w') as f:
            json.dump(cooccurrence_to_json(results), f, indent=2)
        logger.info(f'Saved {json_path}')

        # Save correlation matrix at peak scale
        if results:
            corr_mat = cooccurrence_to_matrix(results)
            csv_path = results_dir / f'{region}_cooccurrence_matrix.csv'
            corr_mat.to_csv(csv_path)
            logger.info(f'Saved {csv_path}')

        # Summary
        n_sig = sum(1 for r in results.values() if any(r.significant_scales))
        logger.info(f'Pairs with significant co-occurrence: '
                    f'{n_sig}/{len(results)}')

    # ── Regional comparison ──
    logger.info('='*60)
    logger.info('Regional comparison')
    logger.info('='*60)

    region_data = {}
    for region in args.regions:
        parquet_path = data_dir / f'{region}_cells.parquet'
        if not parquet_path.exists():
            continue
        region_df = load_region_data(parquet_path)
        positions, labels = prepare_analysis_data(region_df, args.type_level)
        region_data[region] = (positions, labels)

    if region_data:
        regional_curves = compute_regional_fingerprints(
            region_data,
            n_mocks=min(20, args.n_mocks),
            grid_size=args.grid_size,
            rng=rng,
        )

        json_path = results_dir / 'regional_comparison.json'
        with open(json_path, 'w') as f:
            json.dump(regional_curves, f, indent=2)
        logger.info(f'Saved {json_path}')

    logger.info('Done.')


if __name__ == '__main__':
    main()
