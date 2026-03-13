#!/usr/bin/env python3
"""Compute per-type multiscale spatial fingerprints for each brain region.

Usage:
    python scripts/02_multiscale_fingerprints.py [--data-dir data] [--n-mocks 50]

For each region, computes VMR fingerprints per cell type using label-shuffle
null. Saves results as JSON and summary CSV. (~30-60 min for 3 regions)
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
    compute_per_type_fingerprints,
    compute_jittered_fingerprints,
    fingerprints_to_dataframe,
)
from src.spatial_statistics import ScaleRange

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(name)s %(levelname)s: %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)

REGIONS = ['isocortex', 'hippocampus', 'thalamus']


def fingerprints_to_json(fingerprints):
    """Serialize fingerprints to JSON-compatible dict."""
    out = {}
    for ct, fp in fingerprints.items():
        out[ct] = {
            'cell_type': fp.cell_type,
            'n_cells': fp.n_cells,
            'vmr_curve': fp.vmr_curve,
            'skewness_curve': fp.skewness_curve,
            'scales_um': fp.scales_um,
            'mock_vmr_mean': fp.mock_vmr_mean,
            'mock_vmr_std': fp.mock_vmr_std,
            'chi_squared': fp.chi_squared,
            'f_statistic': fp.f_statistic,
            'p_value': fp.p_value,
            'peak_scale_um': fp.peak_scale_um,
            'is_significant': fp.is_significant,
            'peak_vmr_ratio': fp.peak_vmr_ratio,
            'integrated_excess': fp.integrated_excess,
            'fingerprint_width': fp.fingerprint_width,
            'clustering_direction': fp.clustering_direction,
        }
    return out


def main():
    parser = argparse.ArgumentParser(
        description='Compute per-type multiscale fingerprints'
    )
    parser.add_argument('--data-dir', default='data')
    parser.add_argument('--results-dir', default='results')
    parser.add_argument('--n-mocks', type=int, default=50)
    parser.add_argument('--min-cells', type=int, default=50)
    parser.add_argument('--grid-size', type=int, default=256)
    parser.add_argument('--regions', nargs='+', default=REGIONS)
    parser.add_argument('--type-level', default='class',
                        choices=['class', 'subclass', 'supertype'])
    parser.add_argument('--jitter', action='store_true',
                        help='Run grid jitter robustness check (isocortex only)')
    parser.add_argument('--n-jitters', type=int, default=10)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)
    all_results = {}

    for region in args.regions:
        parquet_path = data_dir / f'{region}_cells.parquet'
        if not parquet_path.exists():
            logger.warning(f'{parquet_path} not found, skipping {region}')
            continue

        logger.info('='*60)
        logger.info(f'Region: {region.upper()}')
        logger.info('='*60)

        # Load data
        region_df = load_region_data(parquet_path)
        positions, labels = prepare_analysis_data(region_df, args.type_level)
        logger.info(f'Loaded {len(positions):,} cells, '
                    f'{len(np.unique(labels))} types')

        # Compute fingerprints
        t0 = time.time()
        fingerprints = compute_per_type_fingerprints(
            positions, labels,
            n_mocks=args.n_mocks,
            min_cells=args.min_cells,
            grid_size=args.grid_size,
            rng=rng,
        )
        elapsed = time.time() - t0
        logger.info(f'Computed {len(fingerprints)} fingerprints in {elapsed:.1f}s')

        # Summary
        summary_df = fingerprints_to_dataframe(fingerprints)
        n_sig = summary_df['is_significant'].sum()
        logger.info(f'Significant: {n_sig}/{len(summary_df)} types '
                    f'({100*n_sig/max(1,len(summary_df)):.0f}%)')

        # Save results
        json_path = results_dir / f'{region}_fingerprints.json'
        with open(json_path, 'w') as f:
            json.dump(fingerprints_to_json(fingerprints), f, indent=2)
        logger.info(f'Saved {json_path}')

        csv_path = results_dir / f'{region}_fingerprints_summary.csv'
        summary_df.to_csv(csv_path, index=False)
        logger.info(f'Saved {csv_path}')

        all_results[region] = fingerprints

    # Grid jitter robustness check
    if args.jitter and 'isocortex' in args.regions:
        logger.info('='*60)
        logger.info('GRID JITTER ROBUSTNESS (isocortex)')
        logger.info('='*60)

        parquet_path = data_dir / 'isocortex_cells.parquet'
        if parquet_path.exists():
            region_df = load_region_data(parquet_path)
            positions, labels = prepare_analysis_data(region_df, args.type_level)

            t0 = time.time()
            jitter_results, offsets = compute_jittered_fingerprints(
                positions, labels,
                n_jitters=args.n_jitters,
                n_mocks=args.n_mocks,
                min_cells=args.min_cells,
                grid_size=args.grid_size,
                rng=rng,
            )
            elapsed = time.time() - t0
            logger.info(f'Jitter analysis completed in {elapsed:.1f}s')

            # Report stability
            all_types = set(jitter_results[0].keys())
            for jr in jitter_results[1:]:
                all_types &= set(jr.keys())

            logger.info(f'Types present in all trials: {len(all_types)}')
            for ct in sorted(all_types):
                peaks = [jr[ct].peak_vmr_ratio for jr in jitter_results]
                cv = np.std(peaks) / np.mean(peaks) if np.mean(peaks) > 0 else 0
                logger.info(f'  {ct}: peak ratio={np.mean(peaks):.2f}±{np.std(peaks):.2f} '
                            f'(CV={cv:.3f})')

            # Save jitter results
            jitter_json = {
                'offsets': [o.tolist() for o in offsets],
                'trials': [fingerprints_to_json(jr) for jr in jitter_results],
            }
            jitter_path = results_dir / 'isocortex_jitter.json'
            with open(jitter_path, 'w') as f:
                json.dump(jitter_json, f, indent=2)
            logger.info(f'Saved {jitter_path}')

    # Cross-region summary
    logger.info('='*60)
    logger.info('CROSS-REGION SUMMARY')
    logger.info('='*60)
    for region, fps in all_results.items():
        n_sig = sum(1 for fp in fps.values() if fp.is_significant)
        if fps:
            top = max(fps.values(), key=lambda fp: fp.integrated_excess)
            logger.info(f'  {region}: {n_sig}/{len(fps)} significant, '
                        f'top effect: {top.cell_type} '
                        f'(peak={top.peak_vmr_ratio:.1f}x, '
                        f'excess={top.integrated_excess:.2f})')
        else:
            logger.info(f'  {region}: no fingerprints computed')

    logger.info('Done.')


if __name__ == '__main__':
    main()
