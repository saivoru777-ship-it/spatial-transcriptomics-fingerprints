#!/usr/bin/env python3
"""Generate all publication figures from computed results.

Usage:
    python scripts/04_generate_figures.py [--results-dir results] [--fig-dir figures]

Reads precomputed fingerprints and co-occurrence data, generates 8 figures.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.cell_type_clustering import TypeFingerprint, CoOccurrenceResult
from src.data_loading import load_region_data, prepare_analysis_data
from src.viz import (
    plot_tissue_maps,
    plot_per_type_fingerprints,
    plot_cross_type_heatmap,
    plot_cooccurrence_curves,
    plot_regional_comparison,
    plot_characteristic_scales,
    plot_label_shuffle_control,
    plot_grid_jitter_robustness,
    plot_top_types_by_effect,
    plot_curated_cooccurrence,
    plot_universal_comparison,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(name)s %(levelname)s: %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)

REGIONS = ['isocortex', 'hippocampus', 'thalamus']


def load_fingerprints(json_path):
    """Load fingerprints from JSON."""
    with open(json_path) as f:
        data = json.load(f)
    fps = {}
    for ct, d in data.items():
        fps[ct] = TypeFingerprint(
            cell_type=d['cell_type'],
            n_cells=d['n_cells'],
            vmr_curve=d['vmr_curve'],
            skewness_curve=d.get('skewness_curve', []),
            scales_um=d['scales_um'],
            mock_vmr_mean=d['mock_vmr_mean'],
            mock_vmr_std=d['mock_vmr_std'],
            chi_squared=d['chi_squared'],
            f_statistic=d['f_statistic'],
            p_value=d['p_value'],
            peak_scale_um=d['peak_scale_um'],
            condition_number=d.get('condition_number', 0),
            n_mocks_used=d.get('n_mocks_used', 0),
        )
    return fps


def load_cooccurrence(json_path):
    """Load co-occurrence results from JSON."""
    with open(json_path) as f:
        data = json.load(f)
    results = {}
    for key, d in data.items():
        res = CoOccurrenceResult(
            type_a=d['type_a'],
            type_b=d['type_b'],
            scales_um=d['scales_um'],
            correlation_curve=d['correlation_curve'],
            null_corr_mean=d['null_corr_mean'],
            null_corr_std=d['null_corr_std'],
            peak_scale_um=d['peak_scale_um'],
            peak_correlation=d['peak_correlation'],
            significant_scales=d['significant_scales'],
        )
        results[(res.type_a, res.type_b)] = res
    return results


def main():
    parser = argparse.ArgumentParser(description='Generate figures')
    parser.add_argument('--data-dir', default='data')
    parser.add_argument('--results-dir', default='results')
    parser.add_argument('--fig-dir', default='figures')
    parser.add_argument('--regions', nargs='+', default=REGIONS)
    parser.add_argument('--type-level', default='class')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    results_dir = Path(args.results_dir)
    fig_dir = Path(args.fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    import matplotlib
    matplotlib.use('Agg')

    # ── Figure 1: Tissue Maps ──
    logger.info('Generating tissue maps...')
    region_data = {}
    for region in args.regions:
        parquet_path = data_dir / f'{region}_cells.parquet'
        if not parquet_path.exists():
            continue
        region_df = load_region_data(parquet_path)
        positions, labels = prepare_analysis_data(region_df, args.type_level)
        region_data[region] = (positions, labels)

    if region_data:
        plot_tissue_maps(
            region_data,
            type_level=args.type_level,
            output_path=fig_dir / 'fig_tissue_maps.pdf',
        )

    # ── Figure 2: Per-Type Fingerprints (best region) ──
    logger.info('Generating per-type fingerprint plots...')
    fingerprints_by_region = {}
    for region in args.regions:
        fp_path = results_dir / f'{region}_fingerprints.json'
        if fp_path.exists():
            fps = load_fingerprints(fp_path)
            fingerprints_by_region[region] = fps

            plot_per_type_fingerprints(
                fps,
                output_path=fig_dir / f'fig_per_type_fingerprints_{region}.pdf',
            )

    # ── Figure 3 & 4: Cross-Type Heatmap and Curves ──
    logger.info('Generating co-occurrence figures...')
    for region in args.regions:
        cooc_path = results_dir / f'{region}_cooccurrence.json'
        mat_path = results_dir / f'{region}_cooccurrence_matrix.csv'

        if mat_path.exists():
            corr_mat = pd.read_csv(mat_path, index_col=0)
            plot_cross_type_heatmap(
                corr_mat,
                output_path=fig_dir / f'fig_cross_type_heatmap_{region}.pdf',
            )

        if cooc_path.exists():
            cooc = load_cooccurrence(cooc_path)
            plot_cooccurrence_curves(
                cooc,
                output_path=fig_dir / f'fig_co_occurrence_curves_{region}.pdf',
            )

    # ── Figure 5: Regional Comparison ──
    logger.info('Generating regional comparison...')
    reg_path = results_dir / 'regional_comparison.json'
    if reg_path.exists():
        with open(reg_path) as f:
            regional_curves = json.load(f)
        plot_regional_comparison(
            regional_curves,
            output_path=fig_dir / 'fig_regional_comparison.pdf',
        )

    # ── Figure 6: Characteristic Scales ──
    if fingerprints_by_region:
        logger.info('Generating characteristic scales...')
        plot_characteristic_scales(
            fingerprints_by_region,
            output_path=fig_dir / 'fig_characteristic_scales.pdf',
        )

    # ── Figure 7: Label-Shuffle Control ──
    logger.info('Generating label-shuffle control...')
    for region in args.regions:
        if region in fingerprints_by_region:
            plot_label_shuffle_control(
                fingerprints_by_region[region],
                output_path=fig_dir / f'fig_label_shuffle_control_{region}.pdf',
            )

    # ── Figure 8: Grid Jitter Robustness ──
    jitter_path = results_dir / 'isocortex_jitter.json'
    if jitter_path.exists():
        logger.info('Generating grid jitter robustness figure...')
        with open(jitter_path) as f:
            jitter_data = json.load(f)
        # Reconstruct TypeFingerprint objects from jitter JSON
        jitter_fps = []
        for trial in jitter_data['trials']:
            fps = {}
            for ct, d in trial.items():
                fps[ct] = TypeFingerprint(
                    cell_type=d['cell_type'],
                    n_cells=d['n_cells'],
                    vmr_curve=d['vmr_curve'],
                    skewness_curve=d.get('skewness_curve', []),
                    scales_um=d['scales_um'],
                    mock_vmr_mean=d['mock_vmr_mean'],
                    mock_vmr_std=d['mock_vmr_std'],
                    chi_squared=d['chi_squared'],
                    f_statistic=d['f_statistic'],
                    p_value=d['p_value'],
                    peak_scale_um=d['peak_scale_um'],
                    condition_number=d.get('condition_number', 0),
                    n_mocks_used=d.get('n_mocks_used', 0),
                )
            jitter_fps.append(fps)

        plot_grid_jitter_robustness(
            jitter_fps,
            output_path=fig_dir / 'fig_grid_jitter_robustness.pdf',
        )

    # ── Figure 9: Top Types by Effect Size ──
    if fingerprints_by_region:
        logger.info('Generating top types by effect size figure...')
        plot_top_types_by_effect(
            fingerprints_by_region,
            output_path=fig_dir / 'fig_top_types_by_effect.pdf',
        )

    # ── Figure 10: Curated Cross-Type Pairs ──
    for region in args.regions:
        cooc_path = results_dir / f'{region}_cooccurrence.json'
        if cooc_path.exists():
            logger.info(f'Generating curated co-occurrence for {region}...')
            cooc = load_cooccurrence(cooc_path)
            plot_curated_cooccurrence(
                cooc,
                output_path=fig_dir / f'fig_curated_cooccurrence_{region}.pdf',
            )

    # ── Figure 11: Subclass-Level Fingerprints (Isocortex) ──
    subclass_fp_path = results_dir / 'isocortex_fingerprints_subclass.json'
    if subclass_fp_path.exists():
        subclass_fps = load_fingerprints(subclass_fp_path)
        logger.info(f'Generating subclass-level isocortex fingerprints '
                    f'({len(subclass_fps)} types)...')
        plot_per_type_fingerprints(
            subclass_fps,
            max_types=16,
            ncols=4,
            output_path=fig_dir / 'fig_isocortex_subclass_fingerprints.pdf',
        )

    # ── Figure 12: Universal Comparison ──
    logger.info('Generating universality figure...')
    # Use first region's aggregate curves as tissue representative
    if reg_path.exists():
        with open(reg_path) as f:
            regional_curves = json.load(f)
        first_region = list(regional_curves.keys())[0]
        tissue_data = regional_curves[first_region]
        tissue_curves = {
            'scales_um': tissue_data['scales_um'],
            'vmr': tissue_data['vmr'],
            'null_mean': tissue_data['null_vmr_mean'],
            'null_std': tissue_data['null_vmr_std'],
        }
    else:
        tissue_curves = None

    # Try loading SMLM and dendrite data if available
    smlm_curves = _try_load_smlm_curves()
    dendrite_curves = _try_load_dendrite_curves()

    plot_universal_comparison(
        tissue_curves=tissue_curves,
        smlm_curves=smlm_curves,
        dendrite_curves=dendrite_curves,
        output_path=fig_dir / 'fig_universal_comparison.pdf',
    )

    logger.info(f'All figures saved to {fig_dir}/')
    logger.info('Done.')


def _try_load_smlm_curves():
    """Try to load SMLM results from previous study."""
    import glob
    candidates = [
        Path.home() / 'research' / 'smlm-clustering' / 'results',
        Path.home() / 'results' / 'smlm',
    ]
    for d in candidates:
        for f in d.glob('*fingerprint*.json') if d.exists() else []:
            try:
                with open(f) as fh:
                    data = json.load(fh)
                if 'vmr' in data and ('scales_nm' in data or 'scales_um' in data):
                    logger.info(f'Loaded SMLM curves from {f}')
                    return data
            except Exception:
                pass
    return None


def _try_load_dendrite_curves():
    """Try to load dendrite results from previous study."""
    candidates = [
        Path.home() / 'research' / 'neurostat' / 'results',
        Path.home() / 'results' / 'dendrite',
    ]
    for d in candidates:
        for f in d.glob('*fingerprint*.json') if d.exists() else []:
            try:
                with open(f) as fh:
                    data = json.load(fh)
                if 'vmr' in data and 'scales_um' in data:
                    logger.info(f'Loaded dendrite curves from {f}')
                    return data
            except Exception:
                pass
    return None


if __name__ == '__main__':
    main()
