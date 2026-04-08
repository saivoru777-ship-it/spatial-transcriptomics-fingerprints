#!/usr/bin/env python3
"""Bootstrap CIs on mediation + continuous depth mediation.

Part 1: Bootstrap confidence intervals on mediation percentages.
  - Re-runs mediation analysis, saves raw mock VMR matrices
  - Resamples mock indices with replacement 200 times
  - Reports 95% CIs on mediation_pct for each cell type

Part 2: Continuous depth mediation for isocortex.
  - Bins z-coordinate into quintiles within each layer (30 strata)
  - Compares chi2(layer+depth) with chi2(layer-only) and chi2(overall)
  - Tests whether continuous depth explains residual beyond discrete layer
"""

import json
import logging
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

from spatial_statistics import ScaleRange, TissueMultiscaleTest, chi_squared_with_covariance

logger = logging.getLogger(__name__)


def extract_cortical_layer(substructure):
    if pd.isna(substructure) or substructure == 'unassigned':
        return None
    m = re.search(r'(\d+(?:/\d+)?[a-b]?)$', substructure)
    if m:
        layer = m.group(1)
        if layer[0] in '123456':
            return f'L{layer}'
    return None


def augment_with_parcellation(parquet_path, region):
    data_dir = Path(parquet_path).parent
    cells = pd.read_parquet(parquet_path)
    logger.info(f'Loaded {len(cells):,} cells from {parquet_path}')

    ccf_path = data_dir / 'abc_atlas' / 'metadata' / 'MERFISH-C57BL6J-638850-CCF' / '20231215' / 'ccf_coordinates.csv'
    ccf = pd.read_csv(ccf_path, dtype={'cell_label': str})
    ccf.set_index('cell_label', inplace=True)

    parc_path = (data_dir / 'abc_atlas' / 'metadata' / 'Allen-CCF-2020' /
                 '20230630' / 'views' /
                 'parcellation_to_parcellation_term_membership_acronym.csv')
    parc = pd.read_csv(parc_path)

    cells = cells.join(ccf[['parcellation_index']], how='left')
    cells = cells.merge(
        parc[['parcellation_index', 'structure', 'substructure']],
        on='parcellation_index', how='left',
    )
    cells.index.name = 'cell_label'

    if region == 'isocortex':
        cells['layer'] = cells['substructure'].apply(extract_cortical_layer)
    elif region == 'thalamus':
        cells['nucleus'] = cells['structure']

    return cells


def within_stratum_shuffle(labels, strata, rng):
    shuffled = labels.copy()
    for stratum in np.unique(strata):
        if pd.isna(stratum):
            continue
        mask = strata == stratum
        idx = np.where(mask)[0]
        shuffled[idx] = rng.permutation(labels[idx])
    return shuffled


def compute_mediation_with_bootstrap(
    positions, labels, strata, scale_range=None,
    n_mocks=50, n_bootstrap=200, min_cells=50, grid_size=256, rng=None,
):
    """Compute mediation with bootstrap CIs on the mock distributions."""
    if rng is None:
        rng = np.random.default_rng(42)

    positions = np.asarray(positions, dtype=float)
    labels = np.asarray(labels)
    strata = np.asarray(strata)

    valid = ~pd.isna(strata) & (strata != 'unassigned')
    positions_v = positions[valid]
    labels_v = labels[valid]
    strata_v = strata[valid]

    tester = TissueMultiscaleTest.from_positions(positions_v, grid_size)

    if scale_range is None:
        roi_size = float(tester.roi_size_um.max())
        scale_range = ScaleRange.for_tissue(
            min_um=10, max_um=min(1000, roi_size * 0.4),
            n_scales=12, roi_size_um=roi_size, grid_size=grid_size,
        )

    logger.info(f'Generating {n_mocks} overall shuffles...')
    overall_shuffles = [rng.permutation(labels_v) for _ in range(n_mocks)]

    logger.info(f'Generating {n_mocks} within-stratum shuffles...')
    stratum_shuffles = [within_stratum_shuffle(labels_v, strata_v, rng)
                        for _ in range(n_mocks)]

    unique_types, type_counts = np.unique(labels_v, return_counts=True)
    valid_types = unique_types[type_counts >= min_cells]
    logger.info(f'Analyzing {len(valid_types)} types with >= {min_cells} cells')

    results = []

    for i, cell_type in enumerate(valid_types):
        type_mask = labels_v == cell_type
        type_pos = positions_v[type_mask]
        n_type = int(type_mask.sum())

        real_curve = tester.compute_curves(type_pos, scale_range)
        real_vmr = np.array(real_curve['vmr'])

        # Collect raw mock VMR curves
        overall_vmrs = []
        for shuffled in overall_shuffles:
            mock_pos = positions_v[shuffled == cell_type]
            if len(mock_pos) < 5:
                continue
            mc = tester.compute_curves(mock_pos, scale_range)
            overall_vmrs.append(mc['vmr'])

        stratum_vmrs = []
        for shuffled in stratum_shuffles:
            mock_pos = positions_v[shuffled == cell_type]
            if len(mock_pos) < 5:
                continue
            mc = tester.compute_curves(mock_pos, scale_range)
            stratum_vmrs.append(mc['vmr'])

        if len(overall_vmrs) < 10 or len(stratum_vmrs) < 10:
            logger.warning(f'{cell_type}: insufficient mocks ({len(overall_vmrs)}, {len(stratum_vmrs)}), skipping')
            continue

        overall_mat = np.array(overall_vmrs)
        stratum_mat = np.array(stratum_vmrs)

        # Point estimate
        test_overall = chi_squared_with_covariance(real_vmr, overall_mat)
        test_stratum = chi_squared_with_covariance(real_vmr, stratum_mat)
        chi2_o = test_overall['chi_squared']
        chi2_s = test_stratum['chi_squared']
        med_point = 100.0 * (1.0 - chi2_s / chi2_o) if chi2_o > 0 else 0.0

        # Bootstrap: resample mock indices with replacement
        boot_mediations = []
        n_o = len(overall_vmrs)
        n_s = len(stratum_vmrs)

        for _ in range(n_bootstrap):
            idx_o = rng.choice(n_o, size=n_o, replace=True)
            idx_s = rng.choice(n_s, size=n_s, replace=True)

            boot_overall = overall_mat[idx_o]
            boot_stratum = stratum_mat[idx_s]

            try:
                t_o = chi_squared_with_covariance(real_vmr, boot_overall)
                t_s = chi_squared_with_covariance(real_vmr, boot_stratum)
                c_o = t_o['chi_squared']
                c_s = t_s['chi_squared']
                if c_o > 0:
                    boot_mediations.append(100.0 * (1.0 - c_s / c_o))
            except Exception:
                continue

        if len(boot_mediations) < 50:
            ci_lo, ci_hi = float('nan'), float('nan')
        else:
            ci_lo = float(np.percentile(boot_mediations, 2.5))
            ci_hi = float(np.percentile(boot_mediations, 97.5))

        # VMR ratios
        overall_mean = overall_mat.mean(axis=0)
        stratum_mean = stratum_mat.mean(axis=0)
        vmr_ratio_overall = [float(r / m) if m > 0 else 1.0
                             for r, m in zip(real_vmr, overall_mean)]
        vmr_ratio_stratum = [float(r / m) if m > 0 else 1.0
                             for r, m in zip(real_vmr, stratum_mean)]

        result = {
            'cell_type': cell_type,
            'n_cells': n_type,
            'chi2_overall': float(chi2_o),
            'p_overall': float(test_overall['p_value']),
            'chi2_within_stratum': float(chi2_s),
            'p_within_stratum': float(test_stratum['p_value']),
            'mediation_pct': float(med_point),
            'mediation_ci_lo': ci_lo,
            'mediation_ci_hi': ci_hi,
            'n_bootstrap': len(boot_mediations),
            'peak_vmr_ratio_overall': float(max(vmr_ratio_overall)),
            'peak_vmr_ratio_within_stratum': float(max(vmr_ratio_stratum)),
            'scales_um': [float(s) for s in real_curve['scales_um']],
            'n_mocks_overall': len(overall_vmrs),
            'n_mocks_stratum': len(stratum_vmrs),
        }
        results.append(result)

        logger.info(
            f'[{i+1}/{len(valid_types)}] {cell_type}: '
            f'mediation={med_point:.1f}% [{ci_lo:.1f}%, {ci_hi:.1f}%]'
        )

    return results


def compute_depth_mediation(
    positions_xy, z_coords, labels, layers,
    n_depth_bins=5, n_mocks=50, min_cells=50, grid_size=256, rng=None,
):
    """Continuous depth mediation for isocortex.

    Creates compound strata: layer × depth-quintile.
    Compares within-(layer×depth) shuffling to within-layer-only shuffling.
    """
    if rng is None:
        rng = np.random.default_rng(123)

    positions_xy = np.asarray(positions_xy, dtype=float)
    z_coords = np.asarray(z_coords, dtype=float)
    labels = np.asarray(labels)
    layers = np.asarray(layers)

    valid = ~pd.isna(layers) & (layers != 'unassigned') & ~np.isnan(z_coords)
    pos_v = positions_xy[valid]
    z_v = z_coords[valid]
    lab_v = labels[valid]
    lay_v = layers[valid]

    # Create compound strata: layer × depth-quintile
    compound_strata = np.empty(len(lab_v), dtype=object)
    for layer in np.unique(lay_v):
        mask = lay_v == layer
        z_layer = z_v[mask]
        # Bin into quintiles within this layer
        try:
            bins = pd.qcut(z_layer, n_depth_bins, labels=False, duplicates='drop')
        except ValueError:
            bins = np.zeros(len(z_layer), dtype=int)
        compound_strata[mask] = [f'{layer}_d{int(b)}' for b in bins]

    n_compound = len(np.unique(compound_strata))
    logger.info(f'Compound strata (layer × depth): {n_compound} unique')

    tester = TissueMultiscaleTest.from_positions(pos_v, grid_size)
    roi_size = float(tester.roi_size_um.max())
    scale_range = ScaleRange.for_tissue(
        min_um=10, max_um=min(1000, roi_size * 0.4),
        n_scales=12, roi_size_um=roi_size, grid_size=grid_size,
    )

    # Generate three sets of shuffles
    logger.info(f'Generating shuffles...')
    overall_shuffles = [rng.permutation(lab_v) for _ in range(n_mocks)]
    layer_shuffles = [within_stratum_shuffle(lab_v, lay_v, rng) for _ in range(n_mocks)]
    compound_shuffles = [within_stratum_shuffle(lab_v, compound_strata, rng) for _ in range(n_mocks)]

    unique_types, type_counts = np.unique(lab_v, return_counts=True)
    valid_types = unique_types[type_counts >= min_cells]
    logger.info(f'Analyzing {len(valid_types)} types')

    results = []

    for i, cell_type in enumerate(valid_types):
        type_mask = lab_v == cell_type
        type_pos = pos_v[type_mask]
        n_type = int(type_mask.sum())

        real_curve = tester.compute_curves(type_pos, scale_range)
        real_vmr = np.array(real_curve['vmr'])

        # Three null distributions
        def collect_mocks(shuffles):
            vmrs = []
            for shuffled in shuffles:
                mock_pos = pos_v[shuffled == cell_type]
                if len(mock_pos) < 5:
                    continue
                mc = tester.compute_curves(mock_pos, scale_range)
                vmrs.append(mc['vmr'])
            return vmrs

        overall_vmrs = collect_mocks(overall_shuffles)
        layer_vmrs = collect_mocks(layer_shuffles)
        compound_vmrs = collect_mocks(compound_shuffles)

        if len(overall_vmrs) < 10 or len(layer_vmrs) < 10 or len(compound_vmrs) < 10:
            continue

        overall_mat = np.array(overall_vmrs)
        layer_mat = np.array(layer_vmrs)
        compound_mat = np.array(compound_vmrs)

        t_o = chi_squared_with_covariance(real_vmr, overall_mat)
        t_l = chi_squared_with_covariance(real_vmr, layer_mat)
        t_c = chi_squared_with_covariance(real_vmr, compound_mat)

        chi2_o = t_o['chi_squared']
        chi2_l = t_l['chi_squared']
        chi2_c = t_c['chi_squared']

        med_layer = 100.0 * (1.0 - chi2_l / chi2_o) if chi2_o > 0 else 0.0
        med_compound = 100.0 * (1.0 - chi2_c / chi2_o) if chi2_o > 0 else 0.0
        depth_increment = med_compound - med_layer

        result = {
            'cell_type': cell_type,
            'n_cells': n_type,
            'chi2_overall': float(chi2_o),
            'chi2_layer_only': float(chi2_l),
            'chi2_layer_plus_depth': float(chi2_c),
            'mediation_layer_only': float(med_layer),
            'mediation_layer_plus_depth': float(med_compound),
            'depth_increment_pct': float(depth_increment),
            'n_compound_strata': n_compound,
        }
        results.append(result)

        logger.info(
            f'[{i+1}/{len(valid_types)}] {cell_type}: '
            f'layer={med_layer:.1f}%, layer+depth={med_compound:.1f}%, '
            f'depth increment={depth_increment:+.1f}%'
        )

    return results


def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s: %(message)s',
    )

    data_dir = Path('data')
    results_dir = Path('results')
    results_dir.mkdir(parents=True, exist_ok=True)

    # ── Part 1: Bootstrap CIs ──
    for region in ['isocortex', 'thalamus']:
        logger.info(f'\n{"="*60}')
        logger.info(f'BOOTSTRAP CIs: {region}')
        logger.info(f'{"="*60}')

        parquet_path = data_dir / f'{region}_cells.parquet'
        if not parquet_path.exists():
            continue

        cells = augment_with_parcellation(parquet_path, region)
        stratum_col = 'layer' if region == 'isocortex' else 'nucleus'

        valid_mask = cells[stratum_col].notna()
        cells_valid = cells[valid_mask]

        positions = cells_valid[['x', 'y']].values.astype(np.float64)
        if positions[:, 0].max() < 20:
            positions = positions * 1000.0

        labels = cells_valid['class'].values.astype(str)
        strata = cells_valid[stratum_col].values.astype(str)

        results = compute_mediation_with_bootstrap(
            positions, labels, strata,
            n_mocks=50, n_bootstrap=200,
            rng=np.random.default_rng(42),
        )

        out_path = results_dir / f'{region}_mediation_bootstrap.json'
        with open(out_path, 'w') as f:
            json.dump({
                'region': region,
                'stratum_variable': stratum_col,
                'n_mocks': 50,
                'n_bootstrap': 200,
                'results': results,
                'summary': {
                    'n_types': len(results),
                    'median_mediation': float(np.median([r['mediation_pct'] for r in results])) if results else 0,
                },
            }, f, indent=2)
        logger.info(f'Saved bootstrap results to {out_path}')

    # ── Part 2: Continuous depth mediation (isocortex only) ──
    logger.info(f'\n{"="*60}')
    logger.info(f'CONTINUOUS DEPTH MEDIATION: isocortex')
    logger.info(f'{"="*60}')

    parquet_path = data_dir / 'isocortex_cells.parquet'
    cells = augment_with_parcellation(parquet_path, 'isocortex')

    valid_mask = cells['layer'].notna()
    cells_valid = cells[valid_mask]

    positions_xy = cells_valid[['x', 'y']].values.astype(np.float64)
    if positions_xy[:, 0].max() < 20:
        positions_xy = positions_xy * 1000.0

    z_coords = cells_valid['z'].values.astype(np.float64)
    labels = cells_valid['class'].values.astype(str)
    layers = cells_valid['layer'].values.astype(str)

    depth_results = compute_depth_mediation(
        positions_xy, z_coords, labels, layers,
        n_depth_bins=5, n_mocks=50,
        rng=np.random.default_rng(99),
    )

    out_path = results_dir / 'isocortex_depth_mediation.json'
    with open(out_path, 'w') as f:
        json.dump({
            'analysis': 'continuous_depth_mediation',
            'region': 'isocortex',
            'n_depth_bins': 5,
            'n_mocks': 50,
            'results': depth_results,
            'summary': {
                'n_types': len(depth_results),
                'mean_depth_increment': float(np.mean([r['depth_increment_pct'] for r in depth_results])) if depth_results else 0,
                'median_depth_increment': float(np.median([r['depth_increment_pct'] for r in depth_results])) if depth_results else 0,
            },
        }, f, indent=2)
    logger.info(f'Saved depth mediation results to {out_path}')


if __name__ == '__main__':
    main()
