#!/usr/bin/env python3
"""Tissue-scale mediation analysis.

Tests whether cortical layer assignment (isocortex) or nuclear membership
(thalamus) mediates cell type spatial fingerprints. Uses a within-stratum
label-shuffle null: instead of shuffling all labels globally, labels are
shuffled only within each stratum (layer or nucleus). If the signal
disappears under within-stratum shuffling, the stratum variable explains
most of the spatial organization.

Mediation % = 1 - chi²(within-stratum null) / chi²(overall null)

High mediation: stratum explains the fingerprint (layered/nuclear architecture).
Low mediation: spatial structure exists beyond stratum assignment.
"""

import argparse
import json
import logging
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path so we can import src as a package
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

# Import directly from modules (avoiding relative import issues)
from spatial_statistics import ScaleRange, TissueMultiscaleTest, chi_squared_with_covariance

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data augmentation: add layer / nucleus labels
# ---------------------------------------------------------------------------

def extract_cortical_layer(substructure: str) -> str:
    """Extract cortical layer from CCF substructure string.

    Examples: SSp-m6b -> L6b, VISpm2/3 -> L2/3, GU1 -> L1, PL5 -> L5
    """
    if pd.isna(substructure) or substructure == 'unassigned':
        return None
    # Match trailing layer designation: digits, optional /digits, optional a-b
    m = re.search(r'(\d+(?:/\d+)?[a-b]?)$', substructure)
    if m:
        layer = m.group(1)
        if layer[0] in '123456':
            return f'L{layer}'
    return None


def augment_with_parcellation(parquet_path, region):
    """Load parquet and merge in parcellation structure/substructure.

    Returns DataFrame with original columns + 'structure', 'substructure',
    and for isocortex: 'layer', for thalamus: 'nucleus'.
    """
    data_dir = Path(parquet_path).parent

    # Load cell data
    cells = pd.read_parquet(parquet_path)
    logger.info(f'Loaded {len(cells):,} cells from {parquet_path}')

    # Load CCF coordinates (cell_label -> parcellation_index)
    ccf_path = data_dir / 'abc_atlas' / 'metadata' / 'MERFISH-C57BL6J-638850-CCF' / '20231215' / 'ccf_coordinates.csv'
    ccf = pd.read_csv(ccf_path, dtype={'cell_label': str})
    ccf.set_index('cell_label', inplace=True)

    # Load parcellation mapping (parcellation_index -> structure/substructure)
    parc_path = (data_dir / 'abc_atlas' / 'metadata' / 'Allen-CCF-2020' /
                 '20230630' / 'views' /
                 'parcellation_to_parcellation_term_membership_acronym.csv')
    parc = pd.read_csv(parc_path)

    # Merge: cells -> parcellation_index -> structure/substructure
    cells = cells.join(ccf[['parcellation_index']], how='left')
    cells = cells.merge(
        parc[['parcellation_index', 'structure', 'substructure']],
        on='parcellation_index', how='left',
    )
    cells.index.name = 'cell_label'

    if region == 'isocortex':
        cells['layer'] = cells['substructure'].apply(extract_cortical_layer)
        n_with_layer = cells['layer'].notna().sum()
        logger.info(f'Layer assignment: {n_with_layer:,}/{len(cells):,} '
                    f'({100*n_with_layer/len(cells):.1f}%)')
    elif region == 'thalamus':
        cells['nucleus'] = cells['structure']
        n_with_nuc = cells['nucleus'].notna().sum()
        logger.info(f'Nucleus assignment: {n_with_nuc:,}/{len(cells):,} '
                    f'({100*n_with_nuc/len(cells):.1f}%)')

    return cells


# ---------------------------------------------------------------------------
# Within-stratum label shuffle
# ---------------------------------------------------------------------------

def within_stratum_shuffle(labels, strata, rng):
    """Shuffle labels independently within each stratum.

    Preserves: all spatial structure, stratum composition per cell.
    Permutes: type-to-position mapping within each stratum.

    Parameters
    ----------
    labels : np.ndarray, shape (N,)
        Cell type labels.
    strata : np.ndarray, shape (N,)
        Stratum assignments (layer or nucleus).
    rng : np.random.Generator

    Returns
    -------
    shuffled : np.ndarray, shape (N,)
        Labels shuffled within each stratum.
    """
    shuffled = labels.copy()
    for stratum in np.unique(strata):
        if pd.isna(stratum):
            continue
        mask = strata == stratum
        idx = np.where(mask)[0]
        shuffled[idx] = rng.permutation(labels[idx])
    return shuffled


# ---------------------------------------------------------------------------
# Mediation analysis
# ---------------------------------------------------------------------------

def compute_mediation(
    positions, labels, strata, scale_range=None,
    n_mocks=50, min_cells=50, grid_size=256, rng=None,
):
    """Compute mediation of spatial fingerprints by stratum variable.

    For each cell type:
    1. Compute VMR fingerprint under overall label-shuffle null -> chi²_overall
    2. Compute VMR fingerprint under within-stratum null -> chi²_within
    3. Mediation % = 1 - chi²_within / chi²_overall

    Parameters
    ----------
    positions : np.ndarray, shape (N, 2)
    labels : np.ndarray, shape (N,)
    strata : np.ndarray, shape (N,)
        Layer (isocortex) or nucleus (thalamus) assignments.
    scale_range, n_mocks, min_cells, grid_size, rng : as in compute_per_type_fingerprints

    Returns
    -------
    results : list of dict
        Per-cell-type mediation results.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    positions = np.asarray(positions, dtype=float)
    labels = np.asarray(labels)
    strata = np.asarray(strata)

    # Filter to cells with valid strata
    valid = ~pd.isna(strata) & (strata != 'unassigned')
    positions_v = positions[valid]
    labels_v = labels[valid]
    strata_v = strata[valid]

    logger.info(f'Valid cells with stratum: {valid.sum():,}/{len(valid):,}')

    # Set up tester
    tester = TissueMultiscaleTest.from_positions(positions_v, grid_size)

    if scale_range is None:
        roi_size = float(tester.roi_size_um.max())
        scale_range = ScaleRange.for_tissue(
            min_um=10, max_um=min(1000, roi_size * 0.4),
            n_scales=12, roi_size_um=roi_size, grid_size=grid_size,
        )

    # Generate both types of shuffled labels
    logger.info(f'Generating {n_mocks} overall shuffles...')
    overall_shuffles = [rng.permutation(labels_v) for _ in range(n_mocks)]

    logger.info(f'Generating {n_mocks} within-stratum shuffles...')
    stratum_shuffles = [within_stratum_shuffle(labels_v, strata_v, rng)
                        for _ in range(n_mocks)]

    # Get valid types
    unique_types, type_counts = np.unique(labels_v, return_counts=True)
    valid_types = unique_types[type_counts >= min_cells]
    logger.info(f'Analyzing {len(valid_types)} types with >= {min_cells} cells')

    results = []

    for i, cell_type in enumerate(valid_types):
        type_mask = labels_v == cell_type
        type_pos = positions_v[type_mask]
        n_type = int(type_mask.sum())

        # Real VMR curve
        real_curve = tester.compute_curves(type_pos, scale_range)
        real_vmr = np.array(real_curve['vmr'])

        # Overall null VMR curves
        overall_vmrs = []
        for shuffled in overall_shuffles:
            mock_pos = positions_v[shuffled == cell_type]
            if len(mock_pos) < 5:
                continue
            mc = tester.compute_curves(mock_pos, scale_range)
            overall_vmrs.append(mc['vmr'])

        # Within-stratum null VMR curves
        stratum_vmrs = []
        for shuffled in stratum_shuffles:
            mock_pos = positions_v[shuffled == cell_type]
            if len(mock_pos) < 5:
                continue
            mc = tester.compute_curves(mock_pos, scale_range)
            stratum_vmrs.append(mc['vmr'])

        if len(overall_vmrs) < 5 or len(stratum_vmrs) < 5:
            logger.warning(f'{cell_type}: insufficient mocks, skipping')
            continue

        # Chi-squared tests
        overall_mat = np.array(overall_vmrs)
        stratum_mat = np.array(stratum_vmrs)

        test_overall = chi_squared_with_covariance(real_vmr, overall_mat)
        test_stratum = chi_squared_with_covariance(real_vmr, stratum_mat)

        chi2_overall = test_overall['chi_squared']
        chi2_stratum = test_stratum['chi_squared']

        # Mediation: fraction of signal explained by stratum
        if chi2_overall > 0:
            mediation_pct = 100.0 * (1.0 - chi2_stratum / chi2_overall)
        else:
            mediation_pct = 0.0

        # VMR ratio curves for reporting
        overall_mean = overall_mat.mean(axis=0)
        stratum_mean = stratum_mat.mean(axis=0)
        vmr_ratio_overall = [float(r / m) if m > 0 else 1.0
                             for r, m in zip(real_vmr, overall_mean)]
        vmr_ratio_stratum = [float(r / m) if m > 0 else 1.0
                             for r, m in zip(real_vmr, stratum_mean)]

        # Peak VMR ratios
        peak_overall = max(vmr_ratio_overall) if vmr_ratio_overall else 1.0
        peak_stratum = max(vmr_ratio_stratum) if vmr_ratio_stratum else 1.0

        # Integrated excess
        log_scales = np.log10(real_curve['scales_um'])
        ie_overall = float(np.trapz(np.array(vmr_ratio_overall) - 1.0, log_scales))
        ie_stratum = float(np.trapz(np.array(vmr_ratio_stratum) - 1.0, log_scales))

        result = {
            'cell_type': cell_type,
            'n_cells': n_type,
            'chi2_overall': float(chi2_overall),
            'p_overall': float(test_overall['p_value']),
            'chi2_within_stratum': float(chi2_stratum),
            'p_within_stratum': float(test_stratum['p_value']),
            'mediation_pct': float(mediation_pct),
            'peak_vmr_ratio_overall': float(peak_overall),
            'peak_vmr_ratio_within_stratum': float(peak_stratum),
            'integrated_excess_overall': ie_overall,
            'integrated_excess_within_stratum': ie_stratum,
            'scales_um': [float(s) for s in real_curve['scales_um']],
            'vmr_ratio_overall': vmr_ratio_overall,
            'vmr_ratio_within_stratum': vmr_ratio_stratum,
            'n_mocks_overall': len(overall_vmrs),
            'n_mocks_stratum': len(stratum_vmrs),
        }
        results.append(result)

        logger.info(
            f'[{i+1}/{len(valid_types)}] {cell_type}: '
            f'chi2_overall={chi2_overall:.1f}, '
            f'chi2_stratum={chi2_stratum:.1f}, '
            f'mediation={mediation_pct:.1f}%'
        )

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Tissue-scale mediation analysis')
    parser.add_argument('--data-dir', default='data', help='Data directory')
    parser.add_argument('--results-dir', default='results', help='Results directory')
    parser.add_argument('--regions', nargs='+', default=['isocortex', 'thalamus'])
    parser.add_argument('--n-mocks', type=int, default=50)
    parser.add_argument('--min-cells', type=int, default=50)
    parser.add_argument('--grid-size', type=int, default=256)
    parser.add_argument('--type-level', default='class',
                        choices=['class', 'subclass', 'supertype'])
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(name)s: %(message)s',
    )

    data_dir = Path(args.data_dir)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)

    for region in args.regions:
        logger.info(f'\n{"="*60}')
        logger.info(f'REGION: {region}')
        logger.info(f'{"="*60}')

        # Load and augment data
        parquet_path = data_dir / f'{region}_cells.parquet'
        if not parquet_path.exists():
            logger.warning(f'Parquet not found: {parquet_path}, skipping')
            continue

        cells = augment_with_parcellation(parquet_path, region)

        # Determine stratum column
        if region == 'isocortex':
            stratum_col = 'layer'
        elif region == 'thalamus':
            stratum_col = 'nucleus'
        else:
            logger.warning(f'No stratum defined for region "{region}", skipping')
            continue

        # Filter to cells with valid stratum
        valid_mask = cells[stratum_col].notna()
        cells_valid = cells[valid_mask]

        # Print stratum distribution
        stratum_counts = cells_valid[stratum_col].value_counts()
        logger.info(f'Stratum distribution:\n{stratum_counts}')

        # Prepare analysis arrays
        positions = cells_valid[['x', 'y']].values.astype(np.float64)
        if positions[:, 0].max() < 20:  # mm -> um conversion
            positions = positions * 1000.0

        labels = cells_valid[args.type_level].values.astype(str)
        strata = cells_valid[stratum_col].values.astype(str)

        # Run mediation
        results = compute_mediation(
            positions, labels, strata,
            n_mocks=args.n_mocks,
            min_cells=args.min_cells,
            grid_size=args.grid_size,
            rng=rng,
        )

        # Save results
        out_path = results_dir / f'{region}_mediation.json'
        with open(out_path, 'w') as f:
            json.dump({
                'region': region,
                'stratum_variable': stratum_col,
                'type_level': args.type_level,
                'n_mocks': args.n_mocks,
                'n_strata': int(stratum_counts.nunique()),
                'strata': stratum_counts.to_dict(),
                'results': results,
                'summary': {
                    'n_types_analyzed': len(results),
                    'mean_mediation_pct': float(np.mean([r['mediation_pct'] for r in results])) if results else 0,
                    'median_mediation_pct': float(np.median([r['mediation_pct'] for r in results])) if results else 0,
                },
            }, f, indent=2)
        logger.info(f'Saved results to {out_path}')

        # Print summary
        if results:
            mediations = [r['mediation_pct'] for r in results]
            logger.info(f'\n--- MEDIATION SUMMARY ({region}) ---')
            logger.info(f'Mean mediation: {np.mean(mediations):.1f}%')
            logger.info(f'Median mediation: {np.median(mediations):.1f}%')
            logger.info(f'Range: {min(mediations):.1f}% - {max(mediations):.1f}%')
            for r in sorted(results, key=lambda x: x['mediation_pct'], reverse=True):
                logger.info(f"  {r['cell_type']}: {r['mediation_pct']:.1f}% "
                            f"(chi2: {r['chi2_overall']:.0f} -> {r['chi2_within_stratum']:.0f})")

        # Also save summary CSV
        if results:
            summary_df = pd.DataFrame([{
                'cell_type': r['cell_type'],
                'n_cells': r['n_cells'],
                'mediation_pct': r['mediation_pct'],
                'chi2_overall': r['chi2_overall'],
                'chi2_within_stratum': r['chi2_within_stratum'],
                'p_overall': r['p_overall'],
                'p_within_stratum': r['p_within_stratum'],
                'peak_vmr_ratio_overall': r['peak_vmr_ratio_overall'],
                'peak_vmr_ratio_within_stratum': r['peak_vmr_ratio_within_stratum'],
                'integrated_excess_overall': r['integrated_excess_overall'],
                'integrated_excess_within_stratum': r['integrated_excess_within_stratum'],
            } for r in results])
            csv_path = results_dir / f'{region}_mediation_summary.csv'
            summary_df.to_csv(csv_path, index=False)
            logger.info(f'Saved summary CSV to {csv_path}')


if __name__ == '__main__':
    main()
