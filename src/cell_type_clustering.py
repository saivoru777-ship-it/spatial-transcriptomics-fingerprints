"""Per-type and cross-type spatial clustering analysis.

Computes multiscale spatial fingerprints for individual cell types and
pairwise co-occurrence patterns. Uses label-shuffle as the primary null
to test whether type-to-position mappings are non-random.

Three analysis modes:
1. Per-type fingerprints — VMR curve per cell type vs label-shuffle null
2. Cross-type co-occurrence — Pearson correlation of type densities per scale
3. Regional comparison — aggregate fingerprints across brain regions
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .null_models import LabelShuffleNull
from .spatial_statistics import ScaleRange, TissueMultiscaleTest, chi_squared_with_covariance

logger = logging.getLogger(__name__)


@dataclass
class TypeFingerprint:
    """Multiscale spatial fingerprint for a single cell type."""
    cell_type: str
    n_cells: int
    vmr_curve: List[float]
    skewness_curve: List[float]
    scales_um: List[float]
    mock_vmr_mean: List[float]
    mock_vmr_std: List[float]
    chi_squared: float
    f_statistic: float
    p_value: float
    peak_scale_um: float
    condition_number: float
    n_mocks_used: int

    @property
    def is_significant(self):
        return self.p_value < 0.05

    @property
    def peak_vmr_ratio(self):
        """Max ratio of real VMR to null mean VMR."""
        ratios = self.vmr_ratio_curve
        return max(ratios) if ratios else 1.0

    @property
    def vmr_ratio_curve(self):
        """VMR / null_mean at each scale."""
        return [r / m if m > 0 else 1.0
                for r, m in zip(self.vmr_curve, self.mock_vmr_mean)]

    @property
    def integrated_excess(self):
        """Area under (VMR_ratio - 1) curve (log-scale trapezoid).

        Measures total clustering strength across all scales.
        Positive = net clustering, negative = net regularity.
        """
        ratios = np.array(self.vmr_ratio_curve)
        log_scales = np.log10(self.scales_um)
        return float(np.trapz(ratios - 1.0, log_scales))

    @property
    def fingerprint_width(self):
        """Scale range (octaves) over which VMR ratio > halfway to peak.

        Narrow = sharp scale-specific peak. Wide = broad clustering.
        """
        ratios = np.array(self.vmr_ratio_curve)
        peak = ratios.max()
        if peak <= 1.05:
            return 0.0
        threshold = 1.0 + (peak - 1.0) * 0.5
        above = ratios >= threshold
        if not above.any():
            return 0.0
        log_s = np.log10(self.scales_um)
        indices = np.where(above)[0]
        return float(log_s[indices[-1]] - log_s[indices[0]])

    @property
    def clustering_direction(self):
        """'clustered', 'regular', or 'mixed' based on VMR ratio curve."""
        ratios = np.array(self.vmr_ratio_curve)
        above = (ratios > 1.05).sum()
        below = (ratios < 0.95).sum()
        if above > 0 and below > 0:
            return 'mixed'
        elif below > above:
            return 'regular'
        else:
            return 'clustered'


@dataclass
class CoOccurrenceResult:
    """Cross-type co-occurrence across scales."""
    type_a: str
    type_b: str
    scales_um: List[float]
    correlation_curve: List[float]
    null_corr_mean: List[float]
    null_corr_std: List[float]
    peak_scale_um: float
    peak_correlation: float
    significant_scales: List[bool]

    @property
    def max_positive_correlation(self):
        return float(max(self.correlation_curve))

    @property
    def max_negative_correlation(self):
        return float(min(self.correlation_curve))

    @property
    def sign_change_scale(self):
        """Scale at which correlation changes sign, or None."""
        corr = np.array(self.correlation_curve)
        for i in range(len(corr) - 1):
            if corr[i] * corr[i + 1] < 0:
                return self.scales_um[i]
        return None

    @property
    def integrated_colocalization(self):
        """Integrated correlation across log-scales."""
        corr = np.array(self.correlation_curve)
        log_s = np.log10(self.scales_um)
        return float(np.trapz(corr, log_s))

    @property
    def excess_over_null(self):
        """Max |correlation - null_mean| in units of null_std."""
        corr = np.array(self.correlation_curve)
        null_m = np.array(self.null_corr_mean)
        null_s = np.array(self.null_corr_std)
        null_s = np.where(null_s > 1e-10, null_s, 1e-10)
        return float(np.max(np.abs(corr - null_m) / null_s))


def compute_per_type_fingerprints(
    positions,
    labels,
    scale_range=None,
    n_mocks=50,
    min_cells=50,
    grid_size=256,
    rng=None,
):
    """Compute multiscale fingerprints for each cell type.

    For each type with sufficient cells:
    1. Extract positions of that type
    2. Compute VMR curve across scales
    3. Generate label-shuffle mocks
    4. Run covariance-aware chi-squared
    5. Record fingerprint shape, peak scale, significance

    Parameters
    ----------
    positions : np.ndarray, shape (N, 2)
        All cell positions.
    labels : np.ndarray, shape (N,)
        Cell type labels.
    scale_range : ScaleRange, optional
        Scales to evaluate. Default: 10-1000μm, 12 scales.
    n_mocks : int
        Number of label-shuffle realizations.
    min_cells : int
        Minimum cells for a type to be analyzed.
    grid_size : int
        Grid resolution for density computation.
    rng : np.random.Generator, optional

    Returns
    -------
    fingerprints : dict of str -> TypeFingerprint
        Fingerprint per cell type.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    positions = np.asarray(positions, dtype=float)
    labels = np.asarray(labels)

    # Set up test
    tester = TissueMultiscaleTest.from_positions(positions, grid_size)

    if scale_range is None:
        roi_size = float(tester.roi_size_um.max())
        scale_range = ScaleRange.for_tissue(
            min_um=10, max_um=min(1000, roi_size * 0.4),
            n_scales=12, roi_size_um=roi_size, grid_size=grid_size,
        )

    # Set up label-shuffle null
    shuffler = LabelShuffleNull()
    shuffler.fit(positions, labels)
    shuffled_labels_list = shuffler.generate_shuffles(n_mocks, rng)

    # Get unique types with enough cells
    unique_types, type_counts = np.unique(labels, return_counts=True)
    valid_types = unique_types[type_counts >= min_cells]
    logger.info(f'Analyzing {len(valid_types)} types with >= {min_cells} cells '
                f'(of {len(unique_types)} total)')

    fingerprints = {}

    for cell_type in valid_types:
        # Real data
        type_mask = labels == cell_type
        type_pos = positions[type_mask]
        real_curve = tester.compute_curves(type_pos, scale_range)

        # Mock data from label shuffles
        mock_vmr_list = []
        for shuffled in shuffled_labels_list:
            mock_pos = positions[shuffled == cell_type]
            if len(mock_pos) < 5:
                continue
            mc = tester.compute_curves(mock_pos, scale_range)
            mock_vmr_list.append(mc['vmr'])

        if len(mock_vmr_list) < 5:
            logger.warning(f'Type "{cell_type}": insufficient mocks '
                           f'({len(mock_vmr_list)}), skipping')
            continue

        # Chi-squared test
        real_vmr = np.array(real_curve['vmr'])
        mock_vmr_mat = np.array(mock_vmr_list)
        test_result = chi_squared_with_covariance(real_vmr, mock_vmr_mat)

        mock_mean = mock_vmr_mat.mean(axis=0).tolist()
        mock_std = mock_vmr_mat.std(axis=0).tolist()

        # Peak scale = scale with maximum VMR excess over null
        excess = real_vmr - mock_vmr_mat.mean(axis=0)
        peak_idx = int(np.argmax(excess))

        fp = TypeFingerprint(
            cell_type=cell_type,
            n_cells=int(type_mask.sum()),
            vmr_curve=real_curve['vmr'],
            skewness_curve=real_curve['skewness'],
            scales_um=real_curve['scales_um'],
            mock_vmr_mean=mock_mean,
            mock_vmr_std=mock_std,
            chi_squared=test_result['chi_squared'],
            f_statistic=test_result['f_statistic'],
            p_value=test_result['p_value'],
            peak_scale_um=real_curve['scales_um'][peak_idx],
            condition_number=test_result['condition_number'],
            n_mocks_used=test_result['n_mocks_used'],
        )
        fingerprints[cell_type] = fp

        sig = '*' if fp.is_significant else ''
        logger.info(f'  {cell_type}: n={fp.n_cells}, '
                    f'peak={fp.peak_scale_um:.0f}μm, '
                    f'p={fp.p_value:.4f}{sig}')

    n_sig = sum(1 for fp in fingerprints.values() if fp.is_significant)
    logger.info(f'Significant: {n_sig}/{len(fingerprints)} types')
    return fingerprints


def compute_cross_type_cooccurrence(
    positions,
    labels,
    scale_range=None,
    n_mocks=50,
    min_cells=50,
    grid_size=256,
    type_pairs=None,
    rng=None,
):
    """Compute cross-type co-occurrence fingerprints.

    For each pair of cell types, at each scale:
    1. Count type A and type B per grid cell
    2. Compute Pearson correlation across grid cells
    3. Compare against label-shuffled null
    Positive = co-localization, negative = segregation.

    Parameters
    ----------
    positions : np.ndarray, shape (N, 2)
        All cell positions.
    labels : np.ndarray, shape (N,)
        Cell type labels.
    scale_range : ScaleRange, optional
    n_mocks : int
    min_cells : int
    grid_size : int
    type_pairs : list of (str, str), optional
        Specific pairs to analyze. Default: all pairs of valid types.
    rng : np.random.Generator, optional

    Returns
    -------
    results : dict of (str, str) -> CoOccurrenceResult
    """
    if rng is None:
        rng = np.random.default_rng(42)

    positions = np.asarray(positions, dtype=float)
    labels = np.asarray(labels)

    tester = TissueMultiscaleTest.from_positions(positions, grid_size)

    if scale_range is None:
        roi_size = float(tester.roi_size_um.max())
        scale_range = ScaleRange.for_tissue(
            min_um=10, max_um=min(1000, roi_size * 0.4),
            n_scales=12, roi_size_um=roi_size, grid_size=grid_size,
        )

    # Valid types
    unique_types, type_counts = np.unique(labels, return_counts=True)
    valid_types = sorted(unique_types[type_counts >= min_cells])

    if type_pairs is None:
        type_pairs = []
        for i, ta in enumerate(valid_types):
            for tb in valid_types[i + 1:]:
                type_pairs.append((ta, tb))

    logger.info(f'Computing co-occurrence for {len(type_pairs)} pairs')

    # Label shuffles
    shuffler = LabelShuffleNull()
    shuffler.fit(positions, labels)
    shuffled_labels_list = shuffler.generate_shuffles(n_mocks, rng)

    results = {}

    for type_a, type_b in type_pairs:
        corr_curve = []
        null_corr_curves = []

        for cs_idx, cs in enumerate(scale_range.cell_sizes_vox):
            # Real correlation
            corr_real = _type_pair_correlation(
                positions, labels, type_a, type_b, tester, cs
            )
            corr_curve.append(corr_real)

            # Null correlations
            null_corrs = []
            for shuffled in shuffled_labels_list:
                corr_null = _type_pair_correlation(
                    positions, shuffled, type_a, type_b, tester, cs
                )
                null_corrs.append(corr_null)
            null_corr_curves.append(null_corrs)

        corr_curve = np.array(corr_curve)
        null_arr = np.array(null_corr_curves)  # (n_scales, n_mocks)
        null_mean = null_arr.mean(axis=1)
        null_std = null_arr.std(axis=1)

        # Significance: |real - null_mean| > 2*null_std
        significant = np.abs(corr_curve - null_mean) > 2 * null_std

        # Peak co-occurrence
        abs_excess = np.abs(corr_curve - null_mean)
        peak_idx = int(np.argmax(abs_excess))

        results[(type_a, type_b)] = CoOccurrenceResult(
            type_a=type_a,
            type_b=type_b,
            scales_um=list(scale_range.cell_sizes_um),
            correlation_curve=corr_curve.tolist(),
            null_corr_mean=null_mean.tolist(),
            null_corr_std=null_std.tolist(),
            peak_scale_um=scale_range.cell_sizes_um[peak_idx],
            peak_correlation=float(corr_curve[peak_idx]),
            significant_scales=significant.tolist(),
        )

    n_any_sig = sum(1 for r in results.values() if any(r.significant_scales))
    logger.info(f'Pairs with any significant scale: {n_any_sig}/{len(results)}')
    return results


def _type_pair_correlation(positions, labels, type_a, type_b, tester, cell_size):
    """Pearson correlation of type A and B counts per grid cell at a given scale."""
    grid_a = tester.grid_positions(positions[labels == type_a])
    grid_b = tester.grid_positions(positions[labels == type_b])

    counts_a = tester.bin_counts(grid_a, cell_size)
    counts_b = tester.bin_counts(grid_b, cell_size)

    if len(counts_a) != len(counts_b):
        min_len = min(len(counts_a), len(counts_b))
        counts_a = counts_a[:min_len]
        counts_b = counts_b[:min_len]

    if counts_a.std() < 1e-10 or counts_b.std() < 1e-10:
        return 0.0

    corr = np.corrcoef(counts_a, counts_b)[0, 1]
    return float(corr) if np.isfinite(corr) else 0.0


def compute_regional_fingerprints(
    region_data,
    scale_range=None,
    n_mocks=50,
    grid_size=256,
    rng=None,
):
    """Compute aggregate VMR fingerprint per brain region.

    Parameters
    ----------
    region_data : dict of str -> (np.ndarray, np.ndarray)
        Map from region name to (positions, labels).
    scale_range : ScaleRange, optional
    n_mocks : int
    grid_size : int
    rng : np.random.Generator, optional

    Returns
    -------
    region_curves : dict of str -> dict
        Per-region VMR curves and null envelopes.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    region_curves = {}

    for region_name, (positions, labels) in region_data.items():
        logger.info(f'Computing regional fingerprint for {region_name} '
                    f'({len(positions):,} cells)')

        tester = TissueMultiscaleTest.from_positions(positions, grid_size)

        if scale_range is None:
            roi_size = float(tester.roi_size_um.max())
            sr = ScaleRange.for_tissue(
                min_um=10, max_um=min(1000, roi_size * 0.4),
                n_scales=12, roi_size_um=roi_size, grid_size=grid_size,
            )
        else:
            sr = scale_range

        # All-cells VMR (no type distinction)
        real_curve = tester.compute_curves(positions, sr)

        # CSR null for baseline
        from .null_models import TissueCSRNull
        csr = TissueCSRNull()
        csr.fit(positions)
        mock_curves = []
        for _ in range(n_mocks):
            mock_pos = csr.sample(len(positions), rng)
            mc = tester.compute_curves(mock_pos, sr)
            mock_curves.append(mc['vmr'])

        mock_arr = np.array(mock_curves)

        region_curves[region_name] = {
            'vmr': real_curve['vmr'],
            'scales_um': real_curve['scales_um'],
            'null_vmr_mean': mock_arr.mean(axis=0).tolist(),
            'null_vmr_std': mock_arr.std(axis=0).tolist(),
            'n_cells': len(positions),
        }

    return region_curves


def fingerprints_to_dataframe(fingerprints, sort_by='integrated_excess'):
    """Convert fingerprint dict to a summary DataFrame.

    Sorted by effect size (not p-value) by default.

    Parameters
    ----------
    fingerprints : dict of str -> TypeFingerprint
    sort_by : str
        Column to sort by. Default 'integrated_excess' (effect size).

    Returns
    -------
    pd.DataFrame
    """
    rows = []
    for ct, fp in fingerprints.items():
        rows.append({
            'cell_type': fp.cell_type,
            'n_cells': fp.n_cells,
            'peak_vmr_ratio': fp.peak_vmr_ratio,
            'peak_scale_um': fp.peak_scale_um,
            'integrated_excess': fp.integrated_excess,
            'fingerprint_width': fp.fingerprint_width,
            'direction': fp.clustering_direction,
            'p_value': fp.p_value,
            'is_significant': fp.is_significant,
        })
    df = pd.DataFrame(rows)
    return df.sort_values(sort_by, ascending=False, key=abs)


def cooccurrence_to_dataframe(results):
    """Convert co-occurrence results to effect-size-ranked DataFrame.

    Parameters
    ----------
    results : dict of (str, str) -> CoOccurrenceResult

    Returns
    -------
    pd.DataFrame
    """
    rows = []
    for (ta, tb), res in results.items():
        rows.append({
            'type_a': res.type_a,
            'type_b': res.type_b,
            'max_positive_corr': res.max_positive_correlation,
            'max_negative_corr': res.max_negative_correlation,
            'peak_correlation': res.peak_correlation,
            'peak_scale_um': res.peak_scale_um,
            'integrated_coloc': res.integrated_colocalization,
            'excess_over_null_sigma': res.excess_over_null,
            'sign_change_scale': res.sign_change_scale,
            'n_significant_scales': sum(res.significant_scales),
        })
    df = pd.DataFrame(rows)
    return df.sort_values('excess_over_null_sigma', ascending=False)


def compute_jittered_fingerprints(
    positions,
    labels,
    n_jitters=10,
    scale_range=None,
    n_mocks=50,
    min_cells=50,
    grid_size=256,
    rng=None,
):
    """Run fingerprint analysis with random grid offsets for robustness.

    Shifts the grid origin by a random fraction of the smallest cell size
    in each jitter trial, then computes per-type fingerprints. This tests
    whether results are stable to arbitrary grid placement.

    Parameters
    ----------
    positions : np.ndarray, shape (N, 2)
    labels : np.ndarray, shape (N,)
    n_jitters : int
        Number of random grid offset trials.
    scale_range : ScaleRange, optional
    n_mocks : int
    min_cells : int
    grid_size : int
    rng : np.random.Generator, optional

    Returns
    -------
    jitter_results : list of dict of str -> TypeFingerprint
        Fingerprints for each jitter trial.
    offsets : list of np.ndarray
        The (x, y) offsets used in each trial.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    positions = np.asarray(positions, dtype=float)
    labels = np.asarray(labels)

    # Determine max offset range from data extent
    span = positions.max(axis=0) - positions.min(axis=0)
    um_per_vox = span.max() / grid_size
    max_offset = um_per_vox * 2  # shift by up to 2 voxels

    jitter_results = []
    offsets = []

    for i in range(n_jitters):
        offset = rng.uniform(-max_offset, max_offset, size=2)
        offsets.append(offset)

        tester = TissueMultiscaleTest.from_positions(
            positions, grid_size, grid_offset=offset
        )

        if scale_range is None:
            roi_size = float(tester.roi_size_um.max())
            sr = ScaleRange.for_tissue(
                min_um=10, max_um=min(1000, roi_size * 0.4),
                n_scales=12, roi_size_um=roi_size, grid_size=grid_size,
            )
        else:
            sr = scale_range

        # Use a fresh shuffler each time with consistent mock labels
        shuffler = LabelShuffleNull()
        shuffler.fit(positions, labels)
        shuffled_labels_list = shuffler.generate_shuffles(n_mocks, rng)

        unique_types, type_counts = np.unique(labels, return_counts=True)
        valid_types = unique_types[type_counts >= min_cells]

        fingerprints = {}
        for cell_type in valid_types:
            type_mask = labels == cell_type
            type_pos = positions[type_mask]
            real_curve = tester.compute_curves(type_pos, sr)

            mock_vmr_list = []
            for shuffled in shuffled_labels_list:
                mock_pos = positions[shuffled == cell_type]
                if len(mock_pos) < 5:
                    continue
                mc = tester.compute_curves(mock_pos, sr)
                mock_vmr_list.append(mc['vmr'])

            if len(mock_vmr_list) < 5:
                continue

            real_vmr = np.array(real_curve['vmr'])
            mock_vmr_mat = np.array(mock_vmr_list)
            test_result = chi_squared_with_covariance(real_vmr, mock_vmr_mat)
            mock_mean = mock_vmr_mat.mean(axis=0).tolist()
            mock_std = mock_vmr_mat.std(axis=0).tolist()
            excess = real_vmr - mock_vmr_mat.mean(axis=0)
            peak_idx = int(np.argmax(excess))

            fingerprints[cell_type] = TypeFingerprint(
                cell_type=cell_type,
                n_cells=int(type_mask.sum()),
                vmr_curve=real_curve['vmr'],
                skewness_curve=real_curve['skewness'],
                scales_um=real_curve['scales_um'],
                mock_vmr_mean=mock_mean,
                mock_vmr_std=mock_std,
                chi_squared=test_result['chi_squared'],
                f_statistic=test_result['f_statistic'],
                p_value=test_result['p_value'],
                peak_scale_um=real_curve['scales_um'][peak_idx],
                condition_number=test_result['condition_number'],
                n_mocks_used=test_result['n_mocks_used'],
            )

        jitter_results.append(fingerprints)
        logger.info(f'  Jitter {i+1}/{n_jitters}: offset=({offset[0]:.1f}, '
                    f'{offset[1]:.1f})μm, {len(fingerprints)} types')

    return jitter_results, offsets


def cooccurrence_to_matrix(results, scale_idx=None):
    """Convert co-occurrence results to a symmetric correlation matrix.

    Parameters
    ----------
    results : dict of (str, str) -> CoOccurrenceResult
    scale_idx : int, optional
        Which scale to extract. Default: scale of peak correlation
        (per pair).

    Returns
    -------
    matrix : pd.DataFrame
        Symmetric correlation matrix.
    """
    types = sorted(set(t for pair in results.keys() for t in pair))
    n = len(types)
    type_to_idx = {t: i for i, t in enumerate(types)}
    mat = np.eye(n)

    for (ta, tb), res in results.items():
        if scale_idx is not None:
            val = res.correlation_curve[scale_idx]
        else:
            # Use peak scale for this pair
            abs_corr = np.abs(res.correlation_curve)
            val = res.correlation_curve[int(np.argmax(abs_corr))]
        i, j = type_to_idx[ta], type_to_idx[tb]
        mat[i, j] = val
        mat[j, i] = val

    return pd.DataFrame(mat, index=types, columns=types)
