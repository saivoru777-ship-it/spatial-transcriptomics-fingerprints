"""2D and 3D multiscale spatial statistics for tissue sections.

Computes VMR (Variance-to-Mean Ratio) and skewness across spatial scales
using grid-based binning. Adapted from smlm-clustering/core/multiscale_detector.py
(3D CIC gridding -> 2D histogram) and neurostat/core/chi_squared.py (Hotelling's T²).

For tissue sections, cells are discrete objects on a 2D plane. We use simple
histogram binning rather than CIC interpolation, since cells are larger objects
(~10μm) rather than point localizations.

The 3D extension (TissueMultiscaleTest3D) uses np.histogramdd for volumetric
analysis, needed for regions like thalamus where z-lamination is prominent.
"""

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from scipy import stats


@dataclass
class ScaleRange:
    """Physical scale range for multiscale analysis.

    Attributes
    ----------
    cell_sizes_vox : list of int
        Bin sizes in grid voxels.
    cell_sizes_um : list of float
        Corresponding physical sizes in micrometers.
    """
    cell_sizes_vox: List[int] = field(default_factory=list)
    cell_sizes_um: List[float] = field(default_factory=list)

    @staticmethod
    def for_tissue(min_um=10, max_um=1000, n_scales=12,
                   roi_size_um=None, grid_size=256):
        """Create a logarithmic scale range for tissue analysis.

        Generates n_scales candidates, converts to voxel sizes, then
        deduplicates so every scale produces a distinct grid cell size.
        The minimum resolvable scale is 2 * (roi_size / grid_size).

        Parameters
        ----------
        min_um, max_um : float
            Physical scale bounds in micrometers.
        n_scales : int
            Target number of scales (actual may be fewer after dedup).
        roi_size_um : float, optional
            Physical ROI size. Required to compute voxel sizes.
        grid_size : int
            Grid resolution.

        Returns
        -------
        ScaleRange
        """
        # Over-sample candidates to get enough unique voxel sizes
        n_candidates = n_scales * 4
        candidate_um = np.logspace(
            np.log10(min_um), np.log10(max_um), n_candidates
        )

        if roi_size_um is not None:
            um_per_vox = roi_size_um / grid_size

            # Enforce minimum resolvable scale
            min_resolvable = 2 * um_per_vox
            candidate_um = candidate_um[candidate_um >= min_resolvable * 0.9]

            # Convert to voxel sizes and deduplicate
            seen_vox = set()
            cell_sizes_vox = []
            cell_sizes_um = []
            for s_um in candidate_um:
                vox = max(2, int(round(s_um / um_per_vox)))
                if vox not in seen_vox and vox <= grid_size // 2:
                    seen_vox.add(vox)
                    cell_sizes_vox.append(vox)
                    cell_sizes_um.append(vox * um_per_vox)  # exact physical size

            # Subsample to n_scales if we have too many
            if len(cell_sizes_vox) > n_scales:
                indices = np.round(np.linspace(
                    0, len(cell_sizes_vox) - 1, n_scales
                )).astype(int)
                cell_sizes_vox = [cell_sizes_vox[i] for i in indices]
                cell_sizes_um = [cell_sizes_um[i] for i in indices]
        else:
            cell_sizes_vox = np.unique(np.logspace(
                np.log10(2), np.log10(max(64, n_scales * 4)),
                n_scales
            ).astype(int)).tolist()
            cell_sizes_um = [float(v) for v in cell_sizes_vox]

        return ScaleRange(
            cell_sizes_vox=cell_sizes_vox,
            cell_sizes_um=cell_sizes_um,
        )

    @staticmethod
    def logarithmic(min_vox=2, max_vox=64, n_scales=12):
        """Create a scale range in grid units."""
        cell_sizes_vox = np.unique(np.logspace(
            np.log10(min_vox), np.log10(max_vox), n_scales
        ).astype(int)).tolist()
        return ScaleRange(
            cell_sizes_vox=cell_sizes_vox,
            cell_sizes_um=[float(v) for v in cell_sizes_vox],
        )

    def __len__(self):
        return len(self.cell_sizes_vox)


class TissueMultiscaleTest:
    """Multiscale VMR/skewness test for 2D tissue cell positions.

    Bins cell positions onto a 2D grid, then measures count statistics
    (VMR, skewness) at multiple coarse-graining scales. Significance is
    assessed via covariance-aware chi-squared (Hotelling's T²) against
    null model realizations.

    Parameters
    ----------
    roi_size_um : array-like, shape (2,)
        Physical ROI size [width, height] in micrometers.
    grid_size : int
        Grid resolution (pixels per side).
    roi_origin : array-like, shape (2,), optional
        Physical (x, y) of the ROI origin. Default (0, 0).
    """

    def __init__(self, roi_size_um, grid_size=256, roi_origin=None):
        self.roi_size_um = np.asarray(roi_size_um, dtype=float)
        if self.roi_size_um.ndim == 0:
            self.roi_size_um = np.array([self.roi_size_um, self.roi_size_um])
        self.grid_size = grid_size
        self.roi_origin = (np.asarray(roi_origin, dtype=float)
                           if roi_origin is not None
                           else np.zeros(2))
        self.um_per_vox = self.roi_size_um / grid_size

    @classmethod
    def from_positions(cls, positions, grid_size=256, padding=0.01,
                       grid_offset=None):
        """Create a test from data bounds.

        Parameters
        ----------
        positions : array-like, shape (N, 2)
            Cell (x, y) coordinates in micrometers.
        grid_size : int
            Grid resolution.
        padding : float
            Fractional padding beyond data extent.
        grid_offset : array-like, shape (2,), optional
            Shift the ROI origin by this amount (in μm). Used for
            grid jitter robustness testing — shifting the grid origin
            by a random fraction of cell_size tests whether results
            depend on arbitrary grid placement.

        Returns
        -------
        TissueMultiscaleTest
        """
        positions = np.asarray(positions, dtype=float)
        mins = positions.min(axis=0)
        maxs = positions.max(axis=0)
        span = maxs - mins
        pad = span * padding
        roi_origin = mins - pad
        roi_size = span + 2 * pad
        if grid_offset is not None:
            roi_origin = roi_origin + np.asarray(grid_offset, dtype=float)
        return cls(roi_size, grid_size, roi_origin)

    def grid_positions(self, positions):
        """Bin positions onto a 2D density grid.

        Uses simple histogram binning (not CIC interpolation) since
        cells are discrete objects, not point localizations.

        Parameters
        ----------
        positions : array-like, shape (N, 2)
            Cell coordinates in micrometers.

        Returns
        -------
        grid : np.ndarray, shape (grid_size, grid_size)
            2D density field (counts per pixel).
        """
        positions = np.asarray(positions, dtype=float)

        # Normalize to grid coordinates
        pos_norm = (positions - self.roi_origin) / self.roi_size_um * self.grid_size
        pos_norm = np.clip(pos_norm, 0, self.grid_size - 1e-9)

        # 2D histogram
        grid, _, _ = np.histogram2d(
            pos_norm[:, 0], pos_norm[:, 1],
            bins=self.grid_size,
            range=[[0, self.grid_size], [0, self.grid_size]],
        )
        return grid

    def bin_counts(self, grid, cell_size):
        """Compute counts in non-overlapping cells of given size.

        Parameters
        ----------
        grid : np.ndarray, shape (grid_size, grid_size)
            Density grid.
        cell_size : int
            Cell size in grid voxels.

        Returns
        -------
        counts : np.ndarray
            Flattened array of cell counts.
        """
        gs = self.grid_size
        n_cells_x = gs // cell_size
        n_cells_y = gs // cell_size

        if n_cells_x < 2 or n_cells_y < 2:
            return np.array([grid.sum()])

        # Trim grid to exact multiple of cell_size
        trimmed = grid[:n_cells_x * cell_size, :n_cells_y * cell_size]

        # Reshape and sum within each cell
        counts = trimmed.reshape(
            n_cells_x, cell_size, n_cells_y, cell_size
        ).sum(axis=(1, 3))

        return counts.ravel()

    def variance_at_scale(self, grid, cell_size):
        """Compute VMR at a given scale.

        Parameters
        ----------
        grid : np.ndarray
            2D density grid.
        cell_size : int
            Cell size in grid voxels.

        Returns
        -------
        float
            Variance-to-Mean Ratio. VMR = 1 for Poisson (random),
            VMR > 1 for clustered, VMR < 1 for regular.
        """
        counts = self.bin_counts(grid, cell_size)
        mean = counts.mean()
        if mean < 1e-10:
            return 1.0
        return float(counts.var() / mean)

    def skewness_at_scale(self, grid, cell_size):
        """Compute skewness at a given scale.

        Parameters
        ----------
        grid : np.ndarray
            2D density grid.
        cell_size : int
            Cell size in grid voxels.

        Returns
        -------
        float
            Skewness of cell counts.
        """
        counts = self.bin_counts(grid, cell_size)
        if counts.std() < 1e-10:
            return 0.0
        return float(stats.skew(counts))

    def compute_curves(self, positions, scale_range):
        """Compute VMR and skewness curves across scales.

        Parameters
        ----------
        positions : array-like, shape (N, 2)
            Cell coordinates.
        scale_range : ScaleRange
            Scales to evaluate.

        Returns
        -------
        dict with:
            'vmr' : list of float — VMR at each scale
            'skewness' : list of float — skewness at each scale
            'scales_um' : list of float — physical scales
            'scales_vox' : list of int — grid scales
            'n_points' : int — number of positions
        """
        grid = self.grid_positions(positions)

        vmr_curve = []
        skew_curve = []
        for cs in scale_range.cell_sizes_vox:
            vmr_curve.append(self.variance_at_scale(grid, cs))
            skew_curve.append(self.skewness_at_scale(grid, cs))

        return {
            'vmr': vmr_curve,
            'skewness': skew_curve,
            'scales_um': list(scale_range.cell_sizes_um),
            'scales_vox': list(scale_range.cell_sizes_vox),
            'n_points': len(positions),
        }

    def test(self, real_positions, mock_positions_list, scale_range,
             shrinkage=0.02):
        """Full multiscale test: compute curves + chi-squared significance.

        Parameters
        ----------
        real_positions : array-like, shape (N, 2)
            Observed cell positions.
        mock_positions_list : list of array-like
            Each element is (N, 2) mock positions from a null model.
        scale_range : ScaleRange
            Scales to evaluate.
        shrinkage : float
            Covariance shrinkage parameter.

        Returns
        -------
        dict with:
            'real_curve' : dict — VMR/skewness from real data
            'mock_curves' : list of dict — from each mock
            'chi_squared' : float
            'f_statistic' : float
            'p_value' : float
            'dof' : int
            'condition_number' : float
            'n_mocks_used' : int
        """
        real_curve = self.compute_curves(real_positions, scale_range)

        mock_curves = []
        for mock_pos in mock_positions_list:
            mc = self.compute_curves(mock_pos, scale_range)
            mock_curves.append(mc)

        # Build matrices for chi-squared test
        real_vmr = np.array(real_curve['vmr'])
        mock_vmr = np.array([mc['vmr'] for mc in mock_curves])

        result = chi_squared_with_covariance(real_vmr, mock_vmr, shrinkage)
        result['real_curve'] = real_curve
        result['mock_curves'] = mock_curves
        return result


class TissueMultiscaleTest3D:
    """3D multiscale VMR/skewness test for volumetric tissue data.

    Extends TissueMultiscaleTest to 3D using np.histogramdd. Needed for
    brain regions where z-lamination is prominent (e.g., thalamus).
    """

    def __init__(self, roi_size_um, grid_size=64, roi_origin=None):
        self.roi_size_um = np.asarray(roi_size_um, dtype=float)
        if self.roi_size_um.ndim == 0:
            self.roi_size_um = np.array([self.roi_size_um] * 3)
        self.grid_size = grid_size
        self.roi_origin = (np.asarray(roi_origin, dtype=float)
                           if roi_origin is not None
                           else np.zeros(3))
        self.um_per_vox = self.roi_size_um / grid_size

    @classmethod
    def from_positions(cls, positions, grid_size=64, padding=0.01):
        """Create from 3D data bounds."""
        positions = np.asarray(positions, dtype=float)
        mins = positions.min(axis=0)
        maxs = positions.max(axis=0)
        span = maxs - mins
        pad = span * padding
        roi_origin = mins - pad
        roi_size = span + 2 * pad
        return cls(roi_size, grid_size, roi_origin)

    def grid_positions(self, positions):
        """Bin positions onto a 3D density grid."""
        positions = np.asarray(positions, dtype=float)
        pos_norm = ((positions - self.roi_origin) / self.roi_size_um
                    * self.grid_size)
        pos_norm = np.clip(pos_norm, 0, self.grid_size - 1e-9)
        grid, _ = np.histogramdd(
            pos_norm,
            bins=self.grid_size,
            range=[[0, self.grid_size]] * 3,
        )
        return grid

    def bin_counts(self, grid, cell_size):
        """Compute counts in non-overlapping 3D cells."""
        gs = self.grid_size
        nc = gs // cell_size
        if nc < 2:
            return np.array([grid.sum()])
        trimmed = grid[:nc * cell_size, :nc * cell_size, :nc * cell_size]
        counts = trimmed.reshape(
            nc, cell_size, nc, cell_size, nc, cell_size
        ).sum(axis=(1, 3, 5))
        return counts.ravel()

    def variance_at_scale(self, grid, cell_size):
        counts = self.bin_counts(grid, cell_size)
        mean = counts.mean()
        if mean < 1e-10:
            return 1.0
        return float(counts.var() / mean)

    def skewness_at_scale(self, grid, cell_size):
        counts = self.bin_counts(grid, cell_size)
        if counts.std() < 1e-10:
            return 0.0
        return float(stats.skew(counts))

    def compute_curves(self, positions, scale_range):
        """Compute VMR and skewness curves across scales (3D)."""
        grid = self.grid_positions(positions)
        vmr_curve = []
        skew_curve = []
        for cs in scale_range.cell_sizes_vox:
            vmr_curve.append(self.variance_at_scale(grid, cs))
            skew_curve.append(self.skewness_at_scale(grid, cs))
        return {
            'vmr': vmr_curve,
            'skewness': skew_curve,
            'scales_um': list(scale_range.cell_sizes_um),
            'scales_vox': list(scale_range.cell_sizes_vox),
            'n_points': len(positions),
        }


def chi_squared_with_covariance(real_values, mock_matrix, shrinkage=0.02):
    """Covariance-aware chi-squared test with Hotelling's T² correction.

    Identical to neurostat.core.chi_squared.chi_squared_with_covariance.
    Included here for self-containment (no neurostat dependency required).

    Parameters
    ----------
    real_values : array-like, shape (n_scales,)
        Observed statistic values across scales.
    mock_matrix : array-like, shape (n_mocks, n_scales)
        Null distribution: each row is one mock's values across scales.
    shrinkage : float
        Regularization strength in [0, 1].

    Returns
    -------
    dict with chi_squared, f_statistic, dof, dof_f, p_value,
    condition_number, n_mocks_used
    """
    real_vals = np.asarray(real_values, dtype=float)
    mock_mat = np.asarray(mock_matrix, dtype=float)

    n_scales = len(real_vals)

    if mock_mat.ndim == 1:
        mock_mat = mock_mat.reshape(1, -1)

    # Filter valid mocks
    valid = [mock_mat[i] for i in range(len(mock_mat))
             if len(mock_mat[i]) == n_scales]
    n_mocks = len(valid)

    if n_mocks < 2:
        return {
            'chi_squared': 0.0, 'f_statistic': 0.0,
            'dof': n_scales, 'dof_f': (n_scales, 0),
            'p_value': 1.0, 'condition_number': np.inf,
            'n_mocks_used': n_mocks,
        }

    mock_mat = np.array(valid)
    mock_mean = mock_mat.mean(axis=0)
    mock_cov = np.cov(mock_mat.T)

    if mock_cov.ndim == 0:
        mock_cov = np.atleast_2d(mock_cov)

    # Ledoit-Wolf style shrinkage toward diagonal
    diag_cov = np.diag(np.diag(mock_cov))
    mock_cov_reg = (1 - shrinkage) * mock_cov + shrinkage * diag_cov
    mock_cov_reg += np.eye(n_scales) * 1e-10

    residual = real_vals - mock_mean

    try:
        cov_inv = np.linalg.inv(mock_cov_reg)
        chi_sq = float(n_mocks / (n_mocks + 1) * residual @ cov_inv @ residual)
        cond = float(np.linalg.cond(mock_cov_reg))
    except np.linalg.LinAlgError:
        var_diag = np.diag(mock_cov) + 1e-10
        chi_sq = float(n_mocks / (n_mocks + 1) * (residual**2 / var_diag).sum())
        cond = np.inf

    # Hotelling's T² -> F correction
    p = n_scales
    n = n_mocks
    df1 = p
    df2 = n - p

    if df2 > 0:
        f_stat = chi_sq * df2 / ((n - 1) * p)
        p_value = float(1 - stats.f.cdf(f_stat, df1, df2))
    else:
        f_stat = chi_sq / p
        p_value = float(1 - stats.chi2.cdf(chi_sq, p))

    return {
        'chi_squared': chi_sq,
        'f_statistic': float(f_stat),
        'dof': n_scales,
        'dof_f': (df1, df2),
        'p_value': p_value,
        'condition_number': cond,
        'n_mocks_used': n_mocks,
    }
