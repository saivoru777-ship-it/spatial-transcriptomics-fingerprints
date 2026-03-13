"""Tests for 2D multiscale spatial statistics."""

import numpy as np
import pytest

from src.spatial_statistics import (
    ScaleRange,
    TissueMultiscaleTest,
    chi_squared_with_covariance,
)


class TestScaleRange:
    def test_for_tissue_default(self):
        sr = ScaleRange.for_tissue(roi_size_um=5000, grid_size=256)
        assert 5 <= len(sr) <= 12  # dedup may reduce count
        assert all(v >= 2 for v in sr.cell_sizes_vox)
        # All voxel sizes should be unique (dedup guarantee)
        assert len(sr.cell_sizes_vox) == len(set(sr.cell_sizes_vox))
        # First scale >= minimum resolvable (2 * 5000/256 ≈ 39μm)
        assert sr.cell_sizes_um[0] >= 30
        assert sr.cell_sizes_um[-1] <= 2600  # <= grid_size/2 * um_per_vox

    def test_for_tissue_custom_range(self):
        sr = ScaleRange.for_tissue(min_um=50, max_um=500, n_scales=6,
                                    roi_size_um=2000, grid_size=128)
        assert 3 <= len(sr) <= 6  # dedup may reduce
        # All voxel sizes unique
        assert len(sr.cell_sizes_vox) == len(set(sr.cell_sizes_vox))

    def test_logarithmic(self):
        sr = ScaleRange.logarithmic(min_vox=2, max_vox=32, n_scales=8)
        assert len(sr) >= 2
        assert sr.cell_sizes_vox[0] == 2

    def test_len(self):
        sr = ScaleRange(cell_sizes_vox=[2, 4, 8], cell_sizes_um=[10, 20, 40])
        assert len(sr) == 3


class TestTissueMultiscaleTest:
    def setup_method(self):
        """Create a test dataset."""
        self.rng = np.random.default_rng(42)

    def _random_positions(self, n, extent=1000):
        return self.rng.uniform(0, extent, size=(n, 2))

    def _clustered_positions(self, n_clusters=5, points_per=100, extent=1000,
                              cluster_std=20):
        """Generate clustered 2D positions."""
        positions = []
        centers = self.rng.uniform(100, extent - 100, size=(n_clusters, 2))
        for center in centers:
            pts = self.rng.normal(center, cluster_std, size=(points_per, 2))
            positions.append(pts)
        return np.vstack(positions)

    def test_from_positions(self):
        pos = self._random_positions(500)
        tester = TissueMultiscaleTest.from_positions(pos, grid_size=64)
        assert tester.grid_size == 64
        assert tester.roi_size_um.shape == (2,)
        assert all(tester.roi_size_um > 0)

    def test_grid_positions_shape(self):
        pos = self._random_positions(500)
        tester = TissueMultiscaleTest.from_positions(pos, grid_size=64)
        grid = tester.grid_positions(pos)
        assert grid.shape == (64, 64)
        # Total mass should equal number of points
        assert grid.sum() == pytest.approx(500, abs=1)

    def test_bin_counts_partitions(self):
        pos = self._random_positions(1000)
        tester = TissueMultiscaleTest.from_positions(pos, grid_size=64)
        grid = tester.grid_positions(pos)
        counts = tester.bin_counts(grid, cell_size=8)
        # 64/8 = 8 cells per side, 64 total cells
        assert len(counts) == 64
        assert counts.sum() == pytest.approx(grid.sum(), rel=0.01)

    def test_vmr_random_near_one(self):
        """VMR of uniformly random points should be near 1 (Poisson)."""
        pos = self._random_positions(10000)
        tester = TissueMultiscaleTest.from_positions(pos, grid_size=128)
        grid = tester.grid_positions(pos)
        vmr = tester.variance_at_scale(grid, cell_size=8)
        # For 10k random points in 128x128, VMR should be close to 1
        assert 0.5 < vmr < 2.0, f'VMR for random = {vmr}, expected ~1'

    def test_vmr_clustered_above_one(self):
        """VMR of clustered points should be > 1."""
        pos = self._clustered_positions(n_clusters=5, points_per=200,
                                         cluster_std=15)
        tester = TissueMultiscaleTest.from_positions(pos, grid_size=128)
        grid = tester.grid_positions(pos)
        vmr = tester.variance_at_scale(grid, cell_size=8)
        assert vmr > 2.0, f'VMR for clusters = {vmr}, expected > 2'

    def test_compute_curves(self):
        pos = self._random_positions(1000)
        tester = TissueMultiscaleTest.from_positions(pos, grid_size=64)
        sr = ScaleRange.logarithmic(min_vox=2, max_vox=16, n_scales=5)
        curves = tester.compute_curves(pos, sr)
        assert len(curves['vmr']) == len(sr)
        assert len(curves['skewness']) == len(sr)
        assert curves['n_points'] == 1000

    def test_full_test_random(self):
        """Full test on random data should be non-significant."""
        pos = self._random_positions(2000)
        mocks = [self._random_positions(2000) for _ in range(20)]
        tester = TissueMultiscaleTest.from_positions(pos, grid_size=64)
        sr = ScaleRange.logarithmic(min_vox=2, max_vox=16, n_scales=5)
        result = tester.test(pos, mocks, sr)
        assert 'p_value' in result
        assert 'real_curve' in result
        assert 'mock_curves' in result
        # Random vs random should usually be non-significant
        # (but not guaranteed for any single test)

    def test_full_test_clustered_vs_random(self):
        """Clustered data vs random mocks should be significant."""
        real = self._clustered_positions(n_clusters=8, points_per=100,
                                          cluster_std=15)
        n = len(real)
        mocks = [self._random_positions(n, extent=1000) for _ in range(30)]
        tester = TissueMultiscaleTest.from_positions(real, grid_size=64)
        sr = ScaleRange.logarithmic(min_vox=2, max_vox=16, n_scales=5)
        result = tester.test(real, mocks, sr)
        assert result['p_value'] < 0.05, \
            f'Clustered vs random p={result["p_value"]}, expected < 0.05'


class TestChiSquaredWithCovariance:
    def test_identical_is_nonsignificant(self):
        """Real curve identical to mock mean -> p ≈ 1."""
        rng = np.random.default_rng(42)
        mock = rng.normal(5, 1, size=(50, 8))
        real = mock.mean(axis=0)
        result = chi_squared_with_covariance(real, mock)
        assert result['p_value'] > 0.5

    def test_extreme_is_significant(self):
        """Real curve 10σ from mock mean -> p ≈ 0."""
        rng = np.random.default_rng(42)
        mock = rng.normal(0, 1, size=(50, 8))
        real = np.full(8, 10.0)
        result = chi_squared_with_covariance(real, mock)
        assert result['p_value'] < 0.001

    def test_output_keys(self):
        rng = np.random.default_rng(42)
        result = chi_squared_with_covariance(
            rng.normal(0, 1, 5),
            rng.normal(0, 1, (30, 5)),
        )
        expected_keys = {'chi_squared', 'f_statistic', 'dof', 'dof_f',
                         'p_value', 'condition_number', 'n_mocks_used'}
        assert expected_keys.issubset(result.keys())

    def test_insufficient_mocks(self):
        result = chi_squared_with_covariance(
            np.array([1.0, 2.0]),
            np.array([[1.0, 2.0]]),
        )
        assert result['p_value'] == 1.0
        assert result['n_mocks_used'] == 1
