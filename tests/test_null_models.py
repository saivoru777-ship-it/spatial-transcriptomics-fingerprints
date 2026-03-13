"""Tests for null models."""

import numpy as np
import pytest

from src.null_models import TissueCSRNull, DensityMatchedNull, LabelShuffleNull


class TestTissueCSRNull:
    def test_fit_and_sample(self):
        rng = np.random.default_rng(42)
        positions = rng.uniform(0, 1000, size=(500, 2))
        null = TissueCSRNull()
        null.fit(positions)
        samples = null.sample(200, rng)
        assert samples.shape == (200, 2)

    def test_samples_inside_hull(self):
        rng = np.random.default_rng(42)
        positions = rng.uniform(0, 1000, size=(500, 2))
        null = TissueCSRNull()
        null.fit(positions)
        samples = null.sample(1000, rng)
        inside = null._in_hull(samples)
        assert inside.all(), f'{(~inside).sum()} samples outside hull'

    def test_generate_mocks(self):
        rng = np.random.default_rng(42)
        positions = rng.uniform(0, 1000, size=(200, 2))
        null = TissueCSRNull()
        null.fit(positions)
        mocks = null.generate_mocks(100, 5, rng)
        assert len(mocks) == 5
        for m in mocks:
            assert m.shape == (100, 2)

    def test_hull_area(self):
        # Square: 4 corners
        positions = np.array([[0, 0], [100, 0], [100, 100], [0, 100]])
        null = TissueCSRNull()
        null.fit(positions)
        assert null.hull_area == pytest.approx(10000, rel=0.01)

    def test_acceptance_rate(self):
        rng = np.random.default_rng(42)
        # Roughly square region -> acceptance ~1
        positions = rng.uniform(0, 100, size=(200, 2))
        null = TissueCSRNull()
        null.fit(positions)
        null.sample(100, rng)
        assert null.acceptance_rate > 0.5


class TestDensityMatchedNull:
    def test_fit_and_sample(self):
        rng = np.random.default_rng(42)
        positions = rng.normal(500, 100, size=(1000, 2))
        null = DensityMatchedNull()
        null.fit(positions)
        samples = null.sample(200, rng)
        assert samples.shape == (200, 2)

    def test_density_preserved(self):
        """Samples should cluster near the data density."""
        rng = np.random.default_rng(42)
        # Bimodal: two clusters
        c1 = rng.normal([200, 200], 30, size=(500, 2))
        c2 = rng.normal([800, 800], 30, size=(500, 2))
        positions = np.vstack([c1, c2])

        null = DensityMatchedNull()
        null.fit(positions)
        samples = null.sample(1000, rng)

        # Samples should cluster around the two centers
        center_dist = np.minimum(
            np.linalg.norm(samples - [200, 200], axis=1),
            np.linalg.norm(samples - [800, 800], axis=1),
        )
        # Most samples within 200μm of a center
        close = (center_dist < 200).mean()
        assert close > 0.6, f'Only {close:.0%} near centers'

    def test_generate_mocks(self):
        rng = np.random.default_rng(42)
        positions = rng.uniform(0, 1000, size=(500, 2))
        null = DensityMatchedNull()
        null.fit(positions)
        mocks = null.generate_mocks(200, 3, rng)
        assert len(mocks) == 3

    def test_large_dataset_subsampling(self):
        """Fitting on >50k points should subsample internally."""
        rng = np.random.default_rng(42)
        positions = rng.uniform(0, 1000, size=(60000, 2))
        null = DensityMatchedNull()
        null.fit(positions)
        samples = null.sample(100, rng)
        assert samples.shape == (100, 2)


class TestLabelShuffleNull:
    def test_fit_and_shuffle(self):
        rng = np.random.default_rng(42)
        positions = rng.uniform(0, 1000, size=(100, 2))
        labels = np.array(['A'] * 50 + ['B'] * 50)
        null = LabelShuffleNull()
        null.fit(positions, labels)
        shuffled = null.shuffle(rng)
        assert len(shuffled) == 100
        # Same composition
        assert (shuffled == 'A').sum() == 50
        assert (shuffled == 'B').sum() == 50

    def test_shuffle_preserves_composition(self):
        rng = np.random.default_rng(42)
        labels = np.array(['X'] * 30 + ['Y'] * 20 + ['Z'] * 10)
        positions = rng.uniform(0, 100, size=(60, 2))
        null = LabelShuffleNull()
        null.fit(positions, labels)

        for _ in range(10):
            shuffled = null.shuffle(rng)
            assert (shuffled == 'X').sum() == 30
            assert (shuffled == 'Y').sum() == 20
            assert (shuffled == 'Z').sum() == 10

    def test_shuffle_changes_order(self):
        """Shuffling should (usually) change label assignments."""
        rng = np.random.default_rng(42)
        labels = np.array(['A', 'A', 'A', 'B', 'B', 'B'] * 20)
        positions = rng.uniform(0, 100, size=(len(labels), 2))
        null = LabelShuffleNull()
        null.fit(positions, labels)
        shuffled = null.shuffle(rng)
        assert not np.array_equal(shuffled, labels)

    def test_generate_shuffles(self):
        rng = np.random.default_rng(42)
        labels = np.array(['A'] * 50 + ['B'] * 50)
        positions = rng.uniform(0, 100, size=(100, 2))
        null = LabelShuffleNull()
        null.fit(positions, labels)
        shuffles = null.generate_shuffles(10, rng)
        assert len(shuffles) == 10

    def test_mock_positions_for_type(self):
        rng = np.random.default_rng(42)
        positions = rng.uniform(0, 100, size=(100, 2))
        labels = np.array(['A'] * 60 + ['B'] * 40)
        null = LabelShuffleNull()
        null.fit(positions, labels)
        shuffled = null.shuffle(rng)
        mock_a = null.mock_positions_for_type('A', shuffled)
        mock_b = null.mock_positions_for_type('B', shuffled)
        assert len(mock_a) == 60
        assert len(mock_b) == 40
        # Positions should come from the original positions array
        assert mock_a.shape[1] == 2
