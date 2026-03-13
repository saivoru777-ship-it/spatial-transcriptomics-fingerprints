"""Null models for spatial transcriptomics analysis.

Three null models of increasing stringency:
1. CSR — Complete Spatial Randomness within the tissue boundary
2. DensityMatched — Preserves overall cell density gradients
3. LabelShuffle — Preserves ALL spatial structure, only permutes type labels

The label-shuffle null is the primary control: any significant deviation means
cell type positions are non-random given the overall spatial arrangement.
"""

import numpy as np
from scipy.spatial import ConvexHull, Delaunay
from sklearn.neighbors import KernelDensity


class TissueCSRNull:
    """Complete Spatial Randomness within the convex hull of cell positions.

    Generates mock datasets by uniformly sampling points inside the convex
    hull of observed positions. Analog of BiologicalNullModel from SMLM.

    Any cell type with VMR above CSR is spatially clustered beyond what
    uniform random placement would produce.
    """

    def __init__(self):
        self.hull = None
        self.delaunay = None
        self.bbox_min = None
        self.bbox_max = None
        self._acceptance_rate = None

    def fit(self, positions):
        """Fit the null model to observed positions.

        Parameters
        ----------
        positions : array-like, shape (N, 2)
            Cell (x, y) coordinates.

        Returns
        -------
        self
        """
        positions = np.asarray(positions, dtype=float)
        self.hull = ConvexHull(positions)
        self.delaunay = Delaunay(positions[self.hull.vertices])
        self.bbox_min = positions.min(axis=0)
        self.bbox_max = positions.max(axis=0)
        return self

    def _in_hull(self, points):
        """Test whether points lie inside the convex hull."""
        return self.delaunay.find_simplex(points) >= 0

    def sample(self, n, rng=None):
        """Sample n points uniformly within the convex hull.

        Uses rejection sampling: draw from bounding box, keep if inside hull.

        Parameters
        ----------
        n : int
            Number of points to sample.
        rng : np.random.Generator, optional
            Random number generator.

        Returns
        -------
        points : np.ndarray, shape (n, 2)
        """
        if rng is None:
            rng = np.random.default_rng()

        points = np.empty((0, 2))
        total_tried = 0

        while len(points) < n:
            batch_size = max(n * 3, 10000)
            candidates = rng.uniform(
                self.bbox_min, self.bbox_max, size=(batch_size, 2)
            )
            inside = candidates[self._in_hull(candidates)]
            points = np.vstack([points, inside]) if len(points) > 0 else inside
            total_tried += batch_size

        self._acceptance_rate = len(points) / total_tried
        return points[:n]

    def generate_mocks(self, n, n_mocks, rng=None):
        """Generate multiple mock datasets.

        Parameters
        ----------
        n : int
            Number of points per mock.
        n_mocks : int
            Number of mock datasets.
        rng : np.random.Generator, optional

        Returns
        -------
        list of np.ndarray, each shape (n, 2)
        """
        if rng is None:
            rng = np.random.default_rng()
        return [self.sample(n, rng) for _ in range(n_mocks)]

    @property
    def acceptance_rate(self):
        return self._acceptance_rate

    @property
    def hull_area(self):
        if self.hull is None:
            return 0.0
        return float(self.hull.volume)  # In 2D, ConvexHull.volume = area


class DensityMatchedNull:
    """Density-matched null model using 2D KDE.

    Samples mock positions proportional to the overall cell density field,
    accounting for density gradients across the tissue. Tests whether
    cell type clustering exceeds what density gradients alone explain.

    Analog of the soma-distance null from dendritic analysis.
    """

    def __init__(self):
        self.kde = None
        self.positions = None
        self.bandwidth = None

    def fit(self, all_positions, bandwidth=None):
        """Fit a 2D KDE to the overall cell density.

        Parameters
        ----------
        all_positions : array-like, shape (N, 2)
            All cell positions (not just one type).
        bandwidth : float, optional
            KDE bandwidth in data units. If None, estimated automatically.

        Returns
        -------
        self
        """
        self.positions = np.asarray(all_positions, dtype=float)

        if bandwidth is None:
            # Silverman's rule for 2D
            n = len(self.positions)
            d = 2
            std = self.positions.std(axis=0).mean()
            bandwidth = std * (n * (d + 2) / 4) ** (-1 / (d + 4))

        self.bandwidth = bandwidth
        # Subsample for KDE fitting if dataset is very large
        if len(self.positions) > 50000:
            rng = np.random.default_rng(42)
            idx = rng.choice(len(self.positions), 50000, replace=False)
            fit_data = self.positions[idx]
        else:
            fit_data = self.positions

        self.kde = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
        self.kde.fit(fit_data)
        return self

    def sample(self, n, rng=None):
        """Sample n points from the fitted density.

        Parameters
        ----------
        n : int
            Number of points.
        rng : np.random.Generator, optional

        Returns
        -------
        points : np.ndarray, shape (n, 2)
        """
        if rng is None:
            rng = np.random.default_rng()

        seed = rng.integers(0, 2**31)
        points = self.kde.sample(n, random_state=seed)
        return points

    def generate_mocks(self, n, n_mocks, rng=None):
        """Generate multiple density-matched mock datasets.

        Parameters
        ----------
        n : int
            Points per mock.
        n_mocks : int
            Number of mocks.
        rng : np.random.Generator, optional

        Returns
        -------
        list of np.ndarray, each shape (n, 2)
        """
        if rng is None:
            rng = np.random.default_rng()
        return [self.sample(n, rng) for _ in range(n_mocks)]


class LabelShuffleNull:
    """Label-shuffle null model — the strongest control.

    Preserves ALL spatial structure of the cell population. Only permutes
    the type label assignments. Any significant deviation from this null
    means the mapping of types to spatial positions is non-random.

    This is the direct analog of the label-shuffle from neurostat
    dendritic analysis (the "killer control").
    """

    def __init__(self):
        self.positions = None
        self.labels = None

    def fit(self, positions, labels):
        """Store positions and labels.

        Parameters
        ----------
        positions : array-like, shape (N, 2)
            Cell coordinates (preserved unchanged in shuffles).
        labels : array-like, shape (N,)
            Cell type labels to shuffle.

        Returns
        -------
        self
        """
        self.positions = np.asarray(positions, dtype=float)
        self.labels = np.asarray(labels)
        return self

    def shuffle(self, rng=None):
        """Return shuffled labels (positions unchanged).

        Parameters
        ----------
        rng : np.random.Generator, optional

        Returns
        -------
        shuffled_labels : np.ndarray, shape (N,)
        """
        if rng is None:
            rng = np.random.default_rng()
        shuffled = self.labels.copy()
        rng.shuffle(shuffled)
        return shuffled

    def generate_shuffles(self, n_shuffles, rng=None):
        """Generate multiple shuffled label assignments.

        Parameters
        ----------
        n_shuffles : int
        rng : np.random.Generator, optional

        Returns
        -------
        list of np.ndarray — each is a permuted label array
        """
        if rng is None:
            rng = np.random.default_rng()
        return [self.shuffle(rng) for _ in range(n_shuffles)]

    def mock_positions_for_type(self, target_type, shuffled_labels):
        """Extract positions of a given type under shuffled labels.

        Parameters
        ----------
        target_type : str
            The cell type to extract.
        shuffled_labels : np.ndarray
            A permuted label array from shuffle().

        Returns
        -------
        np.ndarray, shape (n_type, 2)
            Positions of cells assigned target_type in the shuffled labels.
        """
        mask = shuffled_labels == target_type
        return self.positions[mask]
