"""Allen Brain Cell Atlas MERFISH data download and preprocessing.

Downloads cell metadata (coordinates + type annotations) from the Allen
Brain Cell Atlas via abc_atlas_access. Extracts cells for specific brain
regions and prepares them for multiscale spatial analysis.

Dataset: Zhang et al. (2023), Nature — ~4M cells, 500-gene MERFISH panel,
whole adult mouse brain registered to Allen CCF v3.

Data structure (actual API):
- MERFISH-C57BL6J-638850 / cell_metadata_with_cluster_annotation
  → cell_label, brain_section_label, cluster_alias, x, y, z,
    neurotransmitter, class, subclass, supertype, cluster
- MERFISH-C57BL6J-638850-CCF / ccf_coordinates
  → cell_label, x, y, z, parcellation_index
- Allen-CCF-2020 / parcellation_to_parcellation_term_membership_acronym
  → parcellation_index, organ, category, division, structure, substructure
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Allen CCF division names for our three target regions
REGION_DIVISIONS = {
    'isocortex': 'Isocortex',
    'hippocampus': 'HPF',       # Hippocampal Formation
    'thalamus': 'TH',
}

# Type hierarchy levels available in the atlas
TYPE_LEVELS = ('class', 'subclass', 'supertype', 'cluster')


def download_merfish_metadata(cache_dir='data/abc_atlas'):
    """Download cell metadata with type annotations and region assignments.

    Downloads three tables and merges them:
    1. Cell metadata with cluster annotations (x, y, z, class, subclass, ...)
    2. CCF coordinates with parcellation_index
    3. Parcellation index → division/structure mapping

    Parameters
    ----------
    cache_dir : str or Path
        Directory for cached downloads.

    Returns
    -------
    cell_df : pd.DataFrame
        All cells with columns: x, y, z, class, subclass, supertype,
        division, structure, ...
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    from abc_atlas_access.abc_atlas_cache.abc_project_cache import AbcProjectCache
    abc = AbcProjectCache.from_cache_dir(cache_dir)

    # 1. Cell metadata with cluster annotations (has type info + coordinates)
    logger.info('Loading cell metadata with cluster annotations...')
    cell_df = abc.get_metadata_dataframe(
        directory='MERFISH-C57BL6J-638850',
        file_name='cell_metadata_with_cluster_annotation',
        dtype={'cell_label': str},
    )
    logger.info(f'Loaded {len(cell_df):,} cells with type annotations')

    # 2. CCF coordinates with parcellation_index
    logger.info('Loading CCF coordinates...')
    ccf_df = abc.get_metadata_dataframe(
        directory='MERFISH-C57BL6J-638850-CCF',
        file_name='ccf_coordinates',
        dtype={'cell_label': str},
    )
    logger.info(f'Loaded {len(ccf_df):,} CCF coordinates')

    # 3. Parcellation index → region mapping
    logger.info('Loading parcellation structure tree...')
    p2t = abc.get_metadata_dataframe(
        directory='Allen-CCF-2020',
        file_name='parcellation_to_parcellation_term_membership_acronym',
    )
    logger.info(f'Loaded {len(p2t)} parcellation entries')

    # Merge parcellation info onto CCF coordinates
    ccf_with_region = ccf_df.merge(
        p2t[['parcellation_index', 'division', 'structure', 'substructure']],
        on='parcellation_index',
        how='left',
    )

    # Merge region info onto cell metadata via cell_label
    cell_df = cell_df.merge(
        ccf_with_region[['cell_label', 'division', 'structure', 'substructure']],
        on='cell_label',
        how='left',
    )
    cell_df.set_index('cell_label', inplace=True)

    logger.info(f'Final cell table: {len(cell_df):,} cells, '
                f'{len(cell_df.columns)} columns')
    logger.info(f'Columns: {list(cell_df.columns)}')

    # Print division summary
    if 'division' in cell_df.columns:
        div_counts = cell_df['division'].value_counts().head(10)
        logger.info(f'Top divisions:\n{div_counts}')

    return cell_df


def extract_region(cell_df, region_name):
    """Filter cells belonging to a specific brain region.

    Parameters
    ----------
    cell_df : pd.DataFrame
        Full cell metadata from download_merfish_metadata().
    region_name : str
        Region name key: 'isocortex', 'hippocampus', or 'thalamus'.

    Returns
    -------
    region_df : pd.DataFrame
        Subset of cells in the requested region.
    """
    division = REGION_DIVISIONS.get(region_name.lower(), region_name)

    if 'division' not in cell_df.columns:
        raise ValueError(
            'No "division" column found. Run download_merfish_metadata() '
            'to get region annotations.'
        )

    mask = cell_df['division'] == division
    region_df = cell_df[mask].copy()

    logger.info(f'Region "{region_name}" (division={division}): '
                f'{len(region_df):,} cells')

    if len(region_df) == 0:
        unique_divs = cell_df['division'].dropna().unique()[:20]
        logger.warning(f'No cells found. Available divisions: {list(unique_divs)}')

    return region_df


def prepare_analysis_data(cells_df, type_level='class'):
    """Extract positions and type labels for spatial analysis.

    Parameters
    ----------
    cells_df : pd.DataFrame
        Cell data with x, y columns and type annotation columns.
    type_level : str
        One of 'class', 'subclass', 'supertype'. Which type hierarchy
        level to use for labels.

    Returns
    -------
    positions : np.ndarray, shape (N, 2)
        Cell (x, y) coordinates in micrometers.
    labels : np.ndarray, shape (N,)
        Cell type labels as strings.
    """
    # The atlas uses bare column names: 'class', 'subclass', 'supertype'
    type_col = type_level
    if type_col not in cells_df.columns:
        # Try with prefix
        alt = f'cell_type_{type_level}'
        if alt in cells_df.columns:
            type_col = alt
        else:
            raise ValueError(
                f'Type column "{type_col}" not found. '
                f'Available: {[c for c in cells_df.columns if "class" in c.lower() or "type" in c.lower() or "super" in c.lower()]}'
            )

    # Drop cells with missing coordinates or type
    valid = cells_df.dropna(subset=['x', 'y', type_col])
    positions = valid[['x', 'y']].values.astype(np.float64)

    # Allen MERFISH coordinates are in mm — convert to μm
    if positions[:, 0].max() < 20:  # heuristic: mm range
        positions = positions * 1000.0

    labels = valid[type_col].values.astype(str)

    logger.info(f'Prepared {len(positions):,} cells, '
                f'{len(np.unique(labels))} unique types at level "{type_level}"')

    return positions, labels


def save_region_data(region_df, output_path):
    """Save preprocessed region data to parquet."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    keep_cols = ['x', 'y', 'z']
    for level in TYPE_LEVELS:
        if level in region_df.columns:
            keep_cols.append(level)

    available = [c for c in keep_cols if c in region_df.columns]
    region_df[available].to_parquet(output_path, index=True)
    logger.info(f'Saved {len(region_df):,} cells to {output_path}')


def load_region_data(path):
    """Load preprocessed region data from parquet."""
    return pd.read_parquet(path)
