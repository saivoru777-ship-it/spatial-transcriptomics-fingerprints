#!/usr/bin/env python3
"""Download Allen Brain Cell Atlas MERFISH data and extract 3 brain regions.

Usage:
    python3 scripts/01_download_and_preprocess.py [--cache-dir data/abc_atlas]

Downloads ~4M cell metadata (coordinates + type annotations) via
abc_atlas_access. Extracts isocortex, hippocampus, and thalamus.
Saves preprocessed parquet files (~5-10 min first run, cached after).
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loading import (
    download_merfish_metadata,
    extract_region,
    prepare_analysis_data,
    save_region_data,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(name)s %(levelname)s: %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)

REGIONS = ['isocortex', 'hippocampus', 'thalamus']


def main():
    parser = argparse.ArgumentParser(
        description='Download and preprocess Allen MERFISH data'
    )
    parser.add_argument('--cache-dir', default='data/abc_atlas',
                        help='Directory for cached ABC atlas downloads')
    parser.add_argument('--output-dir', default='data',
                        help='Directory for output parquet files')
    parser.add_argument('--regions', nargs='+', default=REGIONS,
                        help='Brain regions to extract')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Download full cell metadata
    logger.info('=' * 60)
    logger.info('Step 1: Downloading cell metadata')
    logger.info('=' * 60)
    cell_df = download_merfish_metadata(args.cache_dir)
    logger.info(f'Total cells: {len(cell_df):,}')

    # Print type hierarchy summary
    for level in ['class', 'subclass', 'supertype']:
        if level in cell_df.columns:
            n_unique = cell_df[level].nunique()
            logger.info(f'  {level}: {n_unique} unique types')

    # Step 2: Extract each region
    logger.info('=' * 60)
    logger.info('Step 2: Extracting brain regions')
    logger.info('=' * 60)

    for region in args.regions:
        logger.info(f'\n--- {region.upper()} ---')
        region_df = extract_region(cell_df, region)

        if len(region_df) == 0:
            logger.warning(f'No cells found for {region}, skipping')
            continue

        # Print region summary
        positions, labels = prepare_analysis_data(region_df, type_level='class')
        logger.info(f'  Cells with valid coords+type: {len(positions):,}')
        logger.info(f'  Spatial extent: '
                    f'x=[{positions[:, 0].min():.1f}, {positions[:, 0].max():.1f}], '
                    f'y=[{positions[:, 1].min():.1f}, {positions[:, 1].max():.1f}]')

        # Print type breakdown
        unique, counts = np.unique(labels, return_counts=True)
        order = np.argsort(-counts)
        logger.info(f'  Types ({len(unique)}):')
        for i in order[:10]:
            logger.info(f'    {unique[i]}: {counts[i]:,} cells')
        if len(unique) > 10:
            logger.info(f'    ... and {len(unique) - 10} more')

        # Save
        output_path = output_dir / f'{region}_cells.parquet'
        save_region_data(region_df, output_path)

    logger.info('=' * 60)
    logger.info('Preprocessing complete')
    logger.info('=' * 60)


if __name__ == '__main__':
    main()
