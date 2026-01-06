"""
GeoNet New Zealand Earthquake Catalog Downloader

Downloads earthquake data from GeoNet's FDSN web service.
"""

import requests
import pandas as pd
import numpy as np
import os
from io import StringIO
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# GeoNet FDSN Event Service URL
GEONET_URL = "https://service.geonet.org.nz/fdsnws/event/1/query"

def download_geonet_catalog(
    starttime: str = "1950-01-01",
    endtime: str = "2020-01-01",
    minmagnitude: float = 4.0,
    minlatitude: float = -48.0,
    maxlatitude: float = -34.0,
    minlongitude: float = 165.0,
    maxlongitude: float = 180.0,
    output_file: str = None
) -> pd.DataFrame:
    """
    Download earthquake catalog from GeoNet.
    
    Args:
        starttime: Start date (YYYY-MM-DD)
        endtime: End date (YYYY-MM-DD)
        minmagnitude: Minimum magnitude
        minlatitude, maxlatitude: Latitude bounds
        minlongitude, maxlongitude: Longitude bounds
        output_file: Optional path to save CSV
        
    Returns:
        DataFrame with earthquake catalog
    """
    
    params = {
        "format": "text",
        "starttime": starttime,
        "endtime": endtime,
        "minmagnitude": minmagnitude,
        "minlatitude": minlatitude,
        "maxlatitude": maxlatitude,
        "minlongitude": minlongitude,
        "maxlongitude": maxlongitude,
        "eventtype": "earthquake"
    }
    
    logger.info(f"Downloading GeoNet catalog: {starttime} to {endtime}, M >= {minmagnitude}")
    logger.info(f"Region: lat [{minlatitude}, {maxlatitude}], lon [{minlongitude}, {maxlongitude}]")
    
    response = requests.get(GEONET_URL, params=params, timeout=120)
    response.raise_for_status()
    
    # Parse the text response (pipe-separated)
    # Header line starts with #, e.g.: #EventID | Time | Latitude | ...
    data = StringIO(response.text)
    
    # Read without comment filter since header has #
    df = pd.read_csv(data, sep="|", skipinitialspace=True)
    
    # Strip whitespace from column names and remove # from first column
    df.columns = df.columns.str.strip()
    df.columns = [col.lstrip('#') for col in df.columns]
    
    # Rename columns to match expected format
    df = df.rename(columns={
        "EventID": "id",
        "Time": "time",
        "Latitude": "latitude",
        "Longitude": "longitude",
        "Depth/km": "depth",
        "Magnitude": "magnitude",
        "MagType": "mag_type"
    })
    
    # Parse time
    df["time"] = pd.to_datetime(df["time"])
    
    # Select relevant columns
    df = df[["id", "time", "latitude", "longitude", "magnitude"]].copy()
    df = df.set_index("id")
    df = df.sort_values("time")
    
    logger.info(f"Downloaded {len(df)} events")
    
    if output_file:
        os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else ".", exist_ok=True)
        df.to_csv(output_file)
        logger.info(f"Saved catalog to {output_file}")
    
    return df


def create_nz_polygon(output_file: str = None) -> np.ndarray:
    """
    Create a simple polygon covering New Zealand's main earthquake region.
    
    Returns:
        numpy array of [lat, lon] coordinates
    """
    # Simple rectangular polygon covering NZ
    # Could be made more complex to follow coastline if needed
    polygon = np.array([
        [-34.0, 165.0],
        [-34.0, 180.0],
        [-48.0, 180.0],
        [-48.0, 165.0],
        [-34.0, 165.0],  # Close the polygon
    ])
    
    if output_file:
        os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else ".", exist_ok=True)
        np.save(output_file, polygon)
        logger.info(f"Saved NZ polygon to {output_file}")
    
    return polygon


def download_geonet_catalog_chunked(
    starttime: str = "1950-01-01",
    endtime: str = "2020-01-01",
    minmagnitude: float = 4.0,
    minlatitude: float = -48.0,
    maxlatitude: float = -34.0,
    minlongitude: float = 165.0,
    maxlongitude: float = 180.0,
    output_file: str = None,
    chunk_years: int = 5
) -> pd.DataFrame:
    """
    Download earthquake catalog from GeoNet in chunks to avoid 413 errors.
    """
    from datetime import datetime
    
    start = datetime.strptime(starttime, "%Y-%m-%d")
    end = datetime.strptime(endtime, "%Y-%m-%d")
    
    all_dfs = []
    current = start
    
    while current < end:
        chunk_end = datetime(current.year + chunk_years, 1, 1)
        if chunk_end > end:
            chunk_end = end
            
        try:
            df = download_geonet_catalog(
                starttime=current.strftime("%Y-%m-%d"),
                endtime=chunk_end.strftime("%Y-%m-%d"),
                minmagnitude=minmagnitude,
                minlatitude=minlatitude,
                maxlatitude=maxlatitude,
                minlongitude=minlongitude,
                maxlongitude=maxlongitude,
                output_file=None  # Don't save individual chunks
            )
            all_dfs.append(df)
        except Exception as e:
            logger.warning(f"Failed to download {current} to {chunk_end}: {e}")
        
        current = chunk_end
    
    if not all_dfs:
        raise ValueError("No data downloaded")
    
    catalog = pd.concat(all_dfs)
    catalog = catalog[~catalog.index.duplicated(keep='first')]  # Remove duplicates
    catalog = catalog.sort_values("time")
    
    logger.info(f"Total events downloaded: {len(catalog)}")
    
    if output_file:
        os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else ".", exist_ok=True)
        catalog.to_csv(output_file)
        logger.info(f"Saved catalog to {output_file}")
    
    return catalog


if __name__ == "__main__":
    # Download catalog covering the period needed for Kaikoura (2016) and Canterbury (2010) sequences
    # With auxiliary period starting from 1950
    
    catalog = download_geonet_catalog_chunked(
        starttime="1950-01-01",
        endtime="2020-01-01",
        minmagnitude=4.0,  # Using M4.0+ as in user's config (mc=4.1)
        output_file="../input_data/nzcat.csv",
        chunk_years=10  # Download in 10-year chunks
    )
    
    # Create NZ polygon
    create_nz_polygon(output_file="../input_data/nz_polygon.npy")
    
    print("\n--- Catalog Summary ---")
    print(f"Events: {len(catalog)}")
    print(f"Time range: {catalog['time'].min()} to {catalog['time'].max()}")
    print(f"Magnitude range: {catalog['magnitude'].min():.1f} to {catalog['magnitude'].max():.1f}")
    print(catalog.head())
