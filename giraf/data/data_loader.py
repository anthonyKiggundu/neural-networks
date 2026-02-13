"""
Data loading and preprocessing utilities for GIRAF.
"""

import time
import pandas as pd
from sklearn.model_selection import train_test_split


def stream_kpis_from_dataset(df, feed_interval=1):
    """
    Stream KPI data dynamically row by row from a dataframe.
    
    Args:
        df: Pandas DataFrame containing the KPI data
        feed_interval: Time interval (in seconds) between each step
        
    Yields:
        dict: Row data as dictionary
    """
    for _, row in df.iterrows():
        yield row.to_dict()
        time.sleep(feed_interval)


def prepare_dataset(data, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, random_state=42):
    """
    Split dataset into train, validation, and test sets.
    
    Args:
        data: Pandas DataFrame
        train_ratio: Proportion for training
        val_ratio: Proportion for validation
        test_ratio: Proportion for testing
        random_state: Random seed
        
    Returns:
        tuple: (train_data, val_data, test_data)
    """
    # Ensure critical columns exist
    if "bt_true" not in data.columns:
        data["bt_true"] = 0.95
    if "constraint_coverage" not in data.columns:
        data["constraint_coverage"] = 0.8
    
    # Add KPI input column
    data['kpi_input'] = _create_kpi_input(data)
    data['kpi_description'] = "This should describe the response for the given KPIs."
    
    # Split data
    train_data, temp_data = train_test_split(data, test_size=(val_ratio + test_ratio), random_state=random_state)
    val_data, test_data = train_test_split(temp_data, test_size=test_ratio/(val_ratio + test_ratio), random_state=random_state)
    
    print(f"Data split into training ({len(train_data)} rows), validation ({len(val_data)} rows), "
          f"and testing ({len(test_data)} rows).")
    
    return train_data, val_data, test_data


def _create_kpi_input(data):
    """Create formatted KPI input strings from dataframe."""
    return (
        "Device: " + data['device'].astype(str) +
        "\nTimestamp: " + data.index.astype(str) +
        "\nLocation: (Latitude: " + data['Latitude'].astype(str) +
        ", Longitude: " + data['Longitude'].astype(str) +
        ", Altitude: " + data['Altitude'].astype(str) + ")" +
        "\nMobility:\n  - Speed: " + data['speed_kmh'].astype(str) + " km/h" +
        "\n  - Traffic Jam Factor: " + data['Traffic Jam Factor'].astype(str) +
        "\nNetwork KPIs:\n  - Latency (ping_ms): " + data['ping_ms'].astype(str) +
        "\n  - Jitter: " + data['jitter'].astype(str) +
        "\n  - Datarate: " + data['datarate'].astype(str) +
        "\n  - Target Datarate: " + data['target_datarate'].astype(str) +
        "\nSignal Quality (PCell):\n  - RSRP: " + data['PCell_RSRP_1'].astype(str) + " dBm" +
        "\n  - RSRQ: " + data['PCell_RSRQ_1'].astype(str) + " dB" +
        "\n  - SNR: " + data['PCell_SNR_1'].astype(str) + " dB" +
        "\nResource Utilization:\n  - Downlink Resource Blocks: " + data['PCell_Downlink_Num_RBs'].astype(str) +
        "\n  - Uplink Resource Blocks: " + data['PCell_Uplink_Num_RBs'].astype(str)
    )
