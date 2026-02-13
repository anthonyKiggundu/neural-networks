"""
Unit tests for data loading.
"""

import pytest
import pandas as pd
from giraf.data import prepare_dataset, stream_kpis_from_dataset


def test_prepare_dataset():
    """Test dataset preparation."""
    # Create mock data
    data = pd.DataFrame({
        'device': ['dev1'] * 100,
        'Latitude': [0.0] * 100,
        'Longitude': [0.0] * 100,
        'Altitude': [0.0] * 100,
        'speed_kmh': [50] * 100,
        'Traffic Jam Factor': [5] * 100,
        'ping_ms': [20] * 100,
        'jitter': [5] * 100,
        'datarate': [100] * 100,
        'target_datarate': [100] * 100,
        'PCell_RSRP_1': [-80] * 100,
        'PCell_RSRQ_1': [-10] * 100,
        'PCell_SNR_1': [20] * 100,
        'PCell_Downlink_Num_RBs': [50] * 100,
        'PCell_Uplink_Num_RBs': [25] * 100,
        'measured_qos': [0.9] * 100,
        'operator': ['op1'] * 100
    })
    
    train, val, test = prepare_dataset(data, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2)
    
    assert len(train) == 60
    assert len(val) == 20
    assert len(test) == 20
    assert 'kpi_input' in train.columns
    assert 'kpi_description' in train.columns


def test_stream_kpis():
    """Test KPI streaming."""
    data = pd.DataFrame({
        'value': [1, 2, 3]
    })
    
    stream = stream_kpis_from_dataset(data, feed_interval=0.01)
    
    results = list(stream)
    assert len(results) == 3
    assert results[0]['value'] == 1
