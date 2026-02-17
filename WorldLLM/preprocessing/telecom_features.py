# src/preprocessing/telecom_features.py
import numpy as np
import pandas as pd

# Final world-state channel order
CHANNELS = [
    "rsrp", "rsrq", "snr",
    "dl_load", "ul_load",
    "spectral_efficiency",
    "throughput", "latency", "jitter",
    "user_density", "avg_speed", "traffic_jam"
]

def compute_spectral_efficiency(df):
    """Approx spectral efficiency from MCS."""
    if "PCell_Downlink_Average_MCS" not in df:
        return np.zeros(len(df))
    return df["PCell_Downlink_Average_MCS"].fillna(0) / 31.0

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Reduce huge telecom dataframe to physics features."""
    
    df = df.copy()

    df["rsrp"] = df["PCell_RSRP_max"].fillna(-120)
    df["rsrq"] = df["PCell_RSRQ_max"].fillna(-20)
    df["snr"]  = df["PCell_SNR_1"].fillna(0)

    df["dl_load"] = df["PCell_Downlink_Num_RBs"].fillna(0)
    df["ul_load"] = df["PCell_Uplink_Num_RBs"].fillna(0)

    df["spectral_efficiency"] = compute_spectral_efficiency(df)

    df["throughput"] = df["datarate"].fillna(0)
    df["latency"] = df["ping_ms"].fillna(0)
    df["jitter"] = df["jitter"].fillna(0)

    df["user_density"] = 1.0
    df["avg_speed"] = df["speed_kmh"].fillna(0)
    df["traffic_jam"] = df["Traffic Jam Factor"].fillna(0)

    return df[["timestamp","Latitude","Longitude"] + CHANNELS]

