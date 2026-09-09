import numpy as np
import pandas as pd

def extract_features(df: pd.DataFrame):
    """Accepts a raw DataFrame and returns a transformed DataFrame
    containing only input features and target variables (used for training)

    Works on both train.csv and test.csv
    """
    features = pd.DataFrame(index=df.index)

    # Launch kinematics and angles
    features["launch_vx"] = df["launch_vx"]
    features["launch_vy"] = df["launch_vy"]
    features["launch_vz"] = df["launch_vz"]
    features["launch_v_total"] = np.sqrt(df["launch_vx"] ** 2 + df["launch_vy"] ** 2 + df["launch_vz"] ** 2)

    # Launch elevation and azimuth direction
    features["launch_elevation_rad"] = np.arctan2(df["launch_vz"], np.sqrt(df["launch_vx"] ** 2 + df["launch_vy"] ** 2))

    features["launch_azimuth_rad"] = np.arctan2(df["launch_vy"], df["launch_vz"])

    # Segment by segment interval velocities and time deltas
    cps = [("launch", 0), ("cp1", 1), ("cp2", 2), ("cp3", 3), ("cp4", 4)]

    for i in range(len(cps) - 1):
        p1, p2 = cps[i][0], cps[i+1][0]

        t1 = 0.0 if p1 == "launch" else df[f"{p1}_t"]
        t2 = df[f"{p2}_t"]
        dt = t2 - t1

        dx = df[f"{p2}_x"] - df[f"{p1}_x"]
        dy = df[f"{p2}_y"] - df[f"{p1}_y"]
        dz = df[f"{p2}_z"] - df[f"{p1}_z"]

        features[f"dt_{p1}_to_{p2}"] = dt
        features[f"vx_{p1}_to_{p2}"] = dx / dt
        features[f"vy_{p1}_to_{p2}"] = dy / dt
        features[f"vz_{p1}_to_{p2}"] = dz / dt
        features[f"v_magnitude_{p1}_to_{p2}"] = np.sqrt(dx**2 + dy**2 + dz**2) / dt

    # Overall features ( Launch -> cp4 / 60m)
    net_dt = df["cp4_t"]
    net_dx = df["cp4_x"] - df["launch_x"]
    net_dy = df["cp4_y"] - df["launch_y"]
    net_dz = df["cp4_z"] - df["launch_z"]

    features["net_dt_total"] = net_dt
    features["net_vx_avg"] = net_dx / net_dt
    features["net_vy_avg"] = net_dy / net_dt
    features["net_vz_avg"] = net_dz / net_dt
    features["net_v_magnitude_avg"] = np.sqrt(net_dx**2 + net_dy**2 + net_dz**2) / net_dt

    # Deceleration and vertical/horizontal trajectory ratios
    features["speed_loss"] = features["v_mag_launch_to_cp1"] - features["v_mag_cp3_to_cp4"]
    features["decel_rate"] = features["speed_loss"] / net_dt

    features["net_climb_ratio"] = net_dz / net_dx
    features["net_drift_ratio"] = net_dy / net_dx

    return features

def get_target_columns():
    """Returns the list of ground truth target variables present in train.csv."""
    return [
        "launch_spin_rate",
        "apex_t",
        "apex_x",
        "apex_y",
        "apex_z",
        "landing_t",
        "landing_x",
        "landing_y",
        "landing_z",
    ]
