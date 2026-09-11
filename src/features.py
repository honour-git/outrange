import numpy as np
import pandas as pd
from src.physics import project_trajectory, estimate_cd_cl

def extract_features(df: pd.DataFrame):
    """Accepts a raw DataFrame and returns a transformed DataFrame
    containing only input features and target variables (used for training)

    Works on both train.csv and test.csv
    """
    features = pd.DataFrame(index=df.index)

    # Launch kinematics
    features["launch_vx"] = df["launch_vx"]
    features["launch_vy"] = df["launch_vy"]
    features["launch_vz"] = df["launch_vz"]
    features["launch_v_total"] = np.sqrt(df["launch_vx"] ** 2 + df["launch_vy"] ** 2 + df["launch_vz"] ** 2)

    # Launch elevation and azimuth direction
    features["launch_elevation_rad"] = np.arctan2(df["launch_vz"], np.sqrt(df["launch_vx"] ** 2 + df["launch_vy"] ** 2))

    features["launch_azimuth_rad"] = np.arctan2(df["launch_vy"], df["launch_vx"])

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


    # Checkpoint-by-checkpoint acceleration derivatives (Curvature change)
    features["ax_cp1_cp2"] = (
        features["vx_cp1_to_cp2"] - features["vx_launch_to_cp1"]
    ) / features["dt_cp1_to_cp2"]
    features["az_cp1_cp2"] = (
        features["vz_cp1_to_cp2"] - features["vz_launch_to_cp1"]
    ) / features["dt_cp1_to_cp2"]

    features["ax_cp3_cp4"] = (
        features["vx_cp3_to_cp4"] - features["vx_cp2_to_cp3"]
    ) / features["dt_cp3_to_cp4"]
    features["az_cp3_cp4"] = (
        features["vz_cp3_to_cp4"] - features["vz_cp2_to_cp3"]
    ) / features["dt_cp3_to_cp4"]

    # Vertical lift decay rate (Spin decay proxy)
    features["lift_decay_rate"] = features["az_cp3_cp4"] - features["az_cp1_cp2"]


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
    features["speed_loss"] = features["v_magnitude_launch_to_cp1"] - features["v_magnitude_cp3_to_cp4"]
    features["decel_rate"] = features["speed_loss"] / net_dt

    features["net_climb_ratio"] = net_dz / net_dx
    features["net_drift_ratio"] = net_dy / net_dx


    # ODE trajectory projections and aerodynamic parameter estimations (per-shot)
    phys_results = []
    fitted_cds = []
    fitted_cls = []

    for idx, row in df.iterrows():
        launch_state = [
            row["launch_x"],
            row["launch_y"],
            row["launch_z"],
            row["launch_vx"],
            row["launch_vy"],
            row["launch_vz"],
        ]

        checkpoints_xyzt = [
            (row["cp1_x"], row["cp1_y"], row["cp1_z"], row["cp1_t"]),
            (row["cp2_x"], row["cp2_y"], row["cp2_z"], row["cp2_t"]),
            (row["cp3_x"], row["cp3_y"], row["cp3_z"], row["cp3_t"]),
            (row["cp4_x"], row["cp4_y"], row["cp4_z"], row["cp4_t"]),
        ]

        # Estimate Cd and Cl from provided checkpoint data
        cd_est, cl_est = estimate_cd_cl(launch_state, checkpoints_xyzt)
        fitted_cds.append(cd_est)
        fitted_cls.append(cl_est)

        # Simulate trajectory using baseline drag and lift coefficients
        sim = project_trajectory(launch_state=launch_state, Cd=cd_est, Cl=cl_est)
        phys_results.append(sim)

    features["est_cd"] = fitted_cds
    features["est_cl"] = fitted_cls

    phys_df = pd.DataFrame(phys_results, index=df.index)

    # Add physics outputs as features for LightGBM
    features["phys_apex_x"] = phys_df["apex_x"]
    features["phys_apex_y"] = phys_df["apex_y"]
    features["phys_apex_z"] = phys_df["apex_z"]
    features["phys_apex_t"] = phys_df["apex_t"]

    features["phys_landing_x"] = phys_df["landing_x"]
    features["phys_landing_y"] = phys_df["landing_y"]
    features["phys_landing_t"] = phys_df["landing_t"]

    # Differences between physical baseline prediction and observed CP4 position
    features["cp4_vs_phys_x_diff"] = df["cp4_x"] - phys_df["apex_x"]
    features["cp4_vs_phys_z_diff"] = df["cp4_z"] - phys_df["apex_z"]


    # Lift/Drag ratio proxy
    features["observed_lift_proxy"] = (df["cp4_z"] - df["launch_z"]) / features["net_dt_total"]
    features["observed_drag_proxy"] = (features["v_magnitude_launch_to_cp1"] - features["v_magnitude_cp3_to_cp4"]) / features["net_dt_total"]
    features["lift_to_drag_ratio"] = features["observed_lift_proxy"] / features["observed_drag_proxy"]


    # Aerodynamic Proxy Features
    net_v = features["net_v_magnitude_avg"]

    # Deviation from parabolic vacuum trajectory
    vacuum_dz = (df["launch_vz"] * df["cp4_t"]) - (0.5 * 9.81 * (df["cp4_t"] ** 2))
    actual_dz = df["cp4_z"] - df["launch_z"]
    features["lift_curvature_delta"] = actual_dz - vacuum_dz

    # Estimated Magnus lift proxy
    features["magnus_cl_proxy"] = np.clip(
        (2 * 0.04593 * features["lift_curvature_delta"])
        / (1.225 * np.pi * (0.021335 ** 2) * (net_v ** 2) * (df["cp4_t"] ** 2)),
        -0.5,
        1.0,
    )

    # Inferred spin rate physical approximation (RPM)
    features["estimated_spin_rpm_proxy"] = np.clip(features["est_cl"] * net_v * 120.0, 500.0, 12000.0)

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
