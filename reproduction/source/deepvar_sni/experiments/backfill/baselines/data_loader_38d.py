"""
Aligned 38-d data loader for multi-factor baseline evaluation.

Produces a canonical (5821, 38) daily-changes array:
  - dims 0:25  = IV surface daily changes (diff of flattened 5x5 grid)
  - dims 25:38 = financial factor returns (log-returns or first-differences)

Date alignment: IV surface (5822 rows) and multi-factor data (5825 rows) share
the same trading calendar but differ by 3 dates (2000-09-15, 2000-09-18, 2023-02-27).
Alignment is by date, not by index.

NaN policy:
  - Factor returns: fill with 0.0 (= no change on missing trading days)
  - Factor levels: forward-fill (back-fill only if first row is NaN)
"""

import numpy as np
import pandas as pd
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data"

# Column names for the 38-d representation
IV_CELL_NAMES = [f"iv_{i}_{j}" for i in range(5) for j in range(5)]


def load_aligned_38d_data(
    history_len=30,
    future_len=30,
    train_end=4040,
    test_start=4540,
):
    """Load and align IV surfaces + 13 financial factors.

    Returns dict with:
        dates: (5822,) datetime64 — aligned trading dates
        surfaces: (5822, 5, 5) float64 — original IV surfaces in [0, 1]
        iv_changes_25: (5821, 25) float64 — daily changes of flattened IV
        factor_returns_13: (5822, 13) float32 — aligned factor returns, NaN→0.0
        factor_levels_13: (5822, 13) float32 — aligned factor levels, forward-filled
        factor_return_columns: list[str] — 13 return column names
        factor_level_columns: list[str] — 13 level column names
        joint_changes_38: (5821, 38) float64 — [iv_changes, factor_returns] concatenated
        train_mean_38: (38,) — z-score mean from training set
        train_std_38: (38,) — z-score std from training set
        train_mean_25: (25,) — IV-only z-score mean
        train_std_25: (25,) — IV-only z-score std
        column_names: list[str] — 38 column names
        alignment_indices: (5822,) int — MF array index for each IV row
    """
    # --- Load raw data ---
    iv_data = np.load(DATA_DIR / "vol_surface_with_ret.npz")
    mf_data = np.load(DATA_DIR / "multi_factor_data.npz", allow_pickle=True)
    iv_parquet = pd.read_parquet(DATA_DIR / "spx_vol_surface_history_full_data_fixed.parquet")

    surfaces = iv_data["surface"]  # (5822, 5, 5)
    mf_dates = pd.to_datetime(mf_data["dates"])
    mf_returns = mf_data["returns"]  # (5825, 13)
    mf_levels = mf_data["levels"]  # (5825, 13)
    return_columns = list(mf_data["return_columns"])
    level_columns = list(mf_data["level_columns"])

    iv_dates = pd.to_datetime(iv_parquet["date"])

    # --- Date-based alignment ---
    mf_date_to_idx = {d: i for i, d in enumerate(mf_dates)}
    alignment_indices = np.array(
        [mf_date_to_idx[d] for d in iv_dates], dtype=np.int64
    )
    assert len(alignment_indices) == len(surfaces), (
        f"Alignment mismatch: {len(alignment_indices)} indices vs {len(surfaces)} surfaces"
    )

    # --- Extract aligned factor data ---
    aligned_returns = mf_returns[alignment_indices].copy()  # (5822, 13)
    aligned_levels = mf_levels[alignment_indices].copy()  # (5822, 13)

    # NaN handling: returns → 0.0, levels → forward-fill then back-fill first row
    nan_mask_ret = np.isnan(aligned_returns)
    aligned_returns[nan_mask_ret] = 0.0

    # Forward-fill levels column by column
    for col in range(aligned_levels.shape[1]):
        series = aligned_levels[:, col]
        nan_mask = np.isnan(series)
        if nan_mask.any():
            # Forward-fill
            last_valid = np.nan
            for i in range(len(series)):
                if nan_mask[i]:
                    if not np.isnan(last_valid):
                        series[i] = last_valid
                else:
                    last_valid = series[i]
            # Back-fill remaining NaNs at the start
            first_valid_idx = np.argmax(~np.isnan(series))
            if first_valid_idx > 0:
                series[:first_valid_idx] = series[first_valid_idx]

    # --- Build 38-d daily changes ---
    iv_flat = surfaces.reshape(-1, 25)  # (5822, 25)
    iv_changes = np.diff(iv_flat, axis=0)  # (5821, 25)
    factor_returns_trimmed = aligned_returns[1:]  # (5821, 13) — drop first to match diff

    joint_changes = np.concatenate(
        [iv_changes, factor_returns_trimmed], axis=1
    )  # (5821, 38)

    # --- Training set statistics ---
    # In the changes array, index i corresponds to the change from surface[i] to surface[i+1].
    # Train surfaces are 0:train_end, so train changes are 0:train_end-1.
    train_changes = joint_changes[: train_end - 1]
    train_mean_38 = train_changes.mean(axis=0)
    train_std_38 = train_changes.std(axis=0) + 1e-8

    train_iv_changes = iv_changes[: train_end - 1]
    train_mean_25 = train_iv_changes.mean(axis=0)
    train_std_25 = train_iv_changes.std(axis=0) + 1e-8

    column_names = IV_CELL_NAMES + return_columns

    return {
        "dates": iv_dates.values,
        "surfaces": surfaces,
        "iv_changes_25": iv_changes,
        "factor_returns_13": aligned_returns,
        "factor_levels_13": aligned_levels,
        "factor_return_columns": return_columns,
        "factor_level_columns": level_columns,
        "joint_changes_38": joint_changes,
        "train_mean_38": train_mean_38,
        "train_std_38": train_std_38,
        "train_mean_25": train_mean_25,
        "train_std_25": train_std_25,
        "column_names": column_names,
        "alignment_indices": alignment_indices,
    }


def make_windows(data, history_len=30, future_len=30, start_idx=0, end_idx=None):
    """Create sliding windows from aligned data dict.

    Args:
        data: dict from load_aligned_38d_data()
        history_len, future_len: window sizes
        start_idx, end_idx: surface-level indices for train/val/test split

    Returns dict with:
        history_changes: (N_windows, history_len, 38) daily changes
        future_changes: (N_windows, future_len, 38) daily changes
        anchor_surfaces: (N_windows, 5, 5) last known surface per window (in [0, 1])
        history_surfaces: (N_windows, history_len+1, 5, 5) surface levels for IV reconstruction
        history_factor_levels: (N_windows, history_len, 13) factor levels for regime features
    """
    surfaces = data["surfaces"]
    joint_changes = data["joint_changes_38"]
    factor_levels = data["factor_levels_13"]

    if end_idx is None:
        end_idx = len(surfaces)

    seq_len = history_len + future_len
    # Changes array is 1 shorter than surfaces. Change[i] = surface[i+1] - surface[i].
    # A window starting at surface index s needs:
    #   history changes: changes[s : s + history_len]      (history_len changes)
    #   future changes:  changes[s + history_len : s + seq_len]  (future_len changes)
    # This requires s + seq_len <= len(changes) = len(surfaces) - 1
    # And the anchor surface is surfaces[s + history_len].

    windows = {
        "history_changes": [],
        "future_changes": [],
        "anchor_surfaces": [],
        "history_surfaces": [],
        "history_factor_levels": [],
    }

    for s in range(start_idx, min(end_idx, len(joint_changes) - seq_len + 1)):
        windows["history_changes"].append(joint_changes[s : s + history_len])
        windows["future_changes"].append(
            joint_changes[s + history_len : s + seq_len]
        )
        # Anchor is the surface at the forecast origin
        windows["anchor_surfaces"].append(surfaces[s + history_len])
        # History surfaces: s to s + history_len inclusive (history_len + 1 surfaces)
        windows["history_surfaces"].append(surfaces[s : s + history_len + 1])
        # Factor levels during history window
        windows["history_factor_levels"].append(
            factor_levels[s + 1 : s + history_len + 1]
        )

    return {k: np.array(v) for k, v in windows.items()}


if __name__ == "__main__":
    # Smoke test
    print("Loading aligned 38-d data...")
    data = load_aligned_38d_data()

    print(f"\nShapes:")
    for key, val in data.items():
        if isinstance(val, np.ndarray):
            print(f"  {key}: {val.shape} {val.dtype}")
        elif isinstance(val, list):
            print(f"  {key}: list[{len(val)}]")

    # Verify alignment via SPX returns
    iv_data = np.load(DATA_DIR / "vol_surface_with_ret.npz")
    iv_ret = iv_data["ret"]
    mf_spx = data["factor_returns_13"][:, 0]  # SPX logret is column 0
    mask = (iv_ret != 0) & (mf_spx != 0)
    corr = np.corrcoef(iv_ret[mask], mf_spx[mask])[0, 1]
    print(f"\nAlignment check: SPX return correlation = {corr:.6f}")

    # Verify IV reconstruction
    surfaces = data["surfaces"]
    iv_changes = data["iv_changes_25"]
    reconstructed = surfaces[0].reshape(25) + np.cumsum(iv_changes[:30], axis=0)
    actual = surfaces[1:31].reshape(30, 25)
    max_err = np.abs(reconstructed - actual).max()
    print(f"IV reconstruction max error: {max_err:.2e}")

    # Train stats
    print(f"\nTrain stats (38-d):")
    print(f"  mean range: [{data['train_mean_38'].min():.6f}, {data['train_mean_38'].max():.6f}]")
    print(f"  std range:  [{data['train_std_38'].min():.6f}, {data['train_std_38'].max():.6f}]")

    # Window test
    print(f"\nCreating test windows (4540+)...")
    test_windows = make_windows(data, start_idx=4540)
    for key, val in test_windows.items():
        print(f"  {key}: {val.shape}")

    print("\nAll checks passed.")
