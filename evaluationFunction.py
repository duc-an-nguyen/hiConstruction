# =============================================
# IMPORT
# =============================================
from utility.logger         import aiLogger
from preProcessingDataTEP   import PreProcessData
from prettytable            import PrettyTable

from utility.matpilotDraw   import MatPlot
from sklearn.preprocessing  import MinMaxScaler

import pandas               as pd
import matplotlib.pyplot    as plt
import numpy                as np
import os, warnings
import random
import json



# ==============================
#           MONOTONICITY
# ==============================
def proportion_based_monotonicity(hi_series, segment_size=32):
    """
    Calculate proportion-based monotonicity using segments.

    Parameters:
        hi_series (pd.Series)   : Health indicator series (e.g., HI values over time).
        segment_size (int)      : Size of each segment.

    Returns:
        float                   : Proportion-based monotonicity (range 0 to 1).
    """
    hi_series = hi_series.dropna()  # Remove NaN values
    
    if len(hi_series) < segment_size * 2:
        raise ValueError("Not enough data to calculate monotonicity with the specified segment size.")

    # ----------------------------------
    # Step 1: Divide the data into segments
    # and calculate the mean of each segment
    # -----------------------------------
    num_segments    = len(hi_series) // segment_size
    aggregated      = [hi_series.iloc[i * segment_size:(i + 1) * segment_size].mean() for i in range(num_segments)]


    # ----------------------------------
    # Step 2: Calculate the proportion
    # of consistent steps
    # ----------------------------------
    diffs           = pd.Series(aggregated).diff().dropna()  # Calculate differences between segments
    positive_steps  = (diffs > 0).sum()
    negative_steps  = (diffs < 0).sum()

    # ----------------------------------
    # Step 3: Calculate monotonicity as
    # the proportion of consistent steps
    # ----------------------------------
    total_steps = len(diffs)
    monotonicity = abs((positive_steps - negative_steps) / total_steps)  # Normalize to 0-1
    return monotonicity

def calculate_monotonicity(hi_series: pd.Series) -> float:
    """
    Calculate a single‐trajectory monotonicity:
        Mon = |#(∆HI > 0) − # (∆HI < 0)|  /  (n − 1)

    Parameters:
        hi_series (pd.Series): A single HI trajectory (ordered by time).

    Returns:
        float: Monotonicity in [0, 1].
    """
    hi_series = hi_series.dropna()
    n = len(hi_series)
    if n < 2:
        return 0.0

    # Compute first differences
    diffs = hi_series.diff().dropna()
    pos_steps = (diffs > 0).sum()
    neg_steps = (diffs < 0).sum()

    return abs(pos_steps - neg_steps) / (n - 1)


# ==============================
#           ROBUSTNESS
# ==============================
def calculate_robustness_polyfit(hi_series, degree=25):
    """
    Calculate robustness using polynomial fitting for trend estimation.

    Parameters:
        hi_series (pd.Series)   : Health indicator series (e.g., HI values over time).
        degree (int)            : Degree of the polynomial for trend fitting.

    Returns:
        float                   : Robustness score (range 0 to 1).
    """
    hi_series   = hi_series.dropna()  # Remove NaN values
    n           = len(hi_series)
    if n < degree + 1:
        return 0.0  # Not enough data to fit the polynomial

    # --------------------------------
    # Step 1: Fit a polynomial trend
    # --------------------------------
    x               = np.arange(n)
    y               = hi_series.values
    poly_coeffs     = np.polyfit(x, y, degree)  # Fit polynomial
    poly_trend      = np.polyval(poly_coeffs, x)  # Evaluate polynomial trend

    # --------------------------------
    # Step 2: Calculate deviations and 
    # normalize by standard deviation
    # --------------------------------
    deviations      = abs(hi_series - poly_trend)
    std_dev         = hi_series.std()
    if std_dev == 0:
        return 1.0  # Perfect stability, no variance

    normalized_deviations = deviations / std_dev

    # --------------------------------
    # Step 3: Compute robustness
    # --------------------------------
    robustness      = np.exp(-normalized_deviations.mean())
    return robustness

def chunked_robustness(hi_series, chunk_size =10):
    hi_series = hi_series.dropna().reset_index(drop=True)
    n = len(hi_series)
    if n < chunk_size * 2:
        return 0.0

    num_chunks = n // chunk_size
    chunk_means = [hi_series[i * chunk_size:(i + 1) * chunk_size].mean()
                   for i in range(num_chunks)]
    
    chunk_series = pd.Series(chunk_means)
    std = chunk_series.std()
    if std <= 1e-7:
        return 1.0  # perfect stability
    
    norm_diff = np.abs((chunk_series - chunk_series.mean()) / std)
    return np.exp(-norm_diff.mean())



def robustness_single_hi(hi_series: pd.Series) -> float:
    """
    Calculate robustness for a single health indicator series.

    Parameters:
        hi_series (pd.Series): A single health indicator trajectory.

    Returns:
        float: Robustness value in [0, 1].
    """

    hi_series = hi_series.dropna()
    if len(hi_series) == 0:
        raise ValueError("Input series is empty after dropping NaN values.")

    std = hi_series.std()
    if std <= 1e-7:
        raise ValueError("Standard deviation is too small, indicating no variability in the series.")

    mean_hi             = hi_series.mean()
    normalized_diff     = np.abs((hi_series - mean_hi) / std)
    rob                 = np.exp(-normalized_diff.mean())
    return rob


# ==============================
#           TRENDABILITY
# ==============================
def calculate_trendability(hi_series, time=None, time_step=0.5):
    """
    Calculate trendability using the absolute Pearson correlation between time and HI,
    based on the interpretation in the referenced paper.

    Parameters:
        hi_series (pd.Series): Health indicator series (e.g., HI values over time).
        time (array-like, optional): Time values aligned with the series.
        time_step (float): Used to generate uniform time if 'time' is not provided.

    Returns:
        float: Trendability score in [0, 1].
    """
    time = None
    hi_series = hi_series.dropna()
    n = len(hi_series)
    if n < 2:
        raise ValueError("Not enough data to calculate trendability.")

    time = np.arange(n) * time_step if time is None else np.array(time)
    if len(time) != n:
        raise ValueError("Time and HI series must be of the same length.")

    correlation = np.corrcoef(time, hi_series)[0, 1]
    return abs(correlation)


# ---------------------------------------
# Compute robustness, trendability, 
# and monotonicity for a single HI trajectory.
# ---------------------------------------
def compute_hi_metrics_for_id(df: pd.DataFrame, hiCol = "HI") -> dict:
    """
    Parameters:
        df (pd.DataFrame): DataFrame with 'cycle' and 'HI' columns for one engine ID.

    Returns:
        dict: Dictionary with metrics.
    """
    # hi_series = df.sort_values("cycle")["HI"]
    hi_series = df[hiCol]
    try:
        robustness = chunked_robustness(hi_series)
    except ValueError:
        robustness = np.nan

    try:
        trendability = calculate_trendability(hi_series, time = df["cycle"])
    except ValueError:
        trendability = np.nan

    try:
        monotonicity = proportion_based_monotonicity(hi_series)
        # print(f"Monotonicity for ID {df['id'].iloc[0]}: {monotonicity}")
    except ValueError:
        monotonicity = np.nan

    return {
        "robustness": robustness,
        "trendability": trendability,
        "monotonicity": monotonicity
    }

# ---------------------------------------
# Compute the average robustness, trendability, 
# and monotonicity over all HI trajectories.
# ---------------------------------------
def compute_average_hi_metrics(dfHI, hiCol = "HI"):
    """
    Parameters:
        dfHI (pd.DataFrame): DataFrame with columns 'id', 'cycle', and 'HI'.

    Returns:
        dict: Dictionary with average metric values.
    """
    grouped         = dfHI.groupby("id")
    metrics_list    = [compute_hi_metrics_for_id(group, hiCol) for _, group in grouped]

    # Convert to DataFrame for easy averaging and NaN handling
    metrics_df      = pd.DataFrame(metrics_list)

    return {
        "avg_robustness": metrics_df["robustness"].mean(),
        "avg_trendability": metrics_df["trendability"].mean(),
        "avg_monotonicity": metrics_df["monotonicity"].mean()
    }

