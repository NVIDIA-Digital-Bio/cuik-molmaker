# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Sample usage:
# python fit_distribution.py --desc_file ChEMBL_rdkit2D.csv --column_name MolWt --distribution_type continuous --output_folder ~/GNN/cuik-molmaker/rdkit2D_desc_normalization_acceleration/params
# python fit_distribution.py --desc_file ChEMBL_rdkit2D.csv --column_name fr_thiazole --distribution_type discrete --output_folder ~/GNN/cuik-molmaker/rdkit2D_desc_normalization_acceleration/params

import numpy as np
from scipy.stats._continuous_distns import _distn_names as continuous_distns
from scipy.stats._discrete_distns import _distn_names as discrete_distns
import scipy.stats as st
import argparse
import pandas as pd
import json
import warnings
import time
import os
from typing import List, Tuple

def parse_args():
    parser = argparse.ArgumentParser("Fit distribution")
    parser.add_argument("--desc_file", type=str, required=True, help="Name of .CSV descriptor file")
    parser.add_argument("--column_name", type=str, required=True, help="Name of column (property) to fit distribution")
    parser.add_argument("--n_samples", type=int, default=0, help="Number of samples for fitting distribution. Default: 0 (all samples)")
    parser.add_argument("--distribution_type", type=str, choices=("discrete", "continuous"),
                        required=True, help="Type of distribution to fit")
    parser.add_argument("--output_folder", type=str, required=True, help="Name of output folder")
    parser.add_argument("--tolerance", type=float, default=10, help="Tolerance for cheap distribution in percentage")
    return parser.parse_args()


def get_discrete_cdf(x: np.ndarray, y: np.ndarray) -> Tuple[str, List[float], float, float]:
    """
    Get the empirical CDF for discrete data.
    """
    cdf = np.zeros(x.max() - x.min() + 1, dtype=np.float64)
    cumsum_y = np.cumsum(y)
    curr_y = y[0]
    for i in range(x.min(), x.max() + 1):
        if i in x:
            xidx = x.searchsorted(i)
            cdf[i-x.min()] = cumsum_y[xidx]
            curr_y = cumsum_y[xidx]
        else:
            cdf[i-x.min()] = curr_y

    return ("discrete_cdf", cdf.tolist(), 
            0.0, # No error for discrete data
            0.0 # Negligible time for discrete data
            )

def best_fit_distribution(
    data: np.ndarray,
    distribution_type: str,
    bins: int = 200
) -> List[Tuple[str, List[float], float, float]]:
    """Find the best fitting probability distribution for the given data.
    
    This function attempts to fit various probability distributions to the input data
    and returns either the best fitting distributions (for continuous data) or the
    empirical CDF (for discrete data).
    
    Parameters
    ----------
    data : np.ndarray
        Input data array to fit distributions to
    distribution_type : str
        Type of distribution to fit, either "continuous" or "discrete"
    bins : int, optional
        Number of bins to use for histogram calculation in continuous case, by default 200
        
    Returns
    -------
    List[Tuple[str, List[float], float, float]]
        For continuous data: List of tuples containing (distribution name, parameters, SSE, time)
        For discrete data: Tuple containing (distribution name, empirical CDF, SSE, time)
        
    Raises
    ------
    ValueError
        If distribution_type is not "continuous" or "discrete"
    """
    # Validate distribution type
    if distribution_type not in ["continuous", "discrete"]:
        raise ValueError(f"Unknown data type: {distribution_type}")
    
    # Prepare data for fitting
    if distribution_type == "continuous":
        # Get histogram of original data
        y, x = np.histogram(data, bins=bins, density=True)
        x = (x + np.roll(x, -1))[:-1] / 2.0
    else:  # discrete
        x, y = np.unique(data, return_counts=True)
        y = y / y.sum()
        
        # For discrete data, return histogram CDF
        return get_discrete_cdf(x, y)

    # Get list of distributions to fit
    dist_names = continuous_distns if distribution_type == "continuous" else discrete_distns
    # Remove distributions that are too slow or problematic
    slow_distributions = {'levy_stable', 'studentized_range', 'nct', 'irwinhall'}

    dist_names = [d for d in dist_names if d not in slow_distributions]

    # Store best distributions
    best_distributions = []

    # Try fitting each distribution
    for ii, dist_name in enumerate(dist_names):
        print(f"{ii+1} / {len(dist_names)}: {dist_name}")
        distribution = getattr(st, dist_name)

        try:
            # Ignore warnings from data that can't be fit
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')

                # Fit distribution to data
                params = distribution.fit(data)
                arg = params[:-2]
                loc = params[-2]
                scale = params[-1]

                # Calculate fitted PDF and error
                pdf = distribution.pdf(x, loc=loc, scale=scale, *arg)
                sse = np.sum(np.power(y - pdf, 2.0))
                if np.isnan(sse):
                    continue

                # Measure CDF calculation time
                tic = time.time()
                distribution.cdf(x, loc=loc, scale=scale, *arg)  # Calculate CDF
                toc = time.time()

                best_distributions.append((dist_name, params, sse, toc-tic))

        except ValueError as e:
            print(f"Fitting to {dist_name} failed: {e}. Skipping...")

    # Sort by SSE (sum of squared errors)
    return sorted(best_distributions, key=lambda x: x[2])

def get_fast_distribution(df, tolerance):
    sorted_df = df.sort_values(by="sse", ascending=True)
    best_sse = sorted_df.iloc[0]["sse"]
    print(f"Best SSE: {best_sse}")
    sse_thresh = best_sse + best_sse * tolerance / 100
    print(f"SSE threshold: {sse_thresh}")
    fast_dist_df = sorted_df[sorted_df["sse"] <= sse_thresh]
    res = fast_dist_df[fast_dist_df["time"] == fast_dist_df["time"].min()].iloc[0]
    return res["function_name"], res["params"], res["sse"], res["time"]

if __name__ == "__main__":
    args = parse_args()
    assert args.tolerance < 100, "Tolerance must be less than 100%"

    desc_df = pd.read_csv(args.desc_file)

    if args.n_samples > 0:
        desc_df = desc_df.sample(n=args.n_samples)
        print(f"Sampling {len(desc_df)} random samples and processing them")
    else:
        print(f"Processing all {len(desc_df)} samples")

    data = desc_df[args.column_name].values

    print(f"Number of raw data points: {len(data)}")
    # Remove NaNs
    data = data[~np.isnan(data)]
    # Remove infs
    data = data[~np.isinf(data)]
    print(f"Number of data after removing NaNs and infs: {len(data)}")

    best_dist = best_fit_distribution(data, args.distribution_type)

    if args.distribution_type == "continuous":
        best_dist_df = pd.DataFrame(best_dist, columns=["function_name", "params", "sse", "time"])
        best_dist_df.sort_values(by="sse", ascending=True, inplace=True)
        best_row = best_dist_df.iloc[0]
    elif args.distribution_type == "discrete":
        best_row = {"function_name": best_dist[0], "params": best_dist[1], "sse": best_dist[2], "time": best_dist[3]}
    else:
        raise ValueError(f"Unknown distribution type: {args.distribution_type}")

    # Compute stats for data
    data_min = data.min().astype(np.float64)
    data_max = data.max().astype(np.float64)
    data_mean = data.mean().astype(np.float64)
    data_std = data.std().astype(np.float64)
    
    print(f"Best distribution: {best_row['function_name']}")
    print(f"SSE: {best_row['sse']:.2e}")
    print(f"Time: {best_row['time']:.4e}")
    best_dist_dict = { args.column_name: {
        "normalization_type": args.distribution_type,
        "normalization_name": best_row["function_name"],
        "params": best_row["params"],
        "data_stats": {
        "data_min": data_min,
        "data_max": data_max,
        "data_mean": data_mean,
        "data_std": data_std}}}
    

    if args.distribution_type == "continuous":
        fast_dist = get_fast_distribution(best_dist_df, args.tolerance)
        print(f"Fast distribution: {fast_dist[0]}")
        print(f"SSE: {fast_dist[2]:.2e}")
        print(f"Time: {fast_dist[3]:.4e}")

        fast_dist_dict = { args.column_name: {
            "normalization_type": args.distribution_type,
            "normalization_name": fast_dist[0],
            "params": fast_dist[1],
            "data_stats": {
            "data_min": data_min,
            "data_max": data_max,
            "data_mean": data_mean,
            "data_std": data_std}}}
    elif args.distribution_type == "discrete":
        # Cheapest distribution is the same as the best distribution for discrete data
        fast_dist_dict = best_dist_dict
    
    print(f"Saving best and cheap distributions to {args.output_folder}")
    with open(os.path.join(args.output_folder, f"best_{args.column_name}.json"), "w") as f:
        json.dump(best_dist_dict, f)
    with open(os.path.join(args.output_folder, f"fast_{args.column_name}.json"), "w") as f:
        json.dump(fast_dist_dict, f)
    