import os
import json
from datetime import datetime
import pandas as pd
import numpy as np
from huggingface_hub import HfApi
import config
import data_manager as dm
from hodge_utils import (
    build_graph_from_returns,
    incidence_matrix,
    compute_edge_flow,
    hodge_decomposition,
    get_node_scores
)


def run_for_window(returns, window_days, top_frac):
    """
    Run Hodge decomposition on a rolling window of returns.

    Parameters:
    -----------
    returns : pd.DataFrame
        Wide DataFrame of log returns (columns = tickers, index = date).
    window_days : int
        Number of days to include in the window (takes the last `window_days` rows).
    top_frac : float
        Fraction of strongest correlation edges to keep when building the graph.

    Returns:
    --------
    dict or None
        Dictionary with window results, or None if not enough data or no edges.
    """
    if len(returns) < window_days:
        return None

    # Take the last `window_days` rows
    ret_window = returns.iloc[-window_days:]

    # Build graph and edge list
    G, edge_list, nodes = build_graph_from_returns(ret_window, top_edge_fraction=top_frac)
    if len(edge_list) == 0:
        return None

    # Incidence matrix
    B = incidence_matrix(G, nodes, edge_list)

    # Edge flow vector (return differences on last day)
    f = compute_edge_flow(ret_window, edge_list)

    # Hodge decomposition
    grad, curl, harmonic, potential = hodge_decomposition(
        B, f, eps=config.EPS, max_iter=config.MAX_ITER
    )

    # Node scores = divergence of harmonic flow
    node_scores = get_node_scores(harmonic, B)

    # Build per‑ticker score dictionary
    score_dict = {ticker: float(node_scores[i]) for i, ticker in enumerate(nodes)}

    # Top 3 ETFs by absolute harmonic score
    sorted_scores = sorted(score_dict.items(), key=lambda x: abs(x[1]), reverse=True)
    top_etfs = [{"ticker": t, "harmonic_score": s} for t, s in sorted_scores[:3]]

    return {
        "window": window_days,
        "top_etfs": top_etfs,
        "all_scores": score_dict,
        "n_nodes": len(nodes),
        "n_edges": len(edge_list)
    }


def main():
    """Main training loop: load data, iterate over universes and windows, save results."""
    print("Loading master data via data_manager...")
    # Preload data into cache (optional)
    dm.load_master_data()

    # Prepare results container
    results = {
        "run_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "windows": config.WINDOWS,
        "universes": {}
    }

    # Iterate over each universe defined in config
    for uni_name in config.UNIVERSES.keys():
        print(f"Processing {uni_name}...")
        returns = dm.get_universe_returns(uni_name)
        if returns.empty:
            print(f"  No data for {uni_name} -> skipping")
            continue

        uni_results = []
        for w in config.WINDOWS:
            print(f"  Window {w} days")
            out = run_for_window(returns, w, config.TOP_EDGE_FRACTION)
            if out:
                uni_results.append(out)
            else:
                print(f"    Not enough data or no edges for window {w}")

        results["universes"][uni_name] = uni_results

    # Save results locally
    os.makedirs("output", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = f"output/topological_{timestamp}.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved locally: {out_file}")

    # Upload to Hugging Face dataset repo
    api = HfApi(token=config.HF_TOKEN)
    try:
        api.upload_file(
            path_or_fileobj=out_file,
            path_in_repo=os.path.basename(out_file),
            repo_id=config.OUTPUT_REPO,
            repo_type="dataset"
        )
        print(f"Uploaded to {config.OUTPUT_REPO}")
    except Exception as e:
        print(f"Upload failed: {e}")


if __name__ == "__main__":
    main()
