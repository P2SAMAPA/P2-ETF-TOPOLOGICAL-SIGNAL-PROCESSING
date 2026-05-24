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

def normalize_scores(score_dict):
    """Min-max normalization to [0,1]."""
    scores = np.array(list(score_dict.values()))
    min_s, max_s = scores.min(), scores.max()
    if max_s - min_s < 1e-12:
        return {k: 0.0 for k in score_dict}
    norm = (scores - min_s) / (max_s - min_s)
    return {ticker: float(norm[i]) for i, ticker in enumerate(score_dict.keys())}

def run_for_window(returns, window_days, top_frac):
    if len(returns) < window_days:
        return None
    ret_window = returns.iloc[-window_days:]
    G, edge_list, nodes = build_graph_from_returns(ret_window, top_edge_fraction=top_frac)
    if len(edge_list) == 0:
        return None
    B = incidence_matrix(nodes, edge_list)
    f = compute_edge_flow(ret_window, edge_list)
    grad, curl, harmonic, potential = hodge_decomposition(B, f, G, edge_list, nodes, eps=config.EPS)
    node_scores = get_node_scores(harmonic, B)  # raw scores
    score_dict = {ticker: float(node_scores[i]) for i, ticker in enumerate(nodes)}
    norm_dict = normalize_scores(score_dict)
    # Top 3 by normalized score
    sorted_norm = sorted(norm_dict.items(), key=lambda x: x[1], reverse=True)
    top_etfs = [{"ticker": t, "harmonic_score_norm": s, "raw_score": score_dict[t]} for t, s in sorted_norm[:3]]
    return {
        "window": window_days,
        "top_etfs": top_etfs,
        "all_scores_raw": score_dict,
        "all_scores_norm": norm_dict,
        "n_nodes": len(nodes),
        "n_edges": len(edge_list)
    }

def main():
    print("Loading master data...")
    dm.load_master_data()
    all_results = {
        "run_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "windows": config.WINDOWS,
        "universes": {}
    }
    for uni_name in config.UNIVERSES.keys():
        print(f"Processing {uni_name}...")
        returns = dm.get_universe_returns(uni_name)
        if returns.empty:
            print(f"  No data for {uni_name} -> skipping")
            continue
        per_window = []
        for w in config.WINDOWS:
            print(f"  Window {w} days")
            out = run_for_window(returns, w, config.TOP_EDGE_FRACTION)
            if out:
                per_window.append(out)
            else:
                print(f"    Not enough data or no edges for window {w}")
        # Find best window (highest max absolute raw score)
        best = None
        best_score = -np.inf
        best_data = None
        for pw in per_window:
            raw_scores = list(pw["all_scores_raw"].values())
            max_abs = max(abs(s) for s in raw_scores)
            if max_abs > best_score:
                best_score = max_abs
                best = pw["window"]
                best_data = pw
        if best_data:
            all_results["universes"][uni_name] = {
                "best_window": best,
                "best_window_data": {
                    "top_etfs": best_data["top_etfs"],
                    "all_scores_norm": best_data["all_scores_norm"],
                    "all_scores_raw": best_data["all_scores_raw"],
                    "n_nodes": best_data["n_nodes"],
                    "n_edges": best_data["n_edges"]
                },
                "all_windows": per_window   # kept for reference, not displayed
            }
        else:
            all_results["universes"][uni_name] = None
    # Save and upload
    os.makedirs("output", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = f"output/topological_{timestamp}.json"
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved locally: {out_file}")
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
