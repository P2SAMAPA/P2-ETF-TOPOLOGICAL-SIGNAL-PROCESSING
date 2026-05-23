import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from huggingface_hub import HfApi, HfFileSystem
import config
from hodge_utils import (
    build_graph_from_returns, incidence_matrix, compute_edge_flow,
    hodge_decomposition, get_node_scores
)

def load_master_data():
    """Load master parquet from HF dataset."""
    fs = HfFileSystem(token=config.HF_TOKEN)
    path = f"datasets/{config.DATA_REPO}/master_data.parquet"
    with fs.open(path, "rb") as f:
        df = pd.read_parquet(f)
    return df

def compute_returns(df, tickers):
    """Pivot daily close, compute log returns."""
    df_sub = df[df['ticker'].isin(tickers)].copy()
    df_sub = df_sub.sort_values(['ticker', 'date'])
    pivot = df_sub.pivot(index='date', columns='ticker', values='close')
    returns = np.log(pivot / pivot.shift(1)).dropna()
    return returns

def run_for_window(returns, window_days, universe_name, top_frac):
    """Run Hodge decomposition for one rolling window."""
    if len(returns) < window_days:
        return None
    # Get last window_days of returns
    ret_window = returns.iloc[-window_days:]
    # Build graph
    G, edge_list, nodes = build_graph_from_returns(ret_window, top_edge_fraction=top_frac)
    if len(edge_list) == 0:
        return None
    B = incidence_matrix(G, nodes, edge_list)
    f = compute_edge_flow(ret_window, edge_list)
    grad, curl, harmonic, potential = hodge_decomposition(B, f, eps=config.EPS, max_iter=config.MAX_ITER)
    node_scores = get_node_scores(harmonic, B)
    # Create per-ticker score dict
    score_dict = {ticker: float(node_scores[i]) for i, ticker in enumerate(nodes)}
    # Top 3 by absolute score
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
    print("Loading master data...")
    df = load_master_data()
    df['date'] = pd.to_datetime(df['date'])
    results = {
        "run_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "windows": config.WINDOWS,
        "universes": {}
    }
    for uni_name, tickers in config.UNIVERSES.items():
        print(f"Processing {uni_name}...")
        returns = compute_returns(df, tickers)
        uni_results = []
        for w in config.WINDOWS:
            print(f"  Window {w} days")
            out = run_for_window(returns, w, uni_name, config.TOP_EDGE_FRACTION)
            if out:
                uni_results.append(out)
        results["universes"][uni_name] = uni_results
    # Save locally
    os.makedirs("output", exist_ok=True)
    out_file = f"output/topological_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    # Upload to HF
    api = HfApi(token=config.HF_TOKEN)
    try:
        api.upload_file(
            path_or_fileobj=out_file,
            path_in_repo=out_file.split("/")[-1],
            repo_id=config.OUTPUT_REPO,
            repo_type="dataset"
        )
        print(f"Uploaded to {config.OUTPUT_REPO}")
    except Exception as e:
        print(f"Upload failed: {e}")

if __name__ == "__main__":
    main()
