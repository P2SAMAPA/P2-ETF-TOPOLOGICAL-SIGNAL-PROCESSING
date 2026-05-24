import streamlit as st
import pandas as pd
import json
from huggingface_hub import HfFileSystem
import config
from us_calendar import next_trading_day

st.set_page_config(page_title="Topological Signal Processing", layout="wide")
st.markdown('<h1 style="text-align: center;">🌀 Topological Signal Processing</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center;">Hodge decomposition of ETF return flows | Gradient + Curl + Harmonic | Harmonic = arbitrage‑free persistent signal</p>', unsafe_allow_html=True)

st.sidebar.markdown("## 🧩 Topological Signals")
st.sidebar.markdown(f"**Run Date:** `{st.session_state.get('run_date', 'Not loaded')}`")
st.sidebar.markdown(f"**Next Trading Day:** `{next_trading_day()}`")
st.sidebar.markdown(f"**Windows evaluated:** {', '.join(map(str, config.WINDOWS))} days")
st.sidebar.markdown("**Method:** Graph Helmholtzian (Lim 2020)")

OUTPUT_REPO = config.OUTPUT_REPO
HF_TOKEN = config.HF_TOKEN

@st.cache_data(ttl=3600)
def list_repo_files():
    fs = HfFileSystem(token=HF_TOKEN)
    try:
        files = [f['name'] for f in fs.ls(f"datasets/{OUTPUT_REPO}", detail=True, recursive=True) if f['type'] == 'file']
        return files
    except Exception as e:
        return [f"Error: {e}"]

def find_latest_json(files):
    json_files = [f for f in files if f.endswith('.json') and 'topological_' in f]
    if not json_files:
        return None
    json_files.sort(reverse=True)
    return json_files[0]

@st.cache_data(ttl=3600)
def load_json(path):
    fs = HfFileSystem(token=HF_TOKEN)
    try:
        with fs.open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        return {"error": str(e)}

files = list_repo_files()
latest = find_latest_json(files)
if not latest:
    st.error("No results found. Run trainer first.")
    st.stop()

data = load_json(latest)
if "error" in data:
    st.error(f"Error: {data['error']}")
    st.stop()

st.session_state['run_date'] = data['run_date']

st.header("🚀 Top ETFs by Harmonic Signal Strength")
with st.expander("📖 Interpretation", expanded=True):
    st.markdown("""
    - **Hodge decomposition** splits edge flows (differences in ETF returns) into three orthogonal components:
      - **Gradient**: flow driven by a global potential (like a market index).
      - **Curl**: rotational/cyclic flow patterns.
      - **Harmonic**: flow that is both divergence‑free and curl‑free → persistent, arbitrage‑free circulation.
    - The **harmonic component** cannot be reduced to a single score or pure cycles; it represents multi‑asset relative value.
    - The displayed score is the **divergence of the harmonic flow** at each ETF node. High absolute values indicate strong harmonic involvement.
    - For each universe and window, we show the **3 ETFs with largest absolute harmonic divergence**.
    """)

for universe_name, uni_results in data["universes"].items():
    st.markdown(f'<h2 style="font-size: 1.5rem;">{universe_name.replace("_", " ").title()}</h2>', unsafe_allow_html=True)
    windows_avail = [res["window"] for res in uni_results]
    sel_window = st.selectbox(f"Select window for {universe_name}", windows_avail, key=universe_name)
    res = next(r for r in uni_results if r["window"] == sel_window)
    top3 = res["top_etfs"]
    cols = st.columns(3)
    for idx, etf in enumerate(top3):
        with cols[idx % 3]:
            st.markdown(f"""
            <div style="background: #f0f2f6; padding: 12px; border-radius: 8px; margin: 5px; text-align: center;">
                <strong>{etf['ticker']}</strong><br>
                Harmonic score: {etf['harmonic_score']:.6f}
            </div>
            """, unsafe_allow_html=True)
    with st.expander(f"Full ranking for {universe_name} (window {sel_window}d)"):
        all_scores = res["all_scores"]
        df_full = pd.DataFrame(list(all_scores.items()), columns=["Ticker", "Harmonic Score"])
        df_full = df_full.sort_values("Harmonic Score", ascending=False)
        st.dataframe(df_full, use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.caption("Topological Signal Processing | Hodge decomposition on ETF correlation complex")
