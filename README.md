# Topological Signal Processing for ETFs

Hodge decomposition of ETF return flows on a correlation complex. Extracts **harmonic component** – an arbitrage‑free persistent signal not captured by traditional momentum or cyclical models.

## Features
- Three ETF universes (FI/Commodities, Equity Sectors, Combined)
- Seven rolling windows (63 to 5040 days)
- Graph Helmholtzian decomposition (Lim 2020) yielding gradient, curl, and harmonic flow
- Per‑ETF harmonic divergence score
- Results stored on Hugging Face Datasets (`P2SAMAPA/p2-etf-topological-signal-results`)
- Streamlit dashboard showing top ETFs by harmonic signal

## Usage

1. **Set HF_TOKEN** as environment variable or in GitHub secrets.
2. **Run training locally**:
   ```bash
   python train.py
