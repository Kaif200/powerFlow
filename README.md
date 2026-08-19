# AI-OPF — Intelligent Optimal Power Flow Platform

A research dashboard that runs **real** OPF solves (pandapower/PYPOWER) on the
IEEE 14-bus benchmark, trains a scikit-learn AI-OPF surrogate on the verified
solutions, and reports measured — not fabricated — performance metrics.

## Files

```
app.py              Streamlit dashboard (UI + controls)
aiopf_core.py        Pipeline: scenario generation, OPF ground truth, ML training, evaluation
requirements.txt     Python dependencies
.streamlit/config.toml   Dark theme matching the AI-OPF visual identity
```

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Opens at `http://localhost:8501`. Click **START TRAINING** in the sidebar to
run a live experiment (real OPF solves + real model training, typically
10–90s depending on scenario count).

---

## Deploy — Streamlit Community Cloud (free, easiest)

1. Push this folder to a public (or private, on paid tiers) GitHub repo.
2. Go to https://share.streamlit.io → **New app**.
3. Point it at your repo, branch, and `app.py` as the entrypoint.
4. Deploy. You'll get a URL like `https://your-app.streamlit.app`.

No server management, free tier is sufficient for scenario counts up to a
few hundred (CPU-only; no `numba` acceleration by default — see note below).

---

## Deploy — Hugging Face Spaces (free, good for ML demos)

1. Create a new Space → SDK: **Streamlit**.
2. Upload `app.py`, `aiopf_core.py`, `requirements.txt`, `.streamlit/config.toml`
   (or push via `git` — Spaces are git repos).
3. The Space builds and launches automatically at
   `https://huggingface.co/spaces/<you>/<space-name>`.

Slightly more generous free compute than Streamlit Cloud, and better suited
if you later add the PyTorch/Gymnasium RL stage.

---

## Deploy — Render / Railway / Fly.io (more control, still simple)

These platforms run arbitrary containers/processes instead of just Streamlit apps —
use this route once you split the app into a FastAPI backend + separate frontend,
or if you want a custom domain and persistent storage (e.g. to keep trained models
between restarts, since the free tiers above have ephemeral filesystems).

Minimal `Procfile` / start command for any of these:

```bash
streamlit run app.py --server.port $PORT --server.address 0.0.0.0
```

---

## Deploy — Docker (self-hosted / VPS)

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

```bash
docker build -t ai-opf .
docker run -p 8501:8501 ai-opf
```

Deploy the image to any VPS, AWS ECS/EC2, GCP Cloud Run, or a college server.

---

## Performance note

OPF solves run without `numba` in most minimal environments — install it
(`pip install numba`) on your deploy target for a large speedup, letting you
push `Number of Scenarios` toward the spec's target of 10,000 without long
wait times. Without it, keep scenario counts in the low hundreds for
responsive interactive use.

## Roadmap (not yet implemented — no fabricated metrics for these)

- Stage 2: Gymnasium + Stable-Baselines3 PPO policy for sequential dispatch
- SHAP-based explainability panel
- IEEE 30/57-bus and additional benchmark cases
- Contingency-N-1 scenario library
