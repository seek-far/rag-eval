# Docs

This folder contains the shareable experiment summaries and the small scripts used to regenerate their figures.

## Experiment summaries

- [Natural Questions ablation summary (400 samples)](nq_400_ablation_summary.md)
- [SciFact ablation summary (300 samples)](scifact_300_ablation_summary.md) - includes biomedical/LLM cross-encoder runs, LLM concurrency timing, failure-overlap analysis, offline fusion replay, and learned top-1 fusion.

## Regenerating figures

From the project root:

```bash
python docs/plot_nq_ablation.py
python docs/plot_scifact_ablation.py
```

If you are using the bundled Windows virtual environment in this repo, use:

```powershell
venv-win\Scripts\python.exe docs\plot_nq_ablation.py
venv-win\Scripts\python.exe docs\plot_scifact_ablation.py
```

Generated images are saved under `docs/figures/`.
