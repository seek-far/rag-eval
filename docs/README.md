# Docs

This folder contains the shareable experiment summaries and the small scripts used to regenerate their figures.

## Experiment summaries

- [Natural Questions ablation summary (400 samples)](nq_400_ablation_summary.md)
- [SciFact ablation summary (400 samples)](scifact_400_ablation_summary.md)

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
