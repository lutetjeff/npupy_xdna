# heuristic/
PoC 1 — Heuristic-driven template selection and offload decision.
- `classifier.py` — RegionClassifier: maps op patterns to templates via rules.yaml
- `rules.yaml` — declarative template selection rules
- `cost_model.py` — empirical cost predictors per template + confidence intervals
- `offload.py` — OffloadHeuristic: NPU vs CPU decision with margin threshold
- `viz.py` — matplotlib plot helpers for throughput curves, crossover, decision maps
