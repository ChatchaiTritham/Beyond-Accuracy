# Beyond-Accuracy — A Simulation-Based Evaluation Framework for Safety-Critical CDSS

Companion code for the manuscript:

> **Beyond Accuracy: A Simulation-Based Evaluation Framework for Safety-Critical
> Clinical Decision Support Systems**
> (target: *Machine Learning*, Springer)

The paper develops a pre-deployment safety evaluation framework that scores a
clinical AI on calibration and harm-aware metrics — with formal guarantees — to
surface failure modes before patient contact, rather than relying on aggregate
accuracy.

## Scope of this repository (please read)

This repository currently contains the **figure-generation scripts** behind the
diagrams and the graphical abstract in the manuscript. It is a *methods /
figure* repository, **not yet a full reproducibility package**: there is no
single seeded driver that regenerates the reported empirical tables from raw
data. Values drawn by these scripts come from the study's analysis and are
included for figure transparency, not recomputed at plot time.

A `run_all.py` driver (seed 42) that recomputes every reported metric into
committed `results/*.csv` is planned before journal submission. The underlying
simulation/metrics library is the companion project
[BASICS-CDSS](https://github.com/ChatchaiTritham/BASICS-CDSS).

## Contents

| Path | Contents |
|------|----------|
| `figures/generate_figures.py` | Main manuscript figures |
| `figures/generate_figures_drawio.py` | draw.io source export for diagrams |
| `figures/graphical_abstract.py` | Graphical abstract composition |

## Usage

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python figures/generate_figures.py
```

## Citation

```bibtex
@article{tritham_beyond_accuracy,
  title  = {Beyond Accuracy: A Simulation-Based Evaluation Framework for
            Safety-Critical Clinical Decision Support Systems},
  author = {Tritham, Chatchai and Snae Namahoot, Chakkrit},
  year   = {2026},
  note   = {Manuscript under review}
}
```

Licensed under the MIT License (see `LICENSE`).
