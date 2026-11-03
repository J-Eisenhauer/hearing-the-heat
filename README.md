# Hearing the heat: code and data

Analysis code for

> Eisenhauer, J. Hearing the heat: a proof-of-concept study of nonlinear sound measures for distinguishing
> phases of rapid organic decomposition.

The code reproduces every number, table and figure of the article and its supplement from the 150 analysed
recordings and the measurement protocol. The recordings are archived separately on Zenodo (see Data).

| | |
|---|---|
| Development | https://github.com/J-Eisenhauer/hearing-the-heat |
| Data (audio, CSV files) | https://doi.org/10.5281/zenodo.23891122 |

## Contents

| File | Purpose |
|---|---|
| `settings.py` | Paths, experimental design and all analysis settings (bands, measure parameters, sensitivity scenarios, seeds) |
| `audio_measures.py` | Cropping of the raw recordings; nonlinear measures (PE, LZC78, RSE4, RSR) and conventional indices (ACI, AEI, log band energy, spectral entropy, spectral flatness, temporal entropy) per 15 s window and octave band; broadband indices; mains-hum score |
| `signal_surrogates.py` | Signal-level IAAFT surrogates of the recordings and the control on synthetic linear signals |
| `models.py` | Temperature and time models, leave-one-bucket-out cross-validation, NP benchmark, IAAFT of the temperature course |
| `analysis.py` | Model comparison, temperature-surrogate test, diagnostics, redundancy, sensitivity analysis, broadband indices and further checks |
| `bootstrap.py` | Bayesian bootstrap of the fitted trajectories, the L coefficients and the difference between the heating and cooling branches |
| `loudness_control.py` | Level-invariant control: LZC78, RSE4 and RSR with quantisers fitted per recording (and two checks), models and Bayesian bootstrap as in the main analysis |
| `mass_balance.py` | Mass and moisture balance (Monte Carlo) |
| `tables.py` | LaTeX tables |
| `figures.py` | Figures |
| `run_all.py` | Runs all steps or selected ones |
| `graphical_abstract.py` | Graphical abstract (run separately after `run_all.py`) |

## Data

The audio is not part of this repository. The 150 analysed excerpts (10 min each, cropped from the full
recordings; 8 GB) are archived on Zenodo, together with the two CSV files below and a data README
(`data/README.md`) that describes the recordings in detail:

> Eisenhauer, J. (2026). *Hearing the heat: soil probe recordings of rapid organic decomposition (150 x 10 min)* [Data set]. Zenodo.
> https://doi.org/10.5281/zenodo.23891122

Download the five zip archives (`sample_A_recordings.zip` to `sample_E_recordings.zip`) and unpack all of them into `data/audio/`
(without subfolders) before running the analysis. The full, uncropped
recordings are not archived, so the `crop` step cannot be repeated from the archive; `data/recordings.csv`
documents the crop positions (source file, crop start, recorder start time).

```
data/
  measurement_protocol.csv   date, session, start time and internal temperature of buckets A-E per occasion
  recordings.csv             one row per analysed recording: source file, crop start, recorder start time
  audio/                     from Zenodo: the 150 analysed 600 s excerpts, <bucket><occasion>_<D|N|Baseline>_cropped.wav
                             (session codes: D = afternoon session, 13:00; N = evening session, 21:00; Baseline = 18:00)
```

The results start from the excerpts in `data/audio`.

`results_reference/` holds the outputs of one complete run (tables, figures, CSV and report files) for comparison with your own rerun. The intermediate caches (`results/measures/cache`, `results/loudness_control/cache`) and the bootstrap draw files (`*.npy`, reproducible from the seeds) are left out; `run_all.py` writes to `results/`, which is not tracked.

## Running

Python 3.12 with the packages pinned in `pyproject.toml` / `uv.lock`:

```
uv sync
uv run python run_all.py                 # all steps except crop
uv run python run_all.py analysis tables # selected steps, in the given order
```

or with pip: `pip install .` and `python run_all.py`. The graphical abstract is drawn afterwards with
`python graphical_abstract.py`.

| Step | Output | Runtime (8 cores) |
|---|---|---|
| `crop` | `data/audio`, `data/recordings.csv` (optional; needs the full recordings, not archived) | minutes |
| `measures` | `results/measures` | several hours |
| `surrogates` | `results/signal_surrogates` | several hours |
| `analysis` | `results/models` | 15-20 min |
| `bootstrap` | `results/bootstrap` | minutes |
| `loudness` | `results/loudness_control` | about 30 min |
| `mass` | `results/mass_balance` | seconds |
| `tables` | `results/tables` | seconds |
| `figures` | `results/figures` | under 1 min |

Each step reads only the outputs of earlier steps, so later steps can be rerun on their own. `measures` caches the
window values of each recording in `results/measures/cache` and resumes after an interruption. Each step writes a
`report.txt` with the numbers quoted in the text.

All random draws use fixed seeds (`settings.py` and the module headers). Results are reproducible on one machine;
across operating systems, floating-point differences in the FFT can change the last digits of individual
surrogate values.

Figures are written as PNG and as vector PDF (the PDFs are the article versions). They use the Inter font shipped in `fonts/` (SIL Open Font License, `fonts/OFL.txt`).

## Outputs and the article

| Article | File |
|---|---|
| Table 1 | `results/tables/table1_model_comparison.tex` |
| Tables S2-S10 | `results/tables/tableS2_sensitivity.tex` ... `tableS10_loudness_residuals.tex` |
| Fig. 1 | `results/figures/Figure_1.png` |
| Fig. 2 | `results/figures/Figure_2.png` |
| Fig. 3 | `results/figures/Figure_3.png` |
| Fig. 4 | `results/figures/Figure_4.png` |
| Fig. 5 | `results/figures/Figure_5.png` |
| Fig. 6 | `results/figures/Figure_6.png` |
| Figs. S1-S9 | `results/figures/Figure_S1.png` to `Figure_S9.png` (S5 in two parts, `Figure_S5a`/`Figure_S5b`; each also as .pdf) |
| Graphical abstract | `results/figures/Graphical_abstract.png` |
| Per-configuration results | `results/models/02_cv_per_config.csv` |
| Further checks (Supplementary Analyses) | `results/models/20_...` to `27_...` |

## Licence

Code: MIT. Data (audio on Zenodo, `data/*.csv`): CC BY 4.0. See `LICENSE`.

The fonts in `fonts/` are Inter (Copyright (c) 2016 The Inter Project Authors), licensed under the SIL Open Font License 1.1 (`fonts/OFL.txt`), separate from the MIT licence of the code.
