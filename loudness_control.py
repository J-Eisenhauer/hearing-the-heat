"""Level-invariant control: LZC78, RSE4 and RSR with the quantiser fitted per recording instead of per bucket.

Within one octave band, changes in sound attenuation by the material are expected to act mainly on the absolute level,
whereas faunal sounds also change the temporal pattern. Fitting the quantiser to the RMS envelope of each recording
removes the absolute level from the three quantiser-based measures; PE uses ranks only and is unchanged. The control
is descriptive: windows, standardisation, models, cross-validation and Bayesian bootstrap are those of the main
analysis, and no surrogate test is applied.

Quantisers
  bucket              one per bucket, pooled over all occasions (main analysis)
  recording           per recording, from its own 4000 evenly spaced RMS values (the control)
  window              per 15 s window (check)
  recording, median   per recording, evaluated only in the windows that held the median in the main analysis (check)

Window values are cached per recording in results/loudness_control/cache, so an interrupted run resumes. Writes the
values and Bayesian-bootstrap results of the control, the tables of Tables S9-S10 and Fig. S9, and report.txt to
results/loudness_control.
"""
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

import audio_measures as AM
import bootstrap as B
import models as M
import settings as S
from analysis import heldout, np_mean_fit

OUT = S.LOUDNESS_DIR
CACHE_DIR = OUT / "cache"
P = S.PRIMARY
QUANTISED = ["LZC78", "RSE4", "RSR"]                     # measures that depend on the quantiser
ALPHABETS = sorted({P["lzc_alphabet"], P["rse_alphabet"]})
QUANTISERS = ["bucket", "recording", "window", "recording, median"]
FIG_EXAMPLES = [("RSR", 128), ("RSR", 256), ("RSR", 1024), ("RSR", 4096)]      # Fig. S9
MID_BANDS = [1024, 2048, 4096]                           # 1-8 kHz
TOP_BAND = 8192
COLS = ["file", "bucket", "occasion", "measure", "fmin_hz"]


# =============================================================================
# Window values
# =============================================================================
def fit_edges(values):
    values = np.asarray(values, np.float64)
    return {a: AM.quantiser_edges(values, a) for a in ALPHABETS}


def window_values(file, bucket, quantisers=("recording", "window")):
    """LZC78, RSE4 and RSR in every window of one recording with the given quantisers; band signals, RMS envelopes
    and delays are those of the main analysis."""
    signal, fs = AM.load_excerpt(S.AUDIO_DIR / file)
    pooled = AM.load_quantisers(bucket)
    rows = []
    for fmin, fmax in S.BANDS:
        fs_b = AM.band_rate(fmax)
        rms = AM.local_rms(AM.band_signal(signal, fs, fmin, fmax, fs_b), fs_b, fmin, AM.rms_cycles(fmax, P["rms_k"]))
        delay = AM.symbol_delay(fmax, fs_b, P["lambda_factor"], P["lambda_exponent"])
        starts, length = AM.window_starts(len(rms), fs_b, P["window_s"])
        pos = np.linspace(0, len(rms) - 1, AM.QUANTISER_POINTS, dtype=np.int64)
        fixed = {"bucket": {a: pooled[(bucket, fmin, P["rms_k"], a)] for a in ALPHABETS},
                 "recording": fit_edges(rms[pos])}
        for w, s in enumerate(starts):
            seg = rms[s:s + length]
            for q in quantisers:
                edges = fit_edges(seg) if q == "window" else fixed[q]
                for m, v in AM.nonlinear_measures(seg, delay, edges, P, QUANTISED).items():
                    rows.append((q, m, fmin, w, v))
    return pd.DataFrame(rows, columns=["quantiser", "measure", "fmin_hz", "window", "value"])


def _cache_recording(file, bucket):
    window_values(file, bucket).to_csv(CACHE_DIR / f"{file.removesuffix('.wav')}.csv.gz", index=False)
    return file


def cached_window_values(rec, main_windows):
    """Window values of all recordings (long table), after checking on the first recording that the pooled quantiser
    reproduces the window values of the main analysis."""
    f, b = rec.file[0], rec.bucket[0]
    check = window_values(f, b, ("bucket",)).merge(
        main_windows[main_windows.file == f], on=["measure", "fmin_hz", "window"], suffixes=("", "_main"))
    deviation = np.nanmax(np.abs(check.value - check.value_main))
    if not deviation < 1e-9:
        raise RuntimeError(f"pooled quantiser does not reproduce the main analysis ({f}: {deviation:.1e})")
    print(f"check, {f}: {len(check)} window values reproduced (max deviation {deviation:.1e})", flush=True)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    pending = [(f, b) for f, b in zip(rec.file, rec.bucket)
               if not (CACHE_DIR / f"{f.removesuffix('.wav')}.csv.gz").exists()]
    print(f"window values, {len(pending)} of {len(rec)} recordings to do ({AM.N_WORKERS} workers)", flush=True)
    with ProcessPoolExecutor(AM.N_WORKERS) as pool:
        jobs = [pool.submit(_cache_recording, f, b) for f, b in pending]
        for i, job in enumerate(as_completed(jobs), 1):
            print(f"  {i}/{len(pending)} {job.result()} ({(time.time() - t0) / 60:.0f} min)", flush=True)
    parts = [pd.read_csv(CACHE_DIR / f"{f.removesuffix('.wav')}.csv.gz").assign(file=f) for f in rec.file]
    return pd.concat(parts, ignore_index=True).join(rec.set_index("file")[["bucket", "occasion"]], on="file")


def median_windows(main_windows):
    """Windows that held the median of each recording in the main analysis (the middle rank or the two middle ranks)."""
    w = main_windows[main_windows.measure.isin(QUANTISED) & np.isfinite(main_windows.value)]
    g = w.groupby(["file", "measure", "fmin_hz"]).value
    rank, n = g.rank(method="first") - 1, g.transform("size")
    return w.loc[(rank == (n - 1) // 2) | (rank == n // 2), ["file", "measure", "fmin_hz", "window"]]


def with_quantised(main_values, values):
    """Recording values of the main analysis with LZC78, RSE4 and RSR taken from `values`."""
    return pd.concat([main_values[~main_values.measure.isin(QUANTISED)], values[main_values.columns]],
                     ignore_index=True)


# =============================================================================
# Bayesian bootstrap and model comparison
# =============================================================================
def bootstrap_control():
    """Bayesian bootstrap of the control as in bootstrap.py: coefficient draws of L, the difference between the
    branches and the bands of the examples of Fig. S9."""
    bs = B.Bootstrap(OUT)
    draws = B.coefficient_draws(bs)
    np.save(OUT / "L_coefficient_draws.npy", draws)
    branches = B.branch_difference(bs, draws)[1]
    branches.to_csv(OUT / "branch_difference_summary.csv", index=False)
    bandwidth = M.choose_bandwidth(bs.Y, bs.valid, list(range(len(S.BUCKETS))))
    rng = np.random.default_rng(B.FIT_SEED)
    for measure, fmin in FIG_EXAMPLES:
        B.fit_bands(bs, measure, fmin, rng, with_np=True, np_bandwidth=bandwidth, out=OUT)
    return draws, branches


def roughness(Y, valid):
    """SD of the differences between consecutive occasions of the mean trajectory / SD of the trajectory."""
    y = M.mean_trajectory(Y, valid)
    y = y[:, ~np.isnan(y).any(0)]
    return np.diff(y, axis=1).std(1, ddof=1) / y.std(1, ddof=1)


def evaluate(values):
    """Held-out skill, fit to the mean trajectory and roughness per configuration; residuals of L (occasions x
    configurations)."""
    Y, cfg, valid = M.load_values(values)
    ho = heldout(Y, valid, ["time-only", "L"])
    fits = {"L": M.mean_fit(Y, valid, "L")}
    per = cfg.copy()
    per["skill_time_only"], per["skill_L"] = ho["time-only"]["skill"], ho["L"]["skill"]
    for part in ("heating", "cooling"):
        per[f"skill_L_{part}"] = M.skill(ho["L"]["folds"], part).mean(0)
    per["R2_L"] = fits["L"]["r2"]
    per["R2_NP"] = np_mean_fit(Y, valid, fits)[0]["r2"]
    per["roughness"] = roughness(Y, valid)
    return per, fits["L"]["observed"] - fits["L"]["fitted"]


def excludes_zero(draws):
    """95% interval of the Bayesian-bootstrap draws excludes zero (configurations x coefficients)."""
    lo, hi = np.percentile(draws.astype(float), [2.5, 97.5], axis=1)
    return (lo > 0) | (hi < 0)


def report(per, var, corr, draws, branches):
    c, m = per["recording"], per["bucket"]
    nl = lambda p: p[p.family == "nonlinear"]
    r = var[var.group == "nonlinear"].set_index("quantiser")
    both = lambda col, fmt=".2f": f"{r[col]['recording']:{fmt}} ({r[col]['bucket']:{fmt}})"
    phase = lambda p: f"{nl(p).skill_L_heating.median():.2f} / {nl(p).skill_L_cooling.median():.2f}"
    nonlinear = (c.family == "nonlinear").to_numpy()
    ex = {q: excludes_zero(d)[nonlinear] for q, d in draws.items()}
    rough = {q: p[p.measure.isin(QUANTISED)].roughness.to_numpy() for q, p in (("recording", c), ("bucket", m))}
    br = {q: b.set_index("group").loc["nonlinear"] for q, b in branches.items()}
    span = lambda s: f"{s.abs().min():.2f}-{s.abs().max():.2f}"
    L = ["LEVEL-INVARIANT CONTROL: quantiser per recording (main analysis, quantiser per bucket, in brackets)",
         f"nonlinear ({int(r.n['recording'])} configurations): median held-out skill time-only "
         f"{both('skill_time_only')}, L {both('skill_L')}; R2 of L {both('R2_L')}; L better than time-only in "
         f"{both('n_L_better_than_time', 'd')}",
         f"  skill of L on heating / cooling occasions {phase(c)} ({phase(m)})",
         f"  PE (unchanged): median held-out skill of L {c[c.measure == 'PE'].skill_L.median():.2f}",
         f"95% Bayesian-bootstrap interval excludes 0 (of {nonlinear.sum()} nonlinear configurations): " + "; ".join(
             f"{name} {ex['recording'][:, i].sum()} ({ex['bucket'][:, i].sum()})" for i, name in ((1, "b"), (2, "d"),
                                                                                                   (3, "e"))),
         f"difference between the branches at {br['recording'].T_mid:.1f} C: median |difference| "
         f"{br['recording'].median_abs_difference_mid:.2f} SD ({br['bucket'].median_abs_difference_mid:.2f}); interval "
         f"excludes 0 in {int(br['recording'].n_ci_excludes_zero_mid)} ({int(br['bucket'].n_ci_excludes_zero_mid)}); "
         f"branches cross in {int(br['recording'].n_branches_cross)} ({int(br['bucket'].n_branches_cross)})",
         f"roughness of the mean trajectory ({len(rough['recording'])} configurations of LZC78, RSE4, RSR): median "
         f"{np.median(rough['recording']):.2f} ({np.median(rough['bucket']):.2f}), increased in "
         f"{int((rough['recording'] > rough['bucket']).sum())}; PE {m[m.measure == 'PE'].roughness.median():.2f}",
         "|r| of the residuals of L with those of PE in the same band: "
         + "; ".join(f"{label} {span(s.r_recording)} ({span(s.r_bucket)})" for label, s in
                     (("1-8 kHz", corr[corr.fmin_hz.isin(MID_BANDS)]), (f"{TOP_BAND}-{2 * TOP_BAND} Hz",
                                                                       corr[corr.fmin_hz == TOP_BAND])))]
    return L


def run():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    rec = AM.recordings()
    main_windows = pd.read_csv(S.MEASURES_DIR / "measures_by_window.csv.gz")
    main_values = pd.read_csv(S.MEASURES_DIR / "measures_by_recording.csv.gz")

    # window and recording values of the control and the two checks
    cache = cached_window_values(rec, main_windows[main_windows.measure.isin(QUANTISED)])
    own = cache[cache.quantiser == "recording"]
    key = ["file", "measure", "fmin_hz", "window"]
    q = main_windows.measure.isin(QUANTISED)
    windows = pd.concat([main_windows[~q], main_windows[q].drop(columns="value").merge(own[key + ["value"]], on=key,
                                                                                       how="left")], ignore_index=True)
    windows.to_csv(OUT / "measures_by_window.csv.gz", index=False)
    values = {"bucket": main_values,
              "recording": with_quantised(main_values, AM.recording_medians(own, COLS)),
              "window": with_quantised(main_values, AM.recording_medians(cache[cache.quantiser == "window"], COLS)),
              "recording, median": with_quantised(main_values, AM.recording_medians(
                  own.merge(median_windows(main_windows), on=key), COLS))}     # median of 1-2 windows = their mean
    values["recording"].to_csv(OUT / "measures_by_recording.csv.gz", index=False)

    # Bayesian bootstrap of the control (Fig. S9, coefficients, difference between the branches)
    draws, branches = bootstrap_control()
    draws = {"recording": draws, "bucket": np.load(S.BOOTSTRAP_DIR / "L_coefficient_draws.npy")}
    branches = {"recording": branches, "bucket": pd.read_csv(S.BOOTSTRAP_DIR / "branch_difference_summary.csv")}

    # skill and fit per quantiser (Table S9)
    per, resid = {}, {}
    for name in QUANTISERS:
        per[name], resid[name] = evaluate(values[name])
    rows = []
    for name in QUANTISERS:
        p = per[name]
        for group, s in (("nonlinear", p[p.family == "nonlinear"]), ("quantised", p[p.measure.isin(QUANTISED)])):
            rows.append(dict(quantiser=name, group=group, n=len(s), skill_time_only=s.skill_time_only.median(),
                             skill_L=s.skill_L.median(), R2_L=s.R2_L.median(),
                             n_L_better_than_time=int((s.skill_L > s.skill_time_only).sum())))
    var = pd.DataFrame(rows)
    var.to_csv(OUT / "variants.csv", index=False)
    pd.concat([per[name].assign(quantiser=name) for name in ("bucket", "recording")]).to_csv(
        OUT / "per_config.csv", index=False)

    # residuals of L against those of PE in the same band (Table S10)
    cfg = per["bucket"]
    k = {(m, f): i for i, (m, f) in enumerate(zip(cfg.measure, cfg.fmin_hz))}
    corr = pd.DataFrame([dict(fmin_hz=f, measure=m, **{f"r_{name}": np.corrcoef(resid[name][:, k[(m, f)]],
                                                                               resid[name][:, k[("PE", f)]])[0, 1]
                                                       for name in ("bucket", "recording")})
                         for f in sorted(cfg.fmin_hz.unique()) for m in QUANTISED])
    corr.to_csv(OUT / "residual_correlation_with_PE.csv", index=False)

    L = report(per, var, corr, draws, branches) + [f"runtime {(time.time() - t0) / 60:.0f} min"]
    (OUT / "report.txt").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    run()
