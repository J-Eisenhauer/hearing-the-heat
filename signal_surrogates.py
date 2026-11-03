"""Signal-level IAAFT surrogates of the recordings and the linear control.

Experimental recordings: one 15 s window per recording (fixed seed per file), the band-filtered signal of that
window is replaced by an IAAFT surrogate; the conventional measures are computed from the surrogate signal and
the nonlinear measures from the RMS series of the surrogate signal, identically to the observed window.
Effects are expressed in standard deviations of the main analysis with paired circular moving-block bootstrap
intervals within buckets.

Linear control: five synthetic signals, each a sum of eight band-pass filtered Gaussian white-noise processes,
tested with surrogates of the band signal (as above) and, for comparison, with surrogates of the RMS series.
"""
import hashlib
import math
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt
from scipy.stats import binomtest

import audio_measures as A
import settings as S

WINDOW_SEED = 20260829
TARGET_ERROR = 0.01                   # relative spectral error at which the iteration stops
ACCEPT_ERROR = 0.025                  # surrogates above this error are excluded from the comparison
N_SEEDS = 5                           # seeds 1-4 with quality gates, seed 5 without
GATED_MAX_ITER = 250
GATES = {50: 0.10, 100: 0.05, 150: 0.025, 200: 0.0125}   # iteration: largest acceptable best error so far
FINAL_MAX_ITER = 500
PLATEAU_ITER = 5                      # stop after this many iterations with relative improvement below 1e-4
PLATEAU_IMPROVEMENT = 1e-4

BOOTSTRAP_DRAWS = 5000
BLOCK_LENGTH = 4
BOOTSTRAP_SEED = 20260907

LINEAR_SIGNALS = 5
LINEAR_SEED = 20260914
LINEAR_DURATION_S = 15.0


def seed_from(*parts):
    return int(hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).hexdigest()[:16], 16) % 2 ** 32


# =============================================================================
# IAAFT
# =============================================================================
def iaaft(values, rng, max_iter, gates=None):
    """IAAFT surrogate (Schreiber and Schmitz 1996): alternately impose the Fourier magnitudes and the amplitude
    distribution of `values`. Returns the iterate with the lowest relative spectral error and that error."""
    x = np.asarray(values, dtype=np.float64)
    target = np.abs(np.fft.rfft(x))
    norm = float(np.linalg.norm(target))
    ranked = np.sort(x)
    current = rng.permutation(x)
    spectrum = np.fft.rfft(current)
    best, best_error, plateau = current.copy(), np.inf, 0
    for it in range(1, max_iter + 1):
        magnitude = np.abs(spectrum)
        phase = np.divide(spectrum, magnitude, out=np.ones_like(spectrum), where=magnitude > np.finfo(float).eps)
        candidate = np.fft.irfft(target * phase, n=len(x))
        current = np.empty_like(candidate)
        current[np.argsort(candidate, kind="mergesort")] = ranked
        spectrum = np.fft.rfft(current)
        error = float(np.linalg.norm(np.abs(spectrum) - target) / norm)
        previous = best_error
        if error < best_error:
            best, best_error = current.copy(), error
        if best_error <= TARGET_ERROR or (gates and it in gates and best_error > gates[it]):
            break
        improvement = (previous - best_error) / max(previous, np.finfo(float).eps) if np.isfinite(previous) else np.inf
        plateau = plateau + 1 if np.isfinite(improvement) and improvement < PLATEAU_IMPROVEMENT else 0
        if plateau >= PLATEAU_ITER:
            break
    return best.astype(np.float32), best_error


def adaptive_iaaft(values, *seed_parts):
    """Up to five seeds; the surrogate with the lowest spectral error is kept. Returns (surrogate, error, accepted)."""
    best, best_error = None, np.inf
    for attempt in range(1, N_SEEDS + 1):
        rng = np.random.default_rng(seed_from(*seed_parts, "adaptive_seed", attempt))
        if attempt < N_SEEDS:
            surrogate, error = iaaft(values, rng, GATED_MAX_ITER, GATES)
        else:
            surrogate, error = iaaft(values, rng, FINAL_MAX_ITER)
        if error < best_error:
            best, best_error = surrogate, error
        if error <= TARGET_ERROR:
            break
    return best, best_error, best_error <= ACCEPT_ERROR


# =============================================================================
# Experimental recordings
# =============================================================================
def select_window(file):
    """Start (s) of the 15 s window tested for one recording, drawn uniformly with a filename-based seed."""
    starts = np.arange(0.0, S.EXCERPT_S - S.WINDOW_S + 1e-9, S.WINDOW_S)
    return float(starts[np.random.default_rng(seed_from("iaaft_window_selection", WINDOW_SEED, file))
                        .choice(len(starts))])


def nonlinear_of_signal(x, fs, fmin, fmax, edges):
    """Nonlinear measures of a band-signal window via its RMS series (primary setting)."""
    rms = A.local_rms(x, fs, fmin, A.rms_cycles(fmax, S.PRIMARY["rms_k"]))
    return A.nonlinear_measures(rms, A.symbol_delay(fmax, fs), edges, S.PRIMARY, S.NONLINEAR)


def _test_recording(file, bucket, occasion):
    quantisers = A.load_quantisers(bucket)
    signal, fs = A.load_excerpt(S.AUDIO_DIR / file)
    start_s = select_window(file)
    rows = []
    for fmin, fmax in S.BANDS:
        fs_b = A.band_rate(fmax)
        x = A.band_signal(signal, fs, fmin, fmax, fs_b)
        start = int(round(start_s * fs_b))
        window = np.ascontiguousarray(x[start:start + int(round(S.WINDOW_S * fs_b))])
        surrogate, error, accepted = adaptive_iaaft(window, "iaaft", file, start_s, fmin, fmax, "classical",
                                                    np.nan, 0)
        edges = {a: quantisers[(bucket, fmin, S.PRIMARY["rms_k"], a)]
                 for a in (S.PRIMARY["lzc_alphabet"], S.PRIMARY["rse_alphabet"])}
        observed = {**A.conventional_measures(window, fs_b, fmin, fmax),
                    **nonlinear_of_signal(window, fs_b, fmin, fmax, edges)}
        surr = {**A.conventional_measures(surrogate, fs_b, fmin, fmax),
                **nonlinear_of_signal(surrogate, fs_b, fmin, fmax, edges)}
        for m in observed:
            rows.append(dict(file=file, bucket=bucket, occasion=occasion, window_start_s=start_s, measure=m,
                             fmin_hz=fmin, observed=observed[m], surrogate=surr[m], spectral_error=error,
                             accepted=accepted))
    return pd.DataFrame(rows)


def block_positions(n, rng):
    """Indices of one circular moving-block bootstrap sample of length n."""
    length = min(BLOCK_LENGTH, n)
    starts = rng.integers(0, n, size=math.ceil(n / length))
    return ((starts[:, None] + np.arange(length)[None, :]) % n).ravel()[:n]


def bootstrap_effects(comparisons, sd):
    """Effect per measure and band: mean over buckets of the bucket median of (observed - surrogate) / SD,
    with a 95% paired block-bootstrap interval (resampling recordings within buckets)."""
    work = comparisons.join(sd, on=["bucket", "measure", "fmin_hz"])
    work["d"] = (work.observed - work.surrogate) / work.sd
    work["valid"] = work.accepted & np.isfinite(work.d)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rows = []
    for (m, fmin), g in work.groupby(["measure", "fmin_hz"], sort=False):
        g = g.sort_values(["bucket", "occasion"], kind="mergesort")
        d, valid = g.d.to_numpy(), g.valid.to_numpy()
        pos = [np.flatnonzero(g.bucket.to_numpy() == b) for b in sorted(g.bucket.unique())]
        effect = np.mean([np.median(d[p[valid[p]]]) for p in pos])
        draws = np.empty(BOOTSTRAP_DRAWS)
        for i in range(BOOTSTRAP_DRAWS):
            medians = []
            for p in pos:
                chosen = p[block_positions(len(p), rng)]
                chosen = chosen[valid[chosen]]
                medians.append(np.median(d[chosen]) if len(chosen) else np.nan)
            draws[i] = np.mean(medians)
        lo, hi = np.quantile(draws[np.isfinite(draws)], [0.025, 0.975])
        rows.append(dict(measure=m, fmin_hz=fmin, family=S.FAMILY[m], effect_sd=effect, ci_low=lo, ci_high=hi,
                         ci_excludes_zero=bool(lo > 0 or hi < 0), n_valid=int(valid.sum())))
    return pd.DataFrame(rows)


# =============================================================================
# Linear control
# =============================================================================
def linear_signals():
    """Sums of eight independent Gaussian white-noise processes, each band-pass filtered to one octave."""
    n = int(round(S.SAMPLE_RATE_HZ * LINEAR_DURATION_S))
    out = []
    for i in range(1, LINEAR_SIGNALS + 1):
        rng = np.random.default_rng(LINEAR_SEED + i)
        x = np.zeros(n)
        for fmin, fmax in S.BANDS:
            sos = butter(4, [fmin, fmax], btype="bandpass", fs=S.SAMPLE_RATE_HZ, output="sos")
            c = sosfiltfilt(sos, rng.standard_normal(n))
            x += c / np.std(c)
        out.append(np.ascontiguousarray(x / np.std(x), dtype=np.float32))
    return out


def linear_control():
    """Observed minus surrogate values of the nonlinear measures for the synthetic linear signals."""
    fs = S.SAMPLE_RATE_HZ
    prepared, pools = [], {}
    for i, sig in enumerate(linear_signals(), 1):
        for fmin, fmax in S.BANDS:
            fs_b = A.band_rate(fmax)
            x = A.band_signal(sig, fs, fmin, fmax, fs_b)
            rms = A.local_rms(x, fs_b, fmin, A.rms_cycles(fmax, S.PRIMARY["rms_k"]))
            prepared.append((f"linear_{i:02d}", fmin, fmax, fs_b, x, rms))
            pools.setdefault(fmin, []).append(rms)
    alphabets = (S.PRIMARY["rse_alphabet"], S.PRIMARY["lzc_alphabet"])
    edges = {fmin: {a: A.quantiser_edges(np.concatenate(v).astype(np.float64), a) for a in alphabets}
             for fmin, v in pools.items()}
    rows = []
    for signal_id, fmin, fmax, fs_b, x, rms in prepared:
        delay = A.symbol_delay(fmax, fs_b)
        cycles = float(A.rms_cycles(fmax, S.PRIMARY["rms_k"]))
        observed = A.nonlinear_measures(rms, delay, edges[fmin], S.PRIMARY, S.NONLINEAR)
        # surrogate of the RMS series
        s_rms, err, ok = adaptive_iaaft(rms, "linear_iaaft_test", signal_id, fmin, fmax, "rms", cycles, 0)
        surr = A.nonlinear_measures(s_rms, delay, edges[fmin], S.PRIMARY, S.NONLINEAR)
        rows += [dict(route="rms_series", signal=signal_id, fmin_hz=fmin, measure=m, observed=observed[m],
                      surrogate=surr[m], accepted=ok) for m in observed]
        # surrogate of the band signal, then the RMS series
        s_sig, err, ok = adaptive_iaaft(x, "linear_iaaft_test", signal_id, fmin, fmax, "raw_signal_to_rms", cycles, 0)
        s_rms = A.local_rms(np.asarray(s_sig, dtype=np.float32), fs_b, fmin, cycles)
        surr = A.nonlinear_measures(s_rms, delay, edges[fmin], S.PRIMARY, S.NONLINEAR)
        rows += [dict(route="band_signal", signal=signal_id, fmin_hz=fmin, measure=m, observed=observed[m],
                      surrogate=surr[m], accepted=ok) for m in observed]
    return pd.DataFrame(rows)


def sign_summary(table, keys):
    """Two-sided sign test of observed minus surrogate over the accepted comparisons."""
    rows = []
    for key, g in table.groupby(keys, sort=False):
        d = (g.observed - g.surrogate).to_numpy()
        valid = g.accepted.to_numpy(bool) & np.isfinite(d)
        greater, less = int((d[valid] > 0).sum()), int((d[valid] < 0).sum())
        p = binomtest(greater, greater + less, 0.5).pvalue if greater + less else np.nan
        rows.append(dict(zip(keys, key if isinstance(key, tuple) else (key,)), n_valid=int(valid.sum()),
                         n_greater=greater, n_less=less, median_difference=float(np.median(d[valid])),
                         sign_test_p=float(p)))
    return pd.DataFrame(rows)


# =============================================================================
# Main
# =============================================================================
def run():
    S.SURROGATES_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    rec = A.recordings()
    parts = []
    with ProcessPoolExecutor(A.N_WORKERS) as pool:
        jobs = [pool.submit(_test_recording, f, b, o) for f, b, o in zip(rec.file, rec.bucket, rec.occasion)]
        for i, job in enumerate(as_completed(jobs), 1):
            parts.append(job.result())
            print(f"  signal surrogates {i}/{len(jobs)} ({(time.time() - t0) / 60:.0f} min)", flush=True)
    comp = pd.concat(parts).sort_values(["occasion", "bucket"], kind="mergesort").reset_index(drop=True)
    comp.to_csv(S.SURROGATES_DIR / "signal_surrogate_comparisons.csv.gz", index=False)

    # standard deviation of the recording values used for the main standardisation
    values = pd.read_csv(S.MEASURES_DIR / "measures_by_recording.csv.gz")
    sd = values.groupby(["bucket", "measure", "fmin_hz"])["value"].std(ddof=1).rename("sd")
    effects = bootstrap_effects(comp, sd)
    effects.to_csv(S.SURROGATES_DIR / "signal_surrogate_effects.csv", index=False)

    linear = linear_control()
    linear.to_csv(S.SURROGATES_DIR / "linear_control_comparisons.csv", index=False)
    lin = sign_summary(linear, ["route", "measure"])
    lin.to_csv(S.SURROGATES_DIR / "linear_control_summary.csv", index=False)

    comp["valid"] = comp.accepted & np.isfinite(comp.observed - comp.surrogate)
    lines = ["SIGNAL-LEVEL IAAFT SURROGATES"]
    for fam, group in (("nonlinear", S.NONLINEAR), ("conventional", S.CONVENTIONAL)):
        e = effects[effects.family == fam]
        c = comp[comp.measure.isin(group)]
        excluded = (~c.groupby(["file", "fmin_hz"]).valid.all())
        by_band = excluded.groupby(level="fmin_hz").mean()
        lines.append(f"{fam}: interval excludes zero in {int(e.ci_excludes_zero.sum())} of {len(e)} configurations; "
                     f"median |effect| {e.effect_sd.abs().median():.2f} SD; excluded recording-band series "
                     f"{int(excluded.sum())} of {len(excluded)} (by band: "
                     + ", ".join(f"{int(b)} Hz {v:.0%}" for b, v in by_band.items()) + ")")
    med = comp[comp.valid].assign(d=lambda t: t.observed - t.surrogate).groupby("measure").d.median()
    lines.append("median raw difference, experimental recordings: "
                 + ", ".join(f"{m} {med[m]:.4g}" for m in S.NONLINEAR))
    for route, g in lin.groupby("route", sort=False):
        lines.append(f"linear control, {route} surrogates: " + "; ".join(
            f"{r.measure} {r.n_valid} accepted, {max(r.n_greater, r.n_less)} with the same sign, "
            f"p = {r.sign_test_p:.2f}, median difference {r.median_difference:.2g}" for r in g.itertuples()))
    lines.append(f"runtime {(time.time() - t0) / 60:.0f} min")
    (S.SURROGATES_DIR / "report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    run()
