"""Acoustic measures of the 600 s excerpts.

Steps
  crop       600 s excerpts ending 30 s before the end of each raw recording (only needed to rebuild data/audio)
  measures   quantiser boundaries (pass 1) and all measures in every window (pass 2): the four nonlinear and six
             conventional measures in eight octave bands for the primary setting and every sensitivity scenario,
             the conventional indices computed broadband, and the 50 Hz mains-hum score

Window values are cached per recording in results/measures/cache, so an interrupted run resumes.
"""
import json
import math
import os
import re
import time
import wave
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import soundfile as sf
from numba import njit, types
from numba.typed import Dict
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, hilbert, resample_poly, sosfiltfilt, stft

import settings as S

N_WORKERS = max(1, min((os.cpu_count() or 2) - 1, 6))   # each worker holds ~1.2 GB; lower this if you run out of memory
CACHE_DIR = S.MEASURES_DIR / "cache"
QUANTISER_FILE = S.MEASURES_DIR / "quantiser_edges.csv"

FILTER_ORDER = 6
SAMPLES_PER_FMAX_CYCLE = 10          # band signals are resampled to 10 x the upper band edge (at most 48 kHz)
QUANTISER_POINTS = 4000              # evenly spaced RMS values per recording that define the quantisers
QUANTISER_RMS_K = (2.5, 5.0, 10.0)   # RMS widths of the primary setting and the sensitivity scenarios
QUANTISER_ALPHABETS = (8, 10, 12, 50, 100, 200)
STFT_FRAME = 512
AEI_RESOLUTION_HZ = 10.0
AEI_BINS = 10
AEI_THRESHOLD_DB = -50.0
POWER_FLOOR = 1e-20
AMPLITUDE_FLOOR = 1e-6

# Broadband reference: the conventional indices with their standard parameters over 64-16384 Hz
BROADBAND = (64.0, 16384.0)
ACI_CLUSTER_S = 5.0
AEI_MAX_HZ = 10000.0

# Mains hum: 50 Hz peak against the neighbouring spectrum
HUM_PEAK_HZ = (49.0, 51.0)
HUM_FLANKS_HZ = ((30.0, 46.0), (54.0, 70.0))
HUM_SNR_DB = 25.0

END_TRIM_S = 30.0
RAW_NAME = re.compile(r"^([A-E])(\d{1,2})_(D|N|Baseline)\.wav$")


# =============================================================================
# Recordings
# =============================================================================
def recordings():
    """Table of the analysed recordings (data/recordings.csv)."""
    return pd.read_csv(S.RECORDINGS_FILE)


def load_excerpt(path):
    """600 s single-channel excerpt as float32."""
    with sf.SoundFile(path) as handle:
        fs = handle.samplerate
        if handle.channels != 1 or len(handle) != round(S.EXCERPT_S * fs):
            raise ValueError(f"{path}: expected a single-channel {S.EXCERPT_S:g} s recording")
        signal = handle.read(dtype="float32")
    if not np.all(np.isfinite(signal)):
        raise ValueError(f"{path}: non-finite samples")
    return np.ascontiguousarray(signal), fs


def crop_recordings():
    """Write the 600 s excerpts and data/recordings.csv from the raw recordings and their recorder XML files."""
    S.AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in sorted(S.RAW_AUDIO_DIR.glob("*.wav")):
        match = RAW_NAME.match(path.name)
        if not match:
            continue
        bucket, occasion, session = match.group(1), int(match.group(2)), match.group(3)
        out = S.AUDIO_DIR / f"{path.stem}_cropped.wav"
        with wave.open(str(path)) as src:
            fs, frames = src.getframerate(), src.getnframes()
            stop = frames - round(END_TRIM_S * fs)
            start = stop - round(S.EXCERPT_S * fs)
            if start < 0:
                raise ValueError(f"{path.name} is shorter than {S.EXCERPT_S + END_TRIM_S:g} s")
            if not out.exists():
                src.setpos(start)
                with wave.open(str(out), "wb") as dst:
                    dst.setparams(src.getparams())
                    dst.writeframes(src.readframes(stop - start))
        xml = path.with_suffix(".xml")
        stamp = re.search(r"<DateTime>([^<]+)</DateTime>", xml.read_text()) if xml.exists() else None
        rows.append(dict(file=out.name, source_file=path.name, bucket=bucket, occasion=occasion, session=session,
                         source_duration_s=frames / fs, crop_start_s=start / fs,
                         recorder_start=pd.to_datetime(stamp.group(1), dayfirst=True) if stamp else pd.NaT))
    table = pd.DataFrame(rows).sort_values(["occasion", "bucket"])
    table.to_csv(S.RECORDINGS_FILE, index=False)
    print(f"{len(table)} excerpts in {S.AUDIO_DIR}")


# =============================================================================
# Band signals and time scales
# =============================================================================
def band_rate(fmax):
    return int(min(S.SAMPLE_RATE_HZ, round(SAMPLES_PER_FMAX_CYCLE * fmax)))


def band_signal(signal, fs, fmin, fmax, target_fs):
    """Zero-phase Butterworth band-pass, then polyphase resampling to target_fs."""
    sos = butter(FILTER_ORDER, [fmin, fmax], btype="bandpass", fs=fs, output="sos")
    x = sosfiltfilt(sos, signal).astype(np.float32, copy=False)
    if target_fs != fs:
        g = math.gcd(int(fs), int(target_fs))
        x = resample_poly(x, target_fs // g, fs // g).astype(np.float32, copy=False)
    n = round(len(signal) * target_fs / fs)
    if len(x) > n:
        x = x[:n]
    elif len(x) < n:
        x = np.pad(x, (0, n - len(x)), mode="reflect")
    return np.ascontiguousarray(x, dtype=np.float32)


def scale_factor(fmax, factor=1.0, exponent=-0.5):
    """lambda = 3 (fmax / 128 Hz)^exponent, falling by sqrt(2) per octave in the primary setting."""
    return factor * 3.0 * (fmax / 128.0) ** exponent


def symbol_delay(fmax, fs, factor=1.0, exponent=-0.5):
    """Delay tau of the symbolic analysis in samples: lambda fs / fmax, rounded, at least one."""
    return max(1, int(round(scale_factor(fmax, factor, exponent) * fs / fmax)))


def rms_cycles(fmax, k):
    """RMS window in cycles of the lower band edge: k lambda (always with the primary lambda law)."""
    return round(k * scale_factor(fmax), 6)


def local_rms(x, fs, fmin, cycles):
    """Centred moving RMS with an odd window of `cycles` periods of fmin, reflected at the boundaries."""
    width = max(1, int(round(cycles * fs / fmin)))
    width += 1 - width % 2
    power = uniform_filter1d(np.square(x, dtype=np.float32), size=width, mode="reflect")
    np.maximum(power, 0.0, out=power)
    return np.sqrt(power, out=power)


def window_starts(n, fs, window_s):
    length = int(round(window_s * fs))
    return np.arange(0, n - length + 1, length), length


# =============================================================================
# Nonlinear measures
# =============================================================================
@njit(cache=True)
def lz78_phrase_count(symbols, alphabet):
    """Number of phrases in incremental LZ78 parsing; a non-empty final prefix counts as the last phrase."""
    dictionary = Dict.empty(key_type=types.int64, value_type=types.int64)
    prefix, next_index, count = 0, 1, 0
    for symbol in symbols:
        key = prefix * alphabet + symbol
        if key in dictionary:
            prefix = dictionary[key]
        else:
            dictionary[key] = next_index
            next_index += 1
            count += 1
            prefix = 0
    if prefix != 0:
        count += 1
    return count


@njit(cache=True)
def ordinal_patterns(values, order, delay):
    """Ordinal pattern (Lehmer code) of every delay vector; ties are ranked by position."""
    n = len(values) - (order - 1) * delay
    codes = np.empty(n, dtype=np.int64)
    rank = np.empty(order, dtype=np.int64)
    for row in range(n):
        for axis in range(order):
            rank[axis] = axis
        for pos in range(1, order):                 # stable insertion sort of the axes by value
            axis = rank[pos]
            value = values[row + axis * delay]
            left = pos - 1
            while left >= 0 and values[row + rank[left] * delay] > value:
                rank[left + 1] = rank[left]
                left -= 1
            rank[left + 1] = axis
        code = 0
        for pos in range(order - 1):
            smaller = 0
            for right in range(pos + 1, order):
                if rank[right] < rank[pos]:
                    smaller += 1
            factorial = 1
            for f in range(2, order - pos):
                factorial *= f
            code += smaller * factorial
        codes[row] = code
    return codes


def block_entropy(symbols, alphabet, k):
    """Base-2 entropy of overlapping words of k symbols."""
    n = len(symbols) - (k - 1)
    if n < 2:
        return np.nan
    codes = np.zeros(n, dtype=np.int64)
    for i in range(k):
        codes = codes * alphabet + symbols[i:i + n]
    p = np.unique(codes, return_counts=True)[1] / float(n)
    return -float(np.sum(p * np.log2(p)))


def permutation_entropy(values, order, delay):
    counts = np.bincount(ordinal_patterns(values, order, delay), minlength=math.factorial(order))
    p = counts[counts > 0].astype(np.float64)
    p /= p.sum()
    return -float(np.sum(p * np.log(p))) / math.log(math.factorial(order))


def nonlinear_measures(rms, delay, edges, p, wanted):
    """PE, LZC78, RSE_k and RSR of one RMS window. `edges` maps alphabet size to quantiser boundaries."""
    out = {}
    if "RSE4" in wanted:
        a, k = p["rse_alphabet"], p["rse_block"]
        s = np.searchsorted(edges[a], rms, side="right").astype(np.int64)[::delay]
        h1, hk = block_entropy(s, a, 1), block_entropy(s, a, k)
        out["RSE4"] = (k * h1 - hk) / ((k - 1.0) * h1) if np.isfinite(h1) and h1 != 0 else np.nan
    if "LZC78" in wanted or "RSR" in wanted:
        a = p["lzc_alphabet"]
        s = np.searchsorted(edges[a], rms, side="right").astype(np.int64)[::delay]
        c = lz78_phrase_count(s, a) if len(s) >= 2 else np.nan
        if "LZC78" in wanted:
            out["LZC78"] = float(c)
        if "RSR" in wanted:
            h_lz = c * np.log2(c) / len(s) if len(s) >= 2 and c >= 1 else np.nan
            h1 = block_entropy(s, a, 1)
            out["RSR"] = 1.0 - h_lz / h1 if np.isfinite(h_lz) and np.isfinite(h1) and h1 > 0 else np.nan
    if "PE" in wanted:
        out["PE"] = permutation_entropy(rms, p["pe_order"], delay)
    return out


def quantiser_edges(values, alphabet):
    """Equal-width bins over the pooled 1st-99th percentile range; the outer bins collect the tails."""
    q01, q99 = np.quantile(values, [0.01, 0.99])
    return np.linspace(q01, q99, alphabet + 1)[1:-1]


# =============================================================================
# Conventional measures
# =============================================================================
def spectrogram_magnitude(x, fs, fmin, fmax, window="hann"):
    """Magnitude of a 512-sample STFT without overlap, padding or boundary extension, restricted to the band."""
    n = min(len(x), STFT_FRAME)
    f, _, z = stft(x, fs=fs, window=window, nperseg=n, noverlap=0, nfft=n, boundary=None, padded=False)
    band = (f >= fmin) & (f <= fmax)
    return np.abs(z[band]).astype(np.float64, copy=False)


def normalised_entropy(weights):
    w = np.asarray(weights, dtype=np.float64)
    w = w[np.isfinite(w) & (w >= 0)]
    if len(w) < 2:
        return 0.0 if len(w) == 1 else np.nan
    total = float(np.sum(w))
    if total <= 0:
        return np.nan
    p = w[w > 0] / total
    return float(-np.sum(p * np.log(p)) / math.log(len(w)))


def spectral_flatness(mean_spectrum):
    peak = float(np.max(mean_spectrum))
    rel = mean_spectrum / peak if peak > 0 else np.zeros_like(mean_spectrum)
    rel = np.maximum(rel, AMPLITUDE_FLOOR)
    return float(np.exp(np.mean(np.log(rel))) / np.mean(rel))


def log_energy(x):
    return float(10.0 * np.log10(max(float(np.mean(np.square(x, dtype=np.float64))), POWER_FLOOR)))


def temporal_entropy(x):
    return normalised_entropy(np.abs(hilbert(np.asarray(x, dtype=np.float64))))


def aci(magnitude):
    """Acoustic Complexity Index: summed relative absolute change between frames, per frequency bin."""
    num = np.sum(np.abs(np.diff(magnitude, axis=1)), axis=1)
    den = np.sum(magnitude, axis=1)
    ok = den > POWER_FLOOR
    return float(np.sum(num[ok] / den[ok])) if ok.any() else 0.0


def gini(values):
    """Uncorrected Gini coefficient."""
    v = np.asarray(values, dtype=np.float64)
    n, total = len(v), float(np.sum(v))
    if n < 2 or total <= 0:
        return np.nan
    ranks = np.arange(1, n + 1, dtype=np.float64)
    return float(2.0 * np.sum(ranks * np.sort(v)) / (n * total) - (n + 1.0) / n)


def aei(x, fs, fmin, fmax):
    """Acoustic Evenness Index: Gini coefficient of the share of time-frequency cells above -50 dB re maximum
    in ten equal-width frequency bins between fmin and fmax (both bin edges inclusive)."""
    n = max(16, int(round(fs / AEI_RESOLUTION_HZ)))
    n = min(n + n % 2, len(x))
    f, _, z = stft(x, fs=fs, window="hann", nperseg=n, noverlap=0, nfft=n, boundary=None, padded=False)
    mag = np.abs(z).astype(np.float64, copy=False)
    ref = float(np.max(mag))
    if not np.isfinite(ref) or ref <= 0:
        return np.nan
    level = 20.0 * np.log10(np.maximum(mag / ref, math.sqrt(POWER_FLOOR)))
    edges = np.linspace(fmin, fmax, AEI_BINS + 1)
    shares = [float(np.mean(level[(f >= lo) & (f <= hi)] > AEI_THRESHOLD_DB)) if np.any((f >= lo) & (f <= hi))
              else 0.0 for lo, hi in zip(edges[:-1], edges[1:])]
    return gini(shares)


def conventional_measures(x, fs, fmin, fmax):
    """The six conventional indices of one band-limited window."""
    mag = spectrogram_magnitude(x, fs, fmin, fmax)
    mean_spectrum = mag.mean(axis=1)
    return {"LogBandEnergy": log_energy(x), "ACI": aci(mag), "TemporalEntropy": temporal_entropy(x),
            "SpectralEntropy": normalised_entropy(mean_spectrum), "SpectralFlatness": spectral_flatness(mean_spectrum),
            "AEI": aei(x, fs, fmin, fmax)}


def broadband_measures(x, fs):
    """Conventional indices over 64-16384 Hz with their standard parameters (soundecology / seewave defaults)."""
    lo, hi = BROADBAND
    mag = spectrogram_magnitude(x, fs, lo, hi)
    mean_spectrum = mag.mean(axis=1)
    hamming = spectrogram_magnitude(x, fs, lo, hi, window="hamming")
    per_cluster = int(ACI_CLUSTER_S * fs / STFT_FRAME)
    aci_total = sum(aci(hamming[:, c * per_cluster:(c + 1) * per_cluster])
                    for c in range(hamming.shape[1] // per_cluster))
    return {"LogBandEnergy": log_energy(x), "ACI": float(aci_total), "TemporalEntropy": temporal_entropy(x),
            "SpectralEntropy": normalised_entropy(mean_spectrum), "SpectralFlatness": spectral_flatness(mean_spectrum),
            "AEI": aei(x, fs, 0.0, AEI_MAX_HZ)}


def hum_snr(signal, fs):
    """Signal-to-noise ratio (dB) of the 50 Hz peak in every 15 s window (Hann-windowed power spectrum)."""
    length = int(S.WINDOW_S * fs)
    freqs = np.fft.rfftfreq(length, 1 / fs)
    idx = lambda lo, hi: (int(np.searchsorted(freqs, lo)), int(np.searchsorted(freqs, hi)))
    peak, flanks = idx(*HUM_PEAK_HZ), [idx(*f) for f in HUM_FLANKS_HZ]
    hann = np.hanning(length)
    x = signal.astype(np.float64)
    snr = []
    for start in range(0, len(x) - length + 1, length):
        w = x[start:start + length]
        power = np.abs(np.fft.rfft((w - w.mean()) * hann)) ** 2
        floor = np.median(np.concatenate([power[a:b] for a, b in flanks]))
        snr.append(10 * np.log10(power[peak[0]:peak[1]].max() / floor + 1e-30))
    return np.array(snr)


# =============================================================================
# Pass 1: quantisers
# =============================================================================
def _support_points(file):
    """Evenly spaced RMS values of one recording for every band and RMS width."""
    signal, fs = load_excerpt(S.AUDIO_DIR / file)
    out = {}
    for fmin, fmax in S.BANDS:
        fs_b = band_rate(fmax)
        x = band_signal(signal, fs, fmin, fmax, fs_b)
        for k in QUANTISER_RMS_K:
            r = local_rms(x, fs_b, fmin, rms_cycles(fmax, k))
            pos = np.linspace(0, len(r) - 1, QUANTISER_POINTS, dtype=np.int64)
            out[(fmin, k)] = r[pos] if len(r) > QUANTISER_POINTS else r.copy()
    return out


def fit_quantisers(rec):
    """Quantiser boundaries per bucket, band, RMS width and alphabet, pooled over all occasions of a bucket."""
    pools = {}
    with ProcessPoolExecutor(N_WORKERS) as pool:
        jobs = {pool.submit(_support_points, f): b for f, b in zip(rec.file, rec.bucket)}
        for i, job in enumerate(as_completed(jobs), 1):
            for (fmin, k), values in job.result().items():
                pools.setdefault((jobs[job], fmin, k), []).append(values)
            print(f"  quantiser support {i}/{len(jobs)}", flush=True)
    rows = []
    for (bucket, fmin, k), parts in sorted(pools.items()):
        values = np.concatenate(parts).astype(np.float64)
        for a in QUANTISER_ALPHABETS:
            rows.append(dict(bucket=bucket, fmin_hz=fmin, rms_k=k, alphabet=a,
                             edges=json.dumps(quantiser_edges(values, a).tolist())))
    table = pd.DataFrame(rows)
    table.to_csv(QUANTISER_FILE, index=False)
    return table


def load_quantisers(bucket=None):
    """{(bucket, fmin, rms_k, alphabet): edges}"""
    q = pd.read_csv(QUANTISER_FILE)
    if bucket is not None:
        q = q[q.bucket == bucket]
    return {(r.bucket, r.fmin_hz, r.rms_k, r.alphabet): np.array(json.loads(r.edges)) for r in q.itertuples()}


# =============================================================================
# Pass 2: window values
# =============================================================================
def _scenario_settings():
    """(name, parameters, measures) of the primary setting and every sensitivity scenario."""
    out = [("primary", S.PRIMARY, S.MEASURES)]
    for name, (change, measures) in S.SCENARIOS.items():
        out.append((name, {**S.PRIMARY, **change}, measures))
    return out


def band_window_values(x, fs, fmin, fmax, quantisers, bucket, p, measures, rms_cache):
    """All requested measures in every window of one band signal for one parameter setting."""
    starts, length = window_starts(len(x), fs, p["window_s"])
    rows = []
    conventional = [m for m in measures if m in S.CONVENTIONAL]
    nonlinear = [m for m in measures if m in S.NONLINEAR]
    if conventional:
        for w, s in enumerate(starts):
            for m, v in conventional_measures(x[s:s + length], fs, fmin, fmax).items():
                rows.append((m, fmin, w, v))
    if nonlinear:
        cycles = rms_cycles(fmax, p["rms_k"])
        if cycles not in rms_cache:
            rms_cache[cycles] = local_rms(x, fs, fmin, cycles)
        rms = rms_cache[cycles]
        delay = symbol_delay(fmax, fs, p["lambda_factor"], p["lambda_exponent"])
        edges = {a: quantisers[(bucket, fmin, p["rms_k"], a)] for a in (p["lzc_alphabet"], p["rse_alphabet"])}
        for w, s in enumerate(starts):
            for m, v in nonlinear_measures(rms[s:s + length], delay, edges, p, nonlinear).items():
                rows.append((m, fmin, w, v))
    return rows


def _process_recording(file, bucket):
    """Window values of one recording: all scenarios, broadband indices and mains hum."""
    quantisers = load_quantisers(bucket)
    signal, fs = load_excerpt(S.AUDIO_DIR / file)
    rows = []
    for fmin, fmax in S.BANDS:
        fs_b = band_rate(fmax)
        x = band_signal(signal, fs, fmin, fmax, fs_b)
        rms_cache = {}
        for name, p, measures in _scenario_settings():
            rows += [(name,) + r for r in band_window_values(x, fs_b, fmin, fmax, quantisers, bucket, p, measures,
                                                              rms_cache)]
    x = band_signal(signal, fs, *BROADBAND, fs)
    starts, length = window_starts(len(x), fs, S.WINDOW_S)
    for w, s in enumerate(starts):
        rows += [("broadband", m, BROADBAND[0], w, v) for m, v in broadband_measures(x[s:s + length], fs).items()]
    rows += [("hum_snr_db", "HumSNR", 50.0, w, v) for w, v in enumerate(hum_snr(signal, fs))]
    table = pd.DataFrame(rows, columns=["scenario", "measure", "fmin_hz", "window", "value"])
    table.to_csv(CACHE_DIR / f"{file.removesuffix('.wav')}.csv.gz", index=False)
    return file


def recording_medians(windows, keys):
    """Median over the finite window values of every recording."""
    rows = []
    for key, g in windows.groupby(keys, sort=False):
        v = g["value"].to_numpy(dtype=float)
        v = v[np.isfinite(v)]
        if len(v):
            rows.append((*key, float(np.median(v))))
    return pd.DataFrame(rows, columns=keys + ["value"])


def compute_measures():
    S.MEASURES_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(exist_ok=True)
    rec = recordings()
    t0 = time.time()
    if not QUANTISER_FILE.exists():
        print("pass 1: quantiser boundaries", flush=True)
        fit_quantisers(rec)
    pending = [(f, b) for f, b in zip(rec.file, rec.bucket)
            if not (CACHE_DIR / f"{f.removesuffix('.wav')}.csv.gz").exists()]
    print(f"pass 2: window values, {len(pending)} of {len(rec)} recordings to do ({N_WORKERS} workers)", flush=True)
    with ProcessPoolExecutor(N_WORKERS) as pool:
        jobs = [pool.submit(_process_recording, f, b) for f, b in pending]
        for i, job in enumerate(as_completed(jobs), 1):
            print(f"  {i}/{len(pending)} {job.result()} ({(time.time() - t0) / 60:.0f} min)", flush=True)

    meta = rec.set_index("file")[["bucket", "occasion"]]
    parts = []
    for f in rec.file:
        w = pd.read_csv(CACHE_DIR / f"{f.removesuffix('.wav')}.csv.gz")
        w.insert(0, "file", f)
        parts.append(w)
    windows = pd.concat(parts, ignore_index=True)
    windows = windows.join(meta, on="file")
    cols = ["file", "bucket", "occasion", "measure", "fmin_hz"]

    primary = windows[windows.scenario == "primary"]
    primary[cols + ["window", "value"]].to_csv(S.MEASURES_DIR / "measures_by_window.csv.gz", index=False)
    recording_medians(primary, cols).to_csv(S.MEASURES_DIR / "measures_by_recording.csv.gz", index=False)
    scen = windows[windows.scenario.isin(S.SCENARIOS)]
    recording_medians(scen, ["scenario"] + cols).to_csv(S.MEASURES_DIR / "sensitivity_by_recording.csv.gz",
                                                        index=False)
    broad = windows[windows.scenario == "broadband"]
    recording_medians(broad, cols).to_csv(S.MEASURES_DIR / "broadband_by_recording.csv", index=False)

    hum = windows[windows.scenario == "hum_snr_db"]
    hum = (hum.groupby(["file", "bucket", "occasion"], sort=False)["value"]
           .agg(share_windows_above=lambda v: float(np.mean(v > HUM_SNR_DB)), median_snr_db="median").reset_index())
    hum["hum"] = (hum.share_windows_above > 0.5).astype(int)
    hum.to_csv(S.MEASURES_DIR / "mains_hum_by_recording.csv", index=False)

    discarded = (rec.source_duration_s - S.EXCERPT_S - END_TRIM_S) / 60
    minutes = rec.source_duration_s / 60
    lines = [f"{len(rec)} recordings of {minutes.min():.1f}-{minutes.max():.1f} min;"
             f" discarded start {discarded.min():.1f}-{discarded.max():.1f} min (median {discarded.median():.1f} min)",
             f"mains hum present in {hum.hum.mean():.0%} of the recordings",
             f"runtime {(time.time() - t0) / 60:.0f} min"]
    (S.MEASURES_DIR / "report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    compute_measures()
