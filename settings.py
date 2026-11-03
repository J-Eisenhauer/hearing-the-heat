"""Paths, experimental design and analysis settings shared by all modules."""
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
AUDIO_DIR = DATA_DIR / "audio"                    # the 150 analysed 600 s excerpts (*_cropped.wav)
RAW_AUDIO_DIR = DATA_DIR / "raw_audio"            # full recordings with recorder XML files (crop step only)
PROTOCOL_FILE = DATA_DIR / "measurement_protocol.csv"
RECORDINGS_FILE = DATA_DIR / "recordings.csv"     # one row per analysed recording (crop provenance, start time)

RESULTS_DIR = ROOT / "results"
MEASURES_DIR = RESULTS_DIR / "measures"
SURROGATES_DIR = RESULTS_DIR / "signal_surrogates"
MODELS_DIR = RESULTS_DIR / "models"
BOOTSTRAP_DIR = RESULTS_DIR / "bootstrap"
MASS_DIR = RESULTS_DIR / "mass_balance"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"
LOUDNESS_DIR = RESULTS_DIR / "loudness_control"

# ---------------------------------------------------------------------------
# Experimental design
# ---------------------------------------------------------------------------
BUCKETS = list("ABCDE")
N_OCCASIONS = 31
BASELINE_OCCASION = 1
OCCASION_TIME_OFFSET_H = 0.5        # time of an occasion = protocol start + 30 min
T_REF_C = 30.0                      # centring temperature of the temperature models

# ---------------------------------------------------------------------------
# Acoustic measures
# ---------------------------------------------------------------------------
SAMPLE_RATE_HZ = 48000
EXCERPT_S = 600.0
WINDOW_S = 15.0
BANDS = [(64.0 * 2 ** i, 128.0 * 2 ** i) for i in range(8)]       # 64-128 ... 8192-16384 Hz
NONLINEAR = ["PE", "LZC78", "RSE4", "RSR"]
CONVENTIONAL = ["ACI", "AEI", "LogBandEnergy", "SpectralEntropy", "SpectralFlatness", "TemporalEntropy"]
MEASURES = NONLINEAR + CONVENTIONAL
FAMILY = {**{m: "nonlinear" for m in NONLINEAR}, **{m: "conventional" for m in CONVENTIONAL}}

# Parameters of the primary analysis; the sensitivity scenarios change them one at a time.
PRIMARY = dict(window_s=WINDOW_S, lambda_factor=1.0, lambda_exponent=-0.5, rms_k=5.0,
               lzc_alphabet=100, pe_order=4, rse_block=4, rse_alphabet=10)

# name: (parameter changes, affected measures)
SCENARIOS = {
    "window_low": (dict(window_s=7.5), MEASURES),
    "window_high": (dict(window_s=30.0), MEASURES),
    "lambda_low": (dict(lambda_factor=0.5), NONLINEAR),
    "lambda_high": (dict(lambda_factor=2.0), NONLINEAR),
    "lambda_power_low": (dict(lambda_exponent=-0.25), NONLINEAR),
    "lambda_power_high": (dict(lambda_exponent=-1.0), NONLINEAR),
    "rms_cycles_low": (dict(rms_k=2.5), NONLINEAR),
    "rms_cycles_high": (dict(rms_k=10.0), NONLINEAR),
    "lzc_alphabet_low": (dict(lzc_alphabet=50), ["LZC78", "RSR"]),
    "lzc_alphabet_high": (dict(lzc_alphabet=200), ["LZC78", "RSR"]),
    "pe_order_low": (dict(pe_order=3), ["PE"]),
    "pe_order_high": (dict(pe_order=5), ["PE"]),
    "rse_block_low": (dict(rse_block=3), ["RSE4"]),
    "rse_block_high": (dict(rse_block=5), ["RSE4"]),
    "rse_alphabet_low": (dict(rse_alphabet=8), ["RSE4"]),
    "rse_alphabet_high": (dict(rse_alphabet=12), ["RSE4"]),
    "combined_low_stress": (dict(window_s=7.5, lambda_factor=0.5, rms_k=2.5, lzc_alphabet=50, pe_order=3,
                                 rse_block=3, rse_alphabet=8), NONLINEAR),
    "combined_high_stress": (dict(window_s=30.0, lambda_factor=2.0, rms_k=10.0, lzc_alphabet=200, pe_order=5,
                                  rse_block=5, rse_alphabet=12), NONLINEAR),
}
# Standardisation scenario of the sensitivity analysis: centring within buckets without scaling
CENTRING_SCENARIO = "centring_only"

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
MODELS = ["time-only", "T-only", "L_time", "L"]          # fitted by least squares
SURROGATE_MODELS = ["T-only", "L_time", "L"]             # re-evaluated with every temperature surrogate
N_TEMPERATURE_SURROGATES = 100_000
TEMPERATURE_SURROGATE_SEED = 20261012
N_BETWEEN_BUCKET_PERMUTATIONS = 100_000   # within-occasion permutations of the temperature differences between buckets
BETWEEN_BUCKET_SEED = 20261102
