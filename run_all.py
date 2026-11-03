"""Run the complete analysis, or selected steps in the given order.

    python run_all.py                      all steps except crop
    python run_all.py analysis figures     only these steps
    python run_all.py crop                 rebuild data/audio and data/recordings.csv from the raw recordings
"""
import sys
import time

import analysis
import audio_measures
import bootstrap
import figures
import loudness_control
import mass_balance
import signal_surrogates
import tables

STEPS = {
    "crop": audio_measures.crop_recordings,               # 600 s excerpts of the raw recordings
    "measures": audio_measures.compute_measures,          # acoustic measures of all recordings (hours)
    "surrogates": signal_surrogates.run,                  # signal-level IAAFT test and linear control (hours)
    "analysis": analysis.run,                             # models, temperature surrogates, further checks
    "bootstrap": bootstrap.run,                           # Bayesian bootstrap of the fits
    "loudness": loudness_control.run,                     # level-invariant control, quantisers per recording (~30 min)
    "mass": mass_balance.run,                             # mass and moisture balance
    "tables": tables.run,
    "figures": figures.run,
}

if __name__ == "__main__":
    for name in sys.argv[1:] or [s for s in STEPS if s != "crop"]:
        t0 = time.time()
        print(f"\n=== {name}", flush=True)
        STEPS[name]()
        print(f"=== {name} done in {(time.time() - t0) / 60:.1f} min", flush=True)
