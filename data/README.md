# Hearing the heat: soil probe recordings of rapid organic decomposition (150 x 10 min)

Audio data for: Eisenhauer, J. Hearing the heat: a proof-of-concept study of nonlinear sound measures for
distinguishing phases of rapid organic decomposition.

| | |
|---|---|
| This data set | https://doi.org/10.5281/zenodo.23891122 |
| Author | Jakob Eisenhauer, Department of Earth and Environmental Sciences, Lund University |
| Licence | CC BY 4.0 (audio and CSV files) |

## Summary

150 recordings of 10 min each, made with a piezoelectric soil probe in five buckets (A-E) of decomposing
vermicompost and hay. The buckets were recorded at 31 occasions between 27 June and 12 July 2026, covering one
heating-cooling cycle (internal temperature 28 to 39 to 26 degrees C). Together with the temperatures, the
recordings allow the analysis of the article to be reproduced.

## Files

| File | Content |
|---|---|
| `sample_A_recordings.zip` ... `sample_E_recordings.zip` | The 30 recordings of bucket A to E (about 1.7 GB each, WAV files without subfolders). Unpack all into one folder (`data/audio/` of the analysis code) |
| `featured_*.wav` | Ten example recordings for communication purposes (see below), also contained in the zip files |
| `recordings.csv` | Per recording: file, source file, bucket, occasion, session, source duration (s), crop start (s), recorder start time |
| `measurement_protocol.csv` | Per occasion: date, session, planned start time and internal temperature (degrees C) of buckets A-E |

## Recordings

- File name: `<bucket><occasion>_<session>_cropped.wav`, e.g. `A12_D_cropped.wav`. Occasion 1 to 31. Session
  `Baseline` (occasion 1, 18:00), `D` (afternoon, 13:00) or `N` (evening, 21:00).
- Mono, 16-bit PCM, 48 kHz, 600 s per file. Recorder: ALLSOUNDER ALISO with a piezoelectric soil probe
  (Elekon AG, Switzerland), gain 0 dB. The recordings are not calibrated, so levels are relative.
- The probe was inserted anew for every recording near the centre of the bucket (+-2 cm) at 10.5 +- 0.5 cm depth.
  The buckets were recorded in the order A to E, about 15 min apart.
- Cropping: each full recording (11.5 to 15 min) lost its first 1.0 to 4.5 min (insertion and settling of the
  probe) and its last 30 s. Crop starts are in `recordings.csv`. The uncropped recordings are not included.

## Missing recordings

Five of the 155 planned bucket-occasion recordings are missing (3 not recorded, 2 damaged): **A25, B4, C3, D13,
E13** (bucket and occasion).

## Experiment

Five black polypropylene buckets (22.1 +- 0.6 L) were filled to about 15.5 L with reactivated vermicompost and hay
(wet-mass ratio 2:1) from one batch and kept indoors under a loose polyethylene film. Monitoring began on
27 June 2026 at 18:00 and was repeated at 31 occasions, normally at 13:00 and 21:00. At every occasion, the internal
temperature was measured at 14-15 cm depth with a digital probe thermometer. Details: see the article and its
Supplementary Methods. The data come from one batch and one cycle and do not represent decomposition in general.

## Sounds of interest for communication (not part of the paper analysis)

This section is meant for science communication, sonification and teaching and is **not connected to the analysis
in the article**. Besides the sounds of the decomposing material, the recordings contain short, broadband sound
events that may come from animals moving or feeding in the material (not identified), and indoor sounds such as
50 Hz mains hum (present in roughly half of the recordings) and building noise. The recordings carry no sound-event
labels. Time, session and temperature are given in the CSV files.

The ten featured recordings are examples with many short, broadband sound events:

| File | Bucket | Occasion | Date |
|---|---|---|---|
| `featured_A20_D_cropped.wav` | A | 20 | 7 July 2026, 13:00 |
| `featured_A22_D_cropped.wav` | A | 22 | 8 July 2026, 13:00 |
| `featured_B16_D_cropped.wav` | B | 16 | 5 July 2026, 13:00 |
| `featured_B18_D_cropped.wav` | B | 18 | 6 July 2026, 13:00 |
| `featured_B21_N_cropped.wav` | B | 21 | 7 July 2026, 21:00 |
| `featured_B27_N_cropped.wav` | B | 27 | 10 July 2026, 21:00 |
| `featured_C7_N_cropped.wav` | C | 7 | 30 June 2026, 21:00 |
| `featured_C22_D_cropped.wav` | C | 22 | 8 July 2026, 13:00 |
| `featured_D17_N_cropped.wav` | D | 17 | 5 July 2026, 21:00 |
| `featured_E29_N_cropped.wav` | E | 29 | 11 July 2026, 21:00 |

## Reuse

CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/): please credit the data set and the article. The analysis
code is archived separately under the MIT licence.
