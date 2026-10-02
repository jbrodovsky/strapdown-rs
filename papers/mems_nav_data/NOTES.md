# Open items for the authors

Status: complete draft for *Data in Brief*. It compiles cleanly (`latexmk -pdf main.tex`: no errors, warnings, undefined references or overfull boxes). It runs to 22 pages in the `elsarticle` 12pt preprint layout, and `texcount` gives about 4,100 words of running text plus 360 words of captions. Every red `[TODO: ...]` in the PDF is listed in section 1, and **none may remain at submission**.

## 1. Placeholders in the manuscript

| Where | TODO | Needs |
|---|---|---|
| `main.tex` affiliation | `[TODO: department]` | Department name(s) for both authors (Temple University is all that is stated now). |
| Specifications table, Subject | choose from the Data in Brief subject list | DiB has a fixed subject list; "Engineering" is a guess. |
| Specifications table, Data collection | describe placement and mounting in the vehicle, and the vehicles used | Not recorded anywhere I could find. Reviewers will ask. |
| Specifications table, Data accessibility | new Zenodo DOI (twice) | See section 2. |
| Sec. 2.1 Repository layout | new Zenodo DOI; confirm final directory names | The `raw/ processed/ maps/ benchmark/` layout is a **proposal**. Make the record match it or edit the text. |
| Sec. 3.1 Collection | how the phones were held; whether that changed during a drive; which vehicles | As above. |
| Sec. 3.1 Collection | confirm no extra hardware or external reference receiver | I assumed none; nothing in the data suggests one. |
| Sec. 3.6 Baselines | which measurements each filter consumed; the strapdown-rs commit that produced the benchmark files | The current `core/src/messages.rs` emits barometric relative altitude and a magnetometer yaw at 1 Hz by default. The RBPF uses the barometer as a loop in its mechanization, not as an update. I could not tie the files in `data/output_real2` to a commit. |
| Sec. 4 Limitations | state what is known about phone placement | |
| Ethics statement | volunteers' written consent; IRB determination; trimming trace ends near residences; removing the `device id` field | See section 3. |
| CRediT | confirm roles, including funding acquisition | Drafted as JB: everything except supervision; PD: supervision and review. |
| Acknowledgements | funding statement | The volunteer names are carried over from the IJRR draft. |
| Competing interests | confirm | Standard "none" wording. |

## 2. Zenodo release (author action)

- The existing record (doi:10.5281/zenodo.17582434) holds the old preprint. The public GitHub repository `jbrodovsky/mems-nav-dataset` has 17 trajectories at about 1 Hz. This article describes a new release:
  - all 29 raw recordings at their delivered rate (`data/raw`, 22.2 GB)
  - the 27 processed 10 Hz trajectories **from `data/input_real`, not `data/input`**: `input/` carries simulated ADXL355/RM3100 `grav_*`/`mag_*` columns
  - `segments.json`
  - the three map tiles per trajectory
  - the benchmark configs (`conf/real/{ekf,ukf,rbpf}_{truth,degraded}.toml`) and their `performance_summary.csv` files
- Decide between a new version of the existing record and a new concept DOI. Then fill both DOI placeholders and `memsnav-preprint` in `references.bib`.
- Licence: the GitHub dataset repository is MIT, which is a software licence. For data, CC BY 4.0 is the usual choice and is accepted by DiB. State the licence in the Specifications table once chosen.
- Before uploading, check:
  - `data/raw/2023-08-06_14-48-05/` contains a stray zip from a 2025-06-21 recording. Its device id matches the 2025-06-26 iPhone recording. Inspect it or exclude it.
  - `StudyMetadata.json` files exist in many 2025 recordings. I did not open them; check them for personal information.
  - `2025-06-13_14-55-15` contains only `Metadata.csv` and an empty `Annotation.csv`. Ship it or drop it; the text currently counts it among the 29 and says it has no data.
- Map tiles are third-party products (SRTM15+, Sandwell et al. free-air anomaly, WDMAM) fetched through GMT. Confirm that redistributing the cropped tiles is permitted. If it is not, ship the PyGMT download step instead and edit Sec. 2.4.

## 3. Ethics and privacy (author confirmation needed)

- Most drives start and end at the same few places (for example around 40.10 N, 75.29 W, and 40.03 N, 75.22 W). Those are probably residences of the authors or volunteers. DiB requires explicit attention to identifiable location data. Consider trimming the first and last few hundred metres of every trace in the processed **and** raw files, or document consent for publishing them as they are.
- `Metadata.csv` carries an application-generated `device id` (8 distinct values for 7 models). Consider replacing it with an anonymous label.
- Volunteer consent and IRB status are unknown to me.

## 4. Device list (confirm)

The models are as logged: Pixel 6a, Pixel 9 Pro, Pixel 9 Pro XL, SM-S921U, SM-A146U, iPhone 12 mini and iPhone 13 mini. The manuscript names the Samsungs by model code only. If you want marketing names, SM-S921U is (I believe) the US Galaxy S24 and SM-A146U the US Galaxy A14 5G; please verify. The iPhone 12 mini appears under two different app device ids (4 recordings, and 2025-06-26). Say whether that is one phone after a reinstall or two phones. Table 1 groups by model either way.

## 5. Numbers that differ from the brief, or were not recomputed

Everything in the text comes from `tables/numbers.tex` (generated) or from a source given below. Discrepancies with the brief:

- **2025-06-14_21-17-02.** The brief, and the comment in `conf/ekf_degraded.toml`, give 587 rows and a median advertised horizontal accuracy of 16.2 m. The current files give the following:
  - 200 fixes in the raw `Location.csv`, 197 in the processed file, over 586 s (0.34 fixes/s, gaps up to 36.6 s)
  - median horizontal accuracy 12.2 m (11.9 m in the raw file)
  - 9 fixes at 999 m or worse, up to 1,100 m

  587 is roughly the 1 Hz row count of a 586 s recording, so the old numbers probably come from the 1 Hz pipeline. The text uses the computed values.
- **Benchmark arm.** The brief points to `data/output/...` (headline: RBPF degraded n=25, median 665 km). `data/output` was run on `data/input`, the synthetic gravity/magnetic arm, with 1,000 RBPF particles. The released data are the phone channels, so Table 4 uses `data/output_real2` (`BENCHMARK_ARM = "phone"` in `build.py`), with 2,000 particles per `conf/real/rbpf_*.toml`. The EKF and UKF medians and every full-GNSS number are identical in the two arms. The differences:

  | Quantity | `output_real2` (used) | `output` |
  |---|---|---|
  | EKF degraded mean | 611.4 m | 601.7 m |
  | RBPF degraded n | 26 (missing 2025-07-08_14-12-53) | 25 |
  | RBPF degraded median | 817.7 km | 664.6 km |

  Set `BENCHMARK_ARM = "dedicated"` to switch.
- **Gravity norm.** The dissertation says "85% of samples within 1 mGal on the median trajectory". That is reproduced at 10 Hz (85%). At the **raw** rate, though, every sample of every recording except the SM-A146U lies within 1 mGal of the recording maximum (24 of 25 recordings). The text reports both; the 85% figure is an artefact of 0.1 s vector averaging. The dissertation's Pixel constant is 9.810002 m/s²; the median of the raw per-recording maxima here is 9.810004 m/s². The SM-A146U is **not** pinned: its deficit from its maximum has a median of 73 mGal and a 5th–95th percentile range of 22–156 mGal.
- **Cited from the dissertation, not recomputed:**
  - the 56 µT body-fixed horizontal field
  - the 57% of intensity variance explained by heading
- **Read from the pipeline, not recomputed:** the gravity and magnetic SNRs (0.13 and 0.01; best 0.38 and 0.04) come from `output_real2/geostats/geo_stats_pooled.json`.
- **Recomputed and matching the dissertation:** the hard-iron correction of 67–325 µT, as the median uncalibrated minus calibrated field over the 20 processed recordings that have `MagnetometerUncalibrated`.
- **New finding, worth checking before release.** iPhone GNSS altitude is lower than the Pixel 9 Pro's on shared road cells. The median difference is -46 m with an interquartile range of -112 to 0 m (1,837 cells). Other Android phones agree with the Pixel 9 Pro to within 6 m. A pure MSL-versus-ellipsoid datum difference would have the opposite sign (geoid about -33 m here), so the cause is unknown. The text reports the difference without explaining it.

Numbers written literally in the prose, and their sources:

| Value | Source |
|---|---|
| 10 Hz, 0.1 s bins, 5 s gap, 300 s minimum | `analysis/src/analysis/preprocess.py` |
| 10 ms request; 0 for barometer and location | `Metadata.csv` |
| Sensor Logger 1.18.0–1.48.0 | `Metadata.csv` |
| 147 s | `segments.json` |
| 17 trajectories at 1 Hz | `mems-nav-dataset/data/input` |
| 1 and 3 arc-min (about 1.9 and 5.6 km); 15 arc-sec | `preprocess.download_maps` |
| 60 s, 35 m / 500 s, 1.5 m/s / 100 s, ×5 (×25), seed 42 | `conf/real/*_degraded.toml` |
| 2,000 particles | `conf/real/rbpf_*.toml` |
| 13 of 27 trajectories | `tables/devices.tex` |
| 999 m sentinel threshold | chosen by me |
| 0.002° cells, 15 m/s | chosen by me, defined in `build.py` |
| 56 and 60 Hz | `tables/channels.tex` |

## 6. References to verify

These were added from memory or from the brief, and need their details checked:

- `fu2020android`: ION GNSS+ 2020 pages and DOI
- `herath2020ronin`: ICRA 2020 pages and DOI
- `tozer2019srtm15`
- `wessel2019gmt`
- `sensorlogger`: author name, URL and year; the bib key in the old draft cited the awesome-sensor-logger GitHub repository
- `memsnav-preprint`: title, authors and year of the Zenodo record were **not** checked against Zenodo
- `strapdown-rs`: year; the note says "submitted to JOSS (under review)". Update when accepted, and do not cite it as a published 1.0.
- `brodovsky2026dissertation`: the title and the 2026 date come from `~/Code/dissertation/0-dissertation.tex`. Confirm the defence and acceptance status.
- The characterizations of other datasets in Background need confirming: which use navigation-grade references, the "last minutes" claim for the aerial and handheld sets, and the "dedicated inertial units" wording for ANSFL and the Yampolsky et al. dataset.
- Carried-over entries had abstracts, local file paths and URL-form DOIs stripped. `groves` was corrected to the 2nd edition (2013).

## 7. Venue notes

- **Data in Brief.**
  - **APC: not verified.** Check the current fee and Temple's open-access agreements before submitting.
  - DiB articles describe data and should not draw conclusions. The baseline subsection is written as validation and reference values, with no claims of superiority. A handling editor may still ask for it to be cut down.
  - DiB normally expects a "Related research article". The ITM 2026 UKF paper is listed. Confirm that it is acceptable, or use "None".
  - I did not verify DiB's word limits. The draft has about 4,100 words of text, within the 3,000–4,500 target set for this revision.
- **MDPI *Data* (fallback).** A Data Descriptor uses Summary, Data Description, Methods and User Notes. The content maps over directly: Specifications and Value become the Summary, and Limitations becomes User Notes.
- **Problems in the IJRR draft that this draft no longer has:**
  - the "truth" framing (now "reference", with an explicit no-independent-reference limitation)
  - the "education" framing
  - "It's primary contribution it to"
  - "available upon request"
  - the attitude-update sign error. The new article does not restate the mechanization equations; it cites Groves (2013) and the software.

## 8. Housekeeping

- `scripts/build.py` imports `papers/anom_combined/scripts/{stats,build_figures}.py` unmodified and sets `sys.dont_write_bytecode`. An earlier run, before that flag was added, may have written `papers/anom_combined/scripts/__pycache__/`, which is git-ignored there.
- `scripts/raw_inventory.csv` and `scripts/trajectory_stats.csv` are build by-products (per-file rates and spans, and per-trajectory statistics). `device_id` is dropped from the latter.

- **Benchmark provenance.** The benchmark table reads `data/output_real2`, whose run configurations were not archived (see `papers/anom_combined/NOTES.md`, item 6). The RBPF particle count is therefore a TODO in the text rather than the 2,000 that `conf/real/` lists.
