# Open items before submission

Status (2026-10-02): first complete draft. 19 pages in the NAVIGATION template, including
references (the limit is 20). 9 figures (limit 15), 9 tables, abstract 267 words.

## Needs an author decision

1. **Correction framing (Section 2).** The paper withdraws the ITM 2026 and Pacific PNT 2026
   results explicitly. Agree the wording with Dr. Dames, and decide whether to tell ION about
   the conference papers separately (erratum or retraction notice). Under ION's policy a journal
   version of a conference paper is normal; a journal version that *reverses* it should say so
   in the cover letter as well.
2. **Title.** The working title is descriptive. Alternatives:
   "Is a Smartphone Enough? ..." or "Sensor Quality, Not the Estimator, Decides ...".
3. **Abstract length.** 267 words. The NAVIGATION submission page does not give a limit; check
   it in ScholarOne and trim if needed.
4. **Experimental-design diagram.** Planned as a TikZ figure and left out to stay under 20
   pages. Add it if a reviewer asks, or if the page count drops.
5. **Front matter.** Affiliation, correspondence email (`jbrodovsky@temple.edu` was copied from
   the data paper), acknowledgments, and the conflict-of-interest statement. NAVIGATION also
   needs an author photo and, on acceptance, a 2–3 minute video abstract.

## Provenance gaps

6. **Smartphone-arm runs.** `data/output_real2` (identical to `data/output_real`, Sep 26
   18:31–20:27) has no recorded recipe. `conf/real/rbpf_*.toml` no longer parses with the
   current code. Before submission, regenerate the smartphone arm from a committed config set
   (e.g. `conf/real/` updated to the Canciani RBPF keys, with paths pointing at
   `data/input_real` and `data/output_real`) and the commit that produced `data/output`. Then
   rerun `scripts/*.py`. The data-availability section promises that every number can be
   regenerated, and right now this one arm cannot be.
7. **Commit of record.** Section 5 cites `827db06` (2026-09-26 09:40, the last commit before
   the `data/output` runs at 21:26–23:01). Confirm the binary was built from it.
8. **Conference-era defects.** Every defect in Section 2 was confirmed by reading the source at
   `4a4e3de` (2025-11-14, "Write UKF Anomaly paper") and `4ea73af` (2026-03-11, "anomaly rbpf
   paper"). This assumes those are the revisions the conference runs used. The missing
   heading update (aided vs. unaided streams) applies only to `4ea73af`: at `4a4e3de` neither
   stream had one.

## Numbers that differ from the dissertation (the paper's own are recomputed from the CSVs)

- **UKF α = 0.1**, not 1e-3. `DEFAULT_UKF_ALPHA` was raised in `b70cd55` (2026-09-18), before
  the 2026-09-26 runs, and no config overrides it. Dissertation Ch. 3–4 should be corrected.
- **RBPF particles = 1,000** in every config, not 2,000 with full GNSS.
- **Map gradients.** Along-track median 1.41 mGal/km and 6.28 nT/km (central differences on
  the grid, `build_figures.median_gradients`), against 1.3 and 6.9 in the dissertation. The
  bounds move accordingly: 93 / 1,201 / 39 / 0.62 km against ~100 / 1,100 / 40 / 0.6 km.
- **Gravity-norm vs. map slope.** −0.02 against −0.012 (correlation −0.055 matches).
- **Heading dependence of the magnetic intensity.** 52% of variance from a cos/sin fit to GNSS
  course over ground, against 57% in the dissertation from a heading-circle fit. The
  dissertation's 56 µT body-fixed field and 6,257 nT 60-s residual were **not** reproduced,
  so the paper does not quote them. The 67–325 µT OS hard-iron offset and the
  raw-accelerometer result (50–90× the map signal) are cited to the dissertation, not
  recomputed.
- **The two unaided sets** differ (EKF mean 611.4 vs. 601.7 m) because rescaling the
  magnetometer vector changes the heading input in the last bit. That changes one EKF drive by
  261 m and 22 of 25 RBPF drives by >1% (median 20%). The paper reports this. Each arm is
  scored against its own unaided runs.
- **Paired statistics.** The aided RMSE comes from `analyze geoperformance`
  (`*_detailed_results.csv`). It differs from `analyze performance` by <0.1 m on some drives
  (e.g. 759.95 vs. 759.90 m), presumably time alignment. The baseline table uses
  `performance`, the ratios use `geoperformance`, and each is internally consistent.
- **Known data-integrity issues, handled.** `data/output/rbpf/{grav,both}/2023-08-04_21-47-58.csv`
  are stale leftovers whose baseline failed. The matched-set rule (both runs must exist) drops
  them.

## Citations

- `strapdown-rs`: update to the JOSS DOI once joss-reviews#11377 is accepted.
- `mems-nav-dataset`: cited as the Zenodo record 10.5281/zenodo.17582434. That record holds 17
  drives at ~1 Hz, while this paper uses 27 segments at 10 Hz. Publish the new release and cite
  its DOI.
- `brodovsky2026dissertation`: confirm the title, year and degree status. Check NAVIGATION's
  policy on prior appearance in a dissertation (usually not prior publication; say so in the
  cover letter).
- `groves_ch5` / `groves_ch14` render as Groves (2013a/b). They are the same book, so cite it
  once with chapter pinpoints if the editor prefers.

## Venue facts (ion.org/navi/submit-navi.cfm, read 2026-10-02)

Regular paper ≤ 20 pages, ≤ 15 figures, APA 7 with DOIs, ScholarOne submission. Open access
APC: $1,000 (member) / $1,500 (non-member) for ≤ 20 pages.
