# Papers

This directory holds the [JOSS](https://joss.theoj.org/) submission for
`strapdown-rs`, under review at
[openjournals/joss-reviews#11377](https://github.com/openjournals/joss-reviews/issues/11377).
The review reads the paper from `main`. The `Draft PDF` GitHub Actions workflow
(`.github/workflows/draft-pdf.yml`) builds it on every pull request and push to `main` that
touches `papers/joss/`, and uploads the PDF as a workflow artifact.

- `joss/paper.md` — the submission manuscript
- `joss/paper.bib` — its bibliography

## Research papers

The research manuscripts that previously lived here (`anomaly_rbpf/`,
`anomaly_ukf/`, `data_paper/`) are not part of the v1.0 release scope and have
been removed from version control. They are ignored via `.gitignore`, so local
working copies are left untouched.
