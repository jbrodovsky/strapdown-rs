# strapdown-rs v1.0 work queue

Ordered execution plan for the [v1.0 milestone](https://github.com/jbrodovsky/strapdown-rs/milestone/1).
Live status lives on the [project board](https://github.com/users/jbrodovsky/projects/7); this file is the durable, reviewable copy.

The 14 milestone issues are not independent. #254 and #255 both break the `NavigationFilter`
trait; #253 would deny the very `unwrap`s that #254 removes; #262's `InsEngine` cannot exist
until that trait is object-safe. This queue exists so they are worked in an order where those
constraints are satisfied rather than discovered.

## Topology

A linear **spine** carries the breaking changes, each PR based on the one before it. Additive
work runs on **parallel** branches off `main` and merges laterally. Every branch has its own
git worktree.

**Status as of 2026-09-14.** Spine 0-7 are merged; `StrapdownError`, `ImuSample`, the 15-state
ESKF, NIS gating, ZUPT/ZARU and `InsEngine` are all on `main`. Spine 8 and 9 are not started --
no branch exists for either. Parallel 100-103 are all merged and 104 was closed as
already-satisfied, so the parallel track is complete.

That makes every *ordering* constraint this document used to carry historical. The parallel
branches' old merge points ("before PR 1", "rebase onto PR 2") were satisfied by `main` itself
once spine 2 landed, so 101-103 were rebased onto plain `main` and worked concurrently rather
than stacked. The tables below record what happened; they are no longer an instruction to
sequence anything.

## Spine

| # | Branch | Issues | Summary | Status |
|---|---|---|---|---|
| 0 | [`v1/00-test-speed`](queue/00-test-speed.md) | -- | Test-suite runtime | merged |
| 1 | [`v1/01-eskf-fix`](queue/01-eskf-fix.md) | #266 | ESKF vertical-channel divergence | merged |
| 2 | [`v1/02-filter-api`](queue/02-filter-api.md) | #254, #255 | StrapdownError + ImuSample (breaking API) | merged |
| 3 | [`v1/03-ned-default`](queue/03-ned-default.md) | #255 | Default frame to NED | merged |
| 4 | [`v1/04-eskf-primary`](queue/04-eskf-primary.md) | #258 | 15-state ESKF as the default filter | merged |
| 5 | [`v1/05-filters`](queue/05-filters.md) | #259 | EKF, UKF, PF/RBPF on the new API | merged |
| 6 | [`v1/06-aiding`](queue/06-aiding.md) | #260, #261 | NIS gating + ZUPT/ZARU | merged |
| 7 | [`v1/07-engine`](queue/07-engine.md) | #262 | InsEngine builder + lever-arm compensation | merged (#276) |
| 8 | [`v1/08-validation`](queue/08-validation.md) | #264 | Integration suite + ground-truth validation | **not started** |
| 9 | [`v1/09-release`](queue/09-release.md) | #265, #253 | Docs, examples, release automation + strict-lint deny flip | **not started** |

The `Base` column is gone: every merged spine branch was rebase-merged into `main`, and the two
outstanding items have no branch yet. Branches `v1/02-filter-api` through `v1/07-engine` still
exist on the remote but their PRs are merged; they are safe to prune now that no open PR is
based on any of them.

## Parallel

| # | Branch | Issues | Summary | Status |
|---|---|---|---|---|
| 100 | [`v1/p-lint-config`](queue/p-lint-config.md) | #253, #263 | Lint config (**enforced**, not warn-level) + feature gating | merged |
| 101 | [`v1/p-imu-quality-cov`](queue/p-imu-quality-cov.md) | #257 | `auto_covariance` from `IMUQuality` | merged (#280) |
| 102 | [`v1/p-calibration`](queue/p-calibration.md) | #256 | `ImuCalibration` | merged (#281) |
| 103 | [`v1/p-alignment`](queue/p-alignment.md) | #257 | Coarse alignment and initialisation | merged (#282) |
| 104 | `v1/p-release-automation` | #265 | CI and release workflow | **closed, won't implement** |

101-103 were worked concurrently off current `main`, not stacked, and all three merged. The
concurrency cost exactly two conflicts, both in `core/src/lib.rs` and both one line: 102 and
103 each add a `pub mod` declaration, and 101's `auto_covariance` shares the `impl IMUQuality`
block with later work. Neither needed a decision -- both sides were kept.

104 was closed because `main` already satisfied all three of its acceptance criteria: the CI
matrix covers Linux, macOS and Windows, both jobs pin `dtolnay/rust-toolchain@1.91` to match
`rust-version` in `Cargo.toml`, and `.github/workflows/publish.yml` threads a `workflow_dispatch`
`dry_run` input through all three crates in dependency order. The only outstanding part is
*running* the dry run, which is a workflow dispatch rather than a code change, so it is tracked
under #265 instead.

## Issues split across two PRs

| Issue | First | Then | Why |
|---|---|---|---|
| #253 | `v1/p-lint-config` (**all but the zero-panic lints**) | `v1/02-filter-api` (`unwrap_used`, `expect_used`, `panic`, `missing_errors_doc`, `missing_panics_doc`, `needless_pass_by_value`) | Superseded plan: the pedantic/nursery backlog was cleared in queue 100 rather than deferred to 9, so the strict gate is live for PRs 2-9 instead of arriving after them. The lints that remain off are the ones that need `StrapdownError` to be satisfiable at all, so they switch on in queue 2 alongside it -- not in queue 9. |
| #255 | `v1/02-filter-api` (Delta-v/Delta-theta) | `v1/03-ned-default` (frame) | Two orthogonal risks. Separating them means an integration-metric shift is attributable to one change. |
| #257 | `v1/p-imu-quality-cov` (`auto_covariance`) | `v1/p-alignment` (rest) | Both halves are merged and #257 is closed. The split held, the ordering did not: #266 was fixed in spine 1 before `auto_covariance` existed, so the "lands before PR 1" constraint expired unused. `auto_covariance` shipped as **opt-in** -- `engine::DEFAULT_INITIAL_COVARIANCE` and `sim::initialize_eskf` are unchanged, so no existing integration metric moved. Retuning the default remains #266's call. |
| #265 | ~~`v1/p-release-automation` (CI)~~ | `v1/09-release` (docs + the publish dry run) | The split is void: the CI half turned out to be already done on `main`, so 104 was closed rather than implemented. All of #265 now sits in spine 9. |

## Working the queue

```bash
git config rerere.enabled true      # same conflicts recur across every rebase
git config rebase.updateRefs true   # carries downstream branch tips when the base moves
```

### Merging a spine PR

GitHub now treats these branches as a first-class **stack** and does the restacking itself.
The manual recipe this section used to carry no longer runs: `gh pr merge --rebase` is refused
("must be merged using the asynchronous merge REST API"), `PUT .../pulls/<N>/merge` returns 403
with the same redirection, and retargeting the child with
`PATCH .../pulls/<N+1> -f base=main` returns 422 ("Cannot change the base branch because the
pull request is part of a stack"). Merge through the async endpoint instead:

```bash
HEAD_SHA=$(git rev-parse origin/v1/0N-current)   # the head you actually reviewed and tested
gh api -X PUT repos/OWNER/REPO/pulls/<N>/merge-async \
  -f merge_method=rebase -f sha="$HEAD_SHA"      # -> {"status":"pending","details":{"uuid":...}}
gh api repos/OWNER/REPO/pulls/<N>/merge-async/<uuid> --jq '.status'   # poll until != pending
```

Passing `sha=` pins the merge to the head you verified, so a push landing while you look away
fails the merge rather than silently shipping.

On merge GitHub retargets the child PR's base to `main` and force-pushes the child branch
rebased onto the new `main`, so there is nothing left to rebase by hand. `git fetch --prune`
and confirm.

Deleting the merged branch is the one step still worth care. GitHub **auto-closes** a PR when
its base branch is deleted, and then permanently refuses to reopen it if the head was
force-pushed in the meantime -- which the restack always does. Queue position 1 was lost this
way once (#270, replaced by #285). By the time you get here the child's base is already `main`,
so this is safe; confirm it rather than assume it:

```bash
gh api repos/OWNER/REPO/pulls/<N+1> --jq '.base.ref'   # must print "main"
git push origin --delete v1/0N-current
```

Queue 100 cleared the pedantic/nursery backlog, so `cargo clippy --fix` no
longer has tree-wide work to do; do not run it speculatively regardless, and
note that its `use super::*` expansion is not feature-aware (it broke
`--no-default-features` once already -- see the queue 100 outcome).

Spine PRs are **rebase-merged, not squashed** -- squashing rewrites the base and forces a manual
`--onto` on every subsequent rebase. Keep `cargo fmt` output in its own commit per branch; on a
rebase conflict, `--skip` it and regenerate rather than resolving formatting by hand.

