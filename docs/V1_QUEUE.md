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

## Spine

| # | Branch | Base | Issues | Summary |
|---|---|---|---|---|
| 0 | [`v1/00-test-speed`](queue/00-test-speed.md) | `main` | -- | Test-suite runtime |
| 1 | [`v1/01-eskf-fix`](queue/01-eskf-fix.md) | `v1/00-test-speed` | #266 | ESKF vertical-channel divergence |
| 2 | [`v1/02-filter-api`](queue/02-filter-api.md) | `v1/01-eskf-fix` | #254, #255 | StrapdownError + ImuSample (breaking API) |
| 3 | [`v1/03-ned-default`](queue/03-ned-default.md) | `v1/02-filter-api` | #255 | Default frame to NED |
| 4 | [`v1/04-eskf-primary`](queue/04-eskf-primary.md) | `v1/03-ned-default` | #258 | 15-state ESKF as the default filter |
| 5 | [`v1/05-filters`](queue/05-filters.md) | `v1/04-eskf-primary` | #259 | EKF, UKF, PF/RBPF on the new API |
| 6 | [`v1/06-aiding`](queue/06-aiding.md) | `v1/05-filters` | #260, #261 | NIS gating + ZUPT/ZARU |
| 7 | [`v1/07-engine`](queue/07-engine.md) | `v1/06-aiding` | #262 | InsEngine builder + lever-arm compensation |
| 8 | [`v1/08-validation`](queue/08-validation.md) | `v1/07-engine` | #264 | Integration suite + ground-truth validation |
| 9 | [`v1/09-release`](queue/09-release.md) | `v1/08-validation` | #265, #253 | Docs, examples, release automation + strict-lint deny flip |

## Parallel

| # | Branch | Issues | Summary | Merge point |
|---|---|---|---|---|
| 100 | [`v1/p-lint-config`](queue/p-lint-config.md) | #253, #263 | Lint config (warn-level) + feature gating | before everything |
| 101 | [`v1/p-imu-quality-cov`](queue/p-imu-quality-cov.md) | #257 | auto_covariance from IMUQuality | before PR 1 |
| 102 | [`v1/p-calibration`](queue/p-calibration.md) | #256 | ImuCalibration | rebase onto PR 2 for the final signature |
| 103 | [`v1/p-alignment`](queue/p-alignment.md) | #257 | Coarse alignment and initialisation | rebase onto PR 2 |
| 104 | [`v1/p-release-automation`](queue/p-release-automation.md) | #265 | CI and release workflow | any time |

## Issues split across two PRs

| Issue | First | Then | Why |
|---|---|---|---|
| #253 | `v1/p-lint-config` (warn) | `v1/09-release` (deny) | Denying `unwrap_used` before #254 removes the unwraps forces `#[allow]` scaffolding that #254 then deletes. Pedantic's mechanical churn touches nearly every hunk PRs 2-9 also touch, so the deny flip lands when nothing is stacked above it. |
| #255 | `v1/02-filter-api` (Delta-v/Delta-theta) | `v1/03-ned-default` (frame) | Two orthogonal risks. Separating them means an integration-metric shift is attributable to one change. |
| #257 | `v1/p-imu-quality-cov` (`auto_covariance`) | `v1/p-alignment` (rest) | `auto_covariance` is needed to retune ESKF covariance during the #266 fix, so it lands before PR 1. |
| #265 | `v1/p-release-automation` (CI) | `v1/09-release` (docs) | The CI half depends on nothing and can land at any time. |

## Working the queue

```bash
git config rerere.enabled true      # same conflicts recur across every rebase
git config rebase.updateRefs true   # carries downstream branch tips when the base moves
```

### Merging a spine PR

Order matters. GitHub **auto-closes** a PR when its base branch is deleted, and then permanently
refuses to reopen it if the head was force-pushed in the meantime -- which a stack rebase always
does. Queue position 1 was lost this way once (#270, replaced by #285). So:

```bash
OLD=$(git rev-parse origin/v1/0N-current)      # 1. record the tip BEFORE merging
gh pr merge <N> --rebase                       # 2. merge WITHOUT --delete-branch
gh api -X PATCH repos/OWNER/REPO/pulls/<N+1> -f base=main   # 3. retarget the child FIRST
git rebase --onto origin/main "$OLD" v1/0N+1-next           # 4. then rebase
git push --force-with-lease origin v1/0N+1-next
git push origin --delete v1/0N-current         # 5. only now delete the merged branch
```

Note that GitHub's rebase-merge **rewrites commit SHAs** even when the branch is a
fast-forward, which is why step 1 records the old tip and step 4 needs `--onto` rather than a
plain `git rebase main`.

Spine PRs are **rebase-merged, not squashed** -- squashing rewrites the base and forces a manual
`--onto` on every subsequent rebase. Keep `cargo fmt` output in its own commit per branch; on a
rebase conflict, `--skip` it and regenerate rather than resolving formatting by hand. Do not run
`cargo clippy --fix` tree-wide before queue position 9.

