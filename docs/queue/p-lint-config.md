# Queue 100 -- Lint config (enforced) + feature gating

| | |
|---|---|
| **Branch** | `v1/p-lint-config` |
| **Base** | `main` |
| **Issues** | #253, #263 |
| **Queue position** | 100 |
| **Merge point** | before everything |
| **Queue doc note** | deviates from `V1_QUEUE.md` position 9 deny flip -- see Why |

## Why

The original plan: landing `-D pedantic` early would touch nearly every hunk
PRs 2-9 also touch, and denying `unwrap_used` before #254 removes the unwraps
forces `#[allow]` scaffolding that #254 then deletes -- so config and signal now
at `warn`, deny flip in PR 9.

**Superseded.** The lints are enforced here instead, to get the feedback while
the filter code is still being written rather than after. The two lint families
that genuinely cannot be satisfied without #254's error type are deferred to
queue 2 by name; everything else is resolved. See Outcome.

## Acceptance criteria

- [x] `[workspace.lints]`, `rustfmt.toml`, `clippy.toml`
- [x] ~~Non-blocking CI lint job~~ -- superseded: the lints are **enforced**, so a
      non-blocking job reporting the same findings would be dead CI time
- [x] `--no-default-features` and `--all-features` both build clean
- [x] MIT licensing reaffirmed across all crates (per-crate `LICENSE`, since the
      root file is not included in a published crate's package)

## Outcome

**Deviation from the plan, decided deliberately:** the queue scheduled the deny
flip for position 9. It is enforced here instead -- the blocking gate keeps a
bare `-D warnings`, which promotes every warn in the lint table to an error. The
point of the strict groups is to catch defects while the filter code is still
being written.

That meant clearing the backlog rather than deferring it: **1,014 findings ->
0**, `pixi run lint` green with `-D warnings` under both `--all-features` and
`--no-default-features`.

### What was fixed vs. configured

`cargo clippy --fix` handled the mechanical bulk; ~90 were fixed by hand
(merged match arms, hoisted consts, `finish_non_exhaustive` on the three eliding
`Debug` impls, `Debug` for `Event`/`EventStream`/`RaoBlackwellizedParticleFilter`,
lazy `unwrap_or_else` fallbacks, `unused_self` methods demoted to associated
functions, `check`/`check_at` relaxed from `&mut self` to `&self`).

Three findings were real defects rather than style:

- `cargo clippy --fix` expanded `use super::*` in `sim::execution` into an
  explicit list that included `Args`/`ValueEnum` -- **which exist only under the
  `clap` feature**, breaking `--no-default-features`. Caught by the minimal
  gate. They were unused; the import is now the used set only.
- Seven `.unwrap_or(Normal::new(..).unwrap())` fallbacks were built eagerly on
  every call.
- `bilinear_interpolation` returned `Option` but could never return `None`.

### Lints allowed, each with its reason in `Cargo.toml`

The largest, `suboptimal_flops` (81), was **applied, read, and reverted**. It
rewrites `a - b * c` into `b.mul_add(-c, a)`: a *fused* operation with one
rounding where the expression had two. That is a real change to results in the
WGS84 radii, the Somigliana gravity model, the quaternion update and the UKF
sigma weights -- in a crate that has already lost days to a vertical channel
diverging from a 1e-14 perturbation (#266, #286). It also destroys the
correspondence with the Groves equations that CLAUDE.md asks the mechanization
to preserve, and on haversine it emitted
`(dlat/2.0).sin().mul_add((dlat/2.0).sin(), ..)`, recomputing the sine.

Deferred to **queue 2 (#254/#255)**, where the signatures they concern are
redesigned in one pass: `unwrap_used`/`expect_used`/`panic` (48 library call
sites -- several are genuine panic paths, e.g. `geonav/src/lib.rs:382` panics on
any coordinate outside the map bounds), `missing_errors_doc`/`missing_panics_doc`,
and `needless_pass_by_value` (12, all public constructors).

Declined on domain grounds: `similar_names` (Groves-derived parallel naming),
`cast_precision_loss`/`cast_possible_truncation`/`cast_sign_loss` (the
arithmetic this crate is made of), `unreadable_literal` (constants written as
published), `float_cmp` (all 7 are exact comparisons of exactly-representable
values in tests, checked individually), `doc_markdown`, `must_use_candidate`,
`too_many_lines` (decomposition belongs with #262), `option_if_let_else` and
`redundant_pub_crate` (both readability regressions).

Only two `#[allow]`s are at a call site rather than in the table, each with a
`reason`: `unsafe_derive_deserialize` on the two records whose only `unsafe` is
`Mmap::map` on a file handle, and `struct_field_names` on the test-only
`ErrorStats`.

Part of the [v1.0 work queue](../V1_QUEUE.md) / [project board](https://github.com/users/jbrodovsky/projects/7).

