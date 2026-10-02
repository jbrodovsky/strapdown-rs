# Configuration Files

A configuration file describes a whole `strapdown-sim` run: the input and output, the mode, the
filter and its tuning, how the aiding sensors are scheduled and corrupted, and the limits that
stop a run. It is one `SimulationConfig` (`strapdown::sim`), written as TOML, YAML or JSON -- the
format is chosen by the file extension (`.toml`, `.yaml`/`.yml`, `.json`).

```bash
strapdown-sim --config scenario.toml
```

`--config` supplies the entire run. A subcommand given alongside it, and every flag belonging to
that subcommand (`-i`, `-o`, `--enu`, `--seed`, `--filter`, ...), is **ignored**. Only the global
options still act: `--log-level`, `--log-file`, `--parallel` and `--plot`. Writing a scenario into
a file once it settles is how a run is made exactly repeatable.

`strapdown-sim config` writes a complete starting file interactively; see
[Running Simulations](./simulations.md#config-the-configuration-wizard). Worked files live in
[`examples/configs/`](https://github.com/jbrodovsky/strapdown-rs/tree/main/examples/configs).

## A complete example

Every key below was run through `strapdown-sim --config` on a synthetic trajectory. Most are
optional; a minimal file needs only `mode`.

```toml
mode = "closed-loop"            # required
input = "synthetic.csv"
output = "cfg_out/cl_full.csv"
is_enu = false
generate_plot = true

[logging]
level = "warn"
file = "cfg_out/cl_full.log"

[execution_limits]
max_wall_clock_ratio = 0.5
max_wall_clock_s = 600.0
max_no_progress_s = 60.0

[health_limits]
lat_rad = [-1.0, 1.0]           # radians, not degrees
lon_rad = [-1.0, 1.0]
alt_m = [-500.0, 5000.0]
speed_mps_max = 100.0
cov_diag_max = 1e12
nis_pos_max = 200.0
nis_pos_consec_fail = 50

[closed_loop]
filter = "ukf"
ukf_alpha = 0.1
ukf_beta = 2.0
ukf_kappa = 0.0
estimate_baro_bias = true
innovation_gate = { chi_squared = { confidence = 0.999 } }
gate_recovery = { rejection_inflation = 4.0, forced_update_after = 3 }

[aiding]
seed = 7
baro_noise_std_m = 2.0
max_imu_gap_s = 2.0

[aiding.scheduler]
kind = "duty_cycle"
on_s = 100.0
off_s = 50.0
start_phase_s = 30.0

[aiding.fault]
kind = "degraded"
rho_pos = 0.99
sigma_pos_m = 5.0
rho_vel = 0.95
sigma_vel_mps = 0.3
r_scale = 2.0
tau_pos_s = 60.0
tau_vel_s = 30.0

[aiding.baro_scheduler]
kind = "fixed_interval"
interval_s = 2.0
phase_s = 0.0

[aiding.magnetometer_scheduler]
kind = "pass_through"
```

The same structure in YAML nests the tables as mappings (`aiding:` with `scheduler:` beneath it,
and so on), and in JSON as objects. The one place the three spellings differ is the innovation
gate; see [below](#innovation_gate).

## Names must be exact -- in one direction

Two kinds of mistake behave differently:

- **A misspelled value is an error.** An unknown `mode`, `filter` or `kind` stops the run before
  anything happens, naming the valid choices:
  ``unknown variant `degradd`, expected one of `none`, `degraded`, `slow_bias`, `hijack`, `combo` ``.
  So does a missing required field, such as one of a fault model's parameters.
- **A misspelled key or section is silently ignored.** Unknown keys are not rejected, so
  `ukf_alfa = 0.1` leaves `ukf_alpha` at its default, and `[aiding.faults]` in place of
  `[aiding.fault]` leaves the run with no fault at all -- a scenario that looks configured and
  simulates nothing. Check the names against this page, and against the `Mode:`/`Input:` lines
  the run logs at `info` level.

The two exceptions are the `[particle_filter]` keys that were removed when the filter was
restructured, which are refused by name; see [below](#particle_filter).

## Top-level keys

| Key | Default | Meaning |
|---|---|---|
| `mode` | **required** | what to run; see the table below |
| `input` | `"input.csv"` | input CSV file or directory, as for `-i` |
| `output` | `"output.csv"` | output CSV file or directory, as for `-o` |
| `seed` | `42` | the run's seed: the particle filter's, and the fault models' unless `[aiding] seed` is set; see [Seeds](#seeds) |
| `is_enu` | `false` | the input records are ENU rather than NED; as `--enu` |
| `parallel` | `false` | process a directory's files concurrently |
| `generate_plot` | `false` | write a PNG plot beside each dead-reckoning, closed-loop or particle-filter result |

The input and output path rules -- directory versus file, CSV only, refusing to overwrite an input
-- are the command line's; see [Running Simulations](./simulations.md#input-and-output-paths).

`mode` takes the `SimulationMode` variant names in kebab-case, not the CLI subcommand names:

| `mode` | CLI subcommand | Section it reads |
|---|---|---|
| `dead-reckoning` | `dr` | -- |
| `closed-loop` | `cl` | `[closed_loop]` |
| `particle-filter` | `pf` | `[particle_filter]` |
| `synthetic` | `syn` | `[synthetic]` |
| `open-loop` | `ol` | -- (**not implemented**: exits with an error) |

`mode = "dr"` is rejected as an unknown variant. Every mode reads `[aiding]`,
`[execution_limits]`, `[health_limits]` and `[logging]`, with two exceptions. Dead reckoning has
no aiding, so it ignores `[aiding]`; it applies the execution limits and the position and speed
health limits, as `dr` does on the command line, but not `cov_diag_max` or the NIS pair. Synthetic
mode reads only `[synthetic]` and `[logging]`.

## `[aiding]`

How the aiding measurements are delivered, and how GNSS is corrupted. The section was called
`gnss_degradation` before the v1.0 freeze, and that name is still accepted as an alias -- older
files keep working -- but `aiding` is the name to write.

| Key | Default | Meaning |
|---|---|---|
| `scheduler` | `pass_through` | when GNSS fixes are delivered; see [Schedulers](#schedulers) |
| `fault` | `none` | how GNSS fixes are corrupted; see [Fault models](#fault-models) |
| `baro_scheduler` | `fixed_interval`, 1 s | when barometric altitude is delivered |
| `magnetometer_scheduler` | `fixed_interval`, 1 s | when magnetometer heading is delivered |
| `baro_noise_std_m` | `2.23606797749979` | barometric altitude one-sigma, metres |
| `seed` | the top-level `seed` | seed of the fault models; overrides the top-level `seed` for them |
| `max_imu_gap_s` | `5.0` | longest tolerated gap in the inertial stream, seconds; `0` or less disables the check |

`max_imu_gap_s` bounds the IMU stream only. A record with a missing accelerometer or gyroscope
value produces no propagation step, and the filter coasts across the hole; a hole longer than
this limit is not a scenario but a broken recording, and the run is refused with a
`SensorStreamGap` error. Shorter gaps are tolerated with one warning. A missing GNSS fix is never
an error -- outages are what the scheduler is for. There is no command-line flag for it.

The library type also has a `baro_bias_index` field. `strapdown-sim` overwrites it from the
filter it builds, so setting it in a file has no effect.

Every snippet in the next three subsections is a **fragment of the `[aiding]` section**. Copied to
the top level of a file, it is an unknown key and is ignored.

### Schedulers

The three scheduler keys take the same three kinds, each with its own independent clock, so a
GNSS outage, a barometer outage and a heading outage can be set separately or made to overlap.
All fields of a kind are required.

```toml
[aiding.scheduler]
kind = "pass_through"      # every fix

[aiding.scheduler]
kind = "fixed_interval"    # at most one fix per interval
interval_s = 10.0
phase_s = 0.0              # time of the first emitted fix

[aiding.scheduler]
kind = "duty_cycle"        # alternate availability and denial
on_s = 120.0
off_s = 30.0
start_phase_s = 0.0
```

(Each block is an alternative; a file has one `[aiding.scheduler]`.)

`fixed_interval` takes the first usable fix at or after each tick, so a record without a fix
does not use up the tick. `duty_cycle` runs `start_phase_s` of initial availability, then `off_s`
denied and `on_s` available, repeating -- so **with `start_phase_s = 0` the run opens with an
outage** of `off_s` seconds.

The command line spells these differently: `--sched passthrough|fixed|duty`, and the duty-cycle
phase is `--duty-phase-s`, while `phase_s` in a file belongs to `fixed_interval` only.

### Fault models

A fault corrupts the content of each GNSS fix the scheduler delivers; the two compose. Faults
apply to GNSS only. **Every field of a fault is required** except the two `tau_*` keys of
`degraded`: a fault given with a field missing is refused with ``missing field ...``.

```toml
# No corruption (the default)
[aiding.fault]
kind = "none"

# AR(1)-correlated position and velocity error, with inflated advertised accuracy.
# Models low signal-to-noise or multipath.
[aiding.fault]
kind = "degraded"
rho_pos = 0.99          # AR(1) coefficient of the position error, per fix
sigma_pos_m = 3.0       # position error, m
rho_vel = 0.95          # AR(1) coefficient of the velocity error, per fix
sigma_vel_mps = 0.3     # velocity error, m/s
r_scale = 5.0           # factor on the advertised sigmas, so R grows by its square
tau_pos_s = 60.0        # optional: position-error correlation time, s
tau_vel_s = 30.0        # optional: velocity-error correlation time, s

# A slowly drifting offset. Models a soft spoof.
[aiding.fault]
kind = "slow_bias"
drift_n_mps = 0.02      # northward drift of the offset, m/s
drift_e_mps = 0.0       # eastward drift, m/s
q_bias = 1e-6           # random-walk density of the offset, m^2/s
rotate_omega_rps = 0.0  # rotation rate of the drift direction, rad/s

# A constant offset over a fixed window. Models a hard spoof.
[aiding.fault]
kind = "hijack"
offset_n_m = 50.0
offset_e_m = 0.0
start_s = 120.0         # seconds from the first record
duration_s = 60.0
```

Without `tau_pos_s`, `rho_pos` applies once per delivered fix, so the error's correlation time
depends on the fix interval; with it, the coefficient is $e^{-\Delta t/\tau}$ for the actual
interval and `sigma_pos_m` is the steady-state standard deviation. `tau_vel_s` does the same for
velocity. The [Closed Loop](./closed-loop.md#gnss-faults-what-the-fixes-say) page explains each
parameter, and [GNSS Degradation](../gnss/fault-simulation.md) the models.

**`combo`** applies several faults in sequence, each feeding the next. It exists only in a
configuration file -- `--fault` has no combo choice. Its members go in a `faults` list:

```toml
[aiding.fault]
kind = "combo"

[[aiding.fault.faults]]
kind = "slow_bias"
drift_n_mps = 0.02
drift_e_mps = 0.0
q_bias = 1e-6
rotate_omega_rps = 0.0

[[aiding.fault.faults]]
kind = "hijack"
offset_n_m = 50.0
offset_e_m = 0.0
start_s = 120.0
duration_s = 60.0
```

```yaml
aiding:
  fault:
    kind: combo
    faults:
      - kind: slow_bias
        drift_n_mps: 0.02
        drift_e_mps: 0.0
        q_bias: 1.0e-6
        rotate_omega_rps: 0.0
      - kind: hijack
        offset_n_m: 50.0
        offset_e_m: 0.0
        start_s: 120.0
        duration_s: 60.0
```

Both run on a synthetic trajectory, and the solution shows each stage: pulled about 50 m north
during the hijack window, and several metres off after it ends, as the slow drift accumulates. The
members share one fault state, so combine *different* kinds; two members of the same kind step
the same error states.

### The barometer and magnetometer

`baro_scheduler` and `magnetometer_scheduler` take the same kinds and fields as `scheduler`.
**Both default to `fixed_interval` at 1 Hz, not to `pass_through`.** Delivering one measurement
per record ties a sensor's update rate to the log's rate: right by coincidence on a 1 Hz
recording, but on a 50 Hz synthetic trajectory it delivered fifty pressure readings and fifty
derived headings a second, each entering the filter as an independent measurement with full
weight -- one heading counted fifty times. Set `kind = "pass_through"` when the log's rate really
is the sensor's rate:

```toml
[aiding.magnetometer_scheduler]
kind = "pass_through"
```

`baro_noise_std_m` is the barometer's one-sigma in **metres**, and it is the only aiding-noise
setting in a file: GNSS noise comes from each record's accuracy columns
([Input Data Format](./data-format.md#which-columns-drive-what)), and the input format has no
pressure-accuracy column. It is a standard deviation. The value it replaced was a variance of 5,
so the default is its square root and the filter sees the same $R$; setting
`baro_noise_std_m = 5.0` is five times looser, not the old behaviour.

None of these three has a command-line flag; they are configurable from a file only.

## `[closed_loop]`

Read when `mode = "closed-loop"`. Every key is optional; an omitted key takes the default below,
and an omitted section is the same as an empty one.

| Key | Default | Meaning |
|---|---|---|
| `filter` | `"eskf"` | `"eskf"`, `"ukf"` or `"ekf"` |
| `ukf_alpha` | `0.1` | UKF sigma-point spread |
| `ukf_beta` | `2.0` | UKF prior-distribution parameter |
| `ukf_kappa` | `0.0` | UKF secondary spread parameter |
| `estimate_baro_bias` | `true` | carry a barometric bias state; as the inverse of `--no-estimate-baro-bias` |
| `innovation_gate` | none (accept everything) | reject measurements by NIS; see below |
| `gate_recovery` | inflation 2.0, force after 5 | how the filter recovers from rejections; see below |

The UKF parameters are ignored by the other two filters. `ukf_alpha` is `0.1` rather than the
textbook `1e-3` for a numerical reason: at `1e-3` the weighted sigma-point mean cancels away six
significant digits on every step (see `ClosedLoopConfig::ukf_alpha` in the API documentation).
The `--ukf-alpha` flag has the same default, so a file and the command line run the same filter.

### The barometric bias

A barometer does not read altitude, it reads pressure, and the reference pressure it is converted
against drifts -- roughly a hectopascal an hour, about 8.3 m. A filter that models the reading as
unbiased has nowhere to put that drift except into altitude. `estimate_baro_bias` gives the
filter a state for it, and the output carries it as `baro_bias` and `baro_bias_cov`; with the
state off both columns are empty, which is how a reader tells "no bias state" from "bias
estimated at zero".

It is **on by default** as of the v1.0 freeze, so the key only needs writing to turn it off. The
measurement that justified the default -- improved vertical error and vertical three-sigma
containment on the reference recording, in all three filters -- is recorded on
`ClosedLoopConfig::estimate_baro_bias` in the API documentation. The known case where it hurts is
an AR(1)-degraded GNSS fault, where the bias state absorbs the GNSS fault instead of the
barometer's drift (issue #410); turn it off for such a run.

### `innovation_gate`

Absent, every measurement is accepted. Present, it takes one of two forms:

- `chi_squared` with a `confidence` strictly between 0 and 1: the threshold is the $\chi^2$
  quantile at that probability, for each measurement's own degrees of freedom. Prefer this one.
- `fixed` with a `threshold`: one NIS limit regardless of the measurement's dimension.

This is the one key whose spelling depends on the format. TOML and JSON nest it as a map; YAML
needs a tag:

```toml
[closed_loop]
innovation_gate = { chi_squared = { confidence = 0.999 } }
# or: innovation_gate = { fixed = { threshold = 25.0 } }
```

```yaml
closed_loop:
  innovation_gate: !chi_squared
    confidence: 0.999
  # or: innovation_gate: !fixed { threshold: 25.0 }
```

```json
{ "closed_loop": { "innovation_gate": { "chi_squared": { "confidence": 0.999 } } } }
```

The untagged YAML form `innovation_gate: { chi_squared: { confidence: 0.999 } }` is rejected with
``invalid type: map, expected a YAML tag starting with '!'``.

### `gate_recovery`

Consulted only when a gate is installed. Either field may be omitted and keeps its default.

| Key | Default | Meaning |
|---|---|---|
| `rejection_inflation` | `2.0` | each rejection multiplies the filter's uncertainty by this factor in the directions the rejected measurement observed; `1.0` disables it |
| `forced_update_after` | `5` | when this many measurements in a row fail the gate, apply the last of them regardless (with `5`: four rejected, the fifth forced); `0` disables it (YAML may also write `null`) |

These values are checked before a run starts, exactly as the command-line flags
`--gate-confidence`, `--gate-inflation` and `--gate-force-after` are: a confidence outside (0, 1),
a fixed threshold that is not positive, an inflation below 1, or a `forced_update_after` of 1 is
refused with an error naming `[closed_loop]`.

## `[particle_filter]`

Read when `mode = "particle-filter"`. The keys mirror the `pf` flags; the
[Particle Filter](./particle-filter.md) page explains them and
[Rao-Blackwellized Particle Filter](../filters/rbpf.md) the model behind them.

| Key | Default | Unit |
|---|---|---|
| `num_particles` | `100` | -- |
| `position_init_std_m` | `[10.0, 10.0, 5.0]` | m, north/east/up |
| `velocity_init_std_mps` | `1.0` | m/s |
| `attitude_init_std_rad` | `0.1` | rad |
| `velocity_process_noise_std_mps` | `0.001` | m/s per √s |
| `attitude_process_noise_std_rad` | `0.01` | rad per √s |
| `horizontal_process_noise_std_m` | `[1.0, 1.0]` | m/√s, north/east |
| `baro_loop_time_constant_s` | `10.0` | s |
| `baro_error_std_m` | `8.3` | m |
| `baro_error_time_constant_s` | `3600.0` | s |
| `vertical_accel_error_init_std_mps2` | `0.1` | m/s² |
| `effective_sample_threshold` | `1.0` | fraction of the particle count |
| `roughening_factor` | `0.2` | -- |
| `gravity_variation_std` | derived | mGal |
| `gravity_variation_time_constant_s` | `300.0` | s |
| `magnetic_variation_std` | derived | nT |
| `magnetic_variation_time_constant_s` | `300.0` | s |

- `horizontal_process_noise_std_m` defaults to 1 m/√s per axis, **not** the zero of Canciani &
  Raquet's eq. 19 that the library's `RbpfConfig` keeps: zero diverges with GNSS-rate fixes on
  MEMS data. It must hold exactly two values; any other length is refused.
- `position_init_std_m` must hold three values. A list of any other length is silently replaced
  by the library default, which is also `[10, 10, 5]`.
- The particle seed is the top-level `seed`, not a key of this section.

Four keys from before the restructure are refused by name, with a message saying where the
setting went: `geo_bias_init_std` and `geo_bias_process_noise_std` (now per channel in
`[geophysical]`), `position_process_noise_std_m` (now `horizontal_process_noise_std_m`), and
`zero_vertical_velocity` / `zero_vertical_velocity_std_mps` (now the barometer loop).

## `[geophysical]`

Present, it turns on map-aided navigation for `closed-loop` and `particle-filter` runs. It needs a
binary built with `--features geonav`; without that feature a file carrying the section is
refused rather than run without the maps. In closed-loop mode the filter must be `ukf` or `ekf` --
the ESKF has no geophysical implementation, and since it is the default, a file that omits
`filter` is refused too. See [Geophysical Navigation](../geonav/overview.md).

| Key | Unit | Meaning |
|---|---|---|
| `gravity_resolution` | -- | gravity map resolution; absent, gravity is not used |
| `gravity_map_file` | -- | gravity map path; absent, `<input stem>_gravity.nc` beside the input file |
| `gravity_noise_std` | mGal | measurement noise |
| `gravity_bias` | mGal | initial value of the gravity map-bias state |
| `gravity_bias_init_std` | mGal | its initial standard deviation; defaults to `gravity_noise_std` |
| `gravity_bias_process_noise_std` | mGal/√s | its random-walk rate; defaults to the prior spread over an hour |
| `magnetic_resolution`, `magnetic_map_file`, `magnetic_noise_std`, `magnetic_bias`, `magnetic_bias_init_std`, `magnetic_bias_process_noise_std` | nT | the same for the magnetic map; the default map path is `<input stem>_magnetic.nc` |
| `geo_interval_s` | s | seconds **between** geophysical measurements, for both maps |

Resolutions are written in snake_case in a file (`"one_minute"`, `"thirty_seconds"`) and in
kebab-case on the command line (`one-minute`). `geo_interval_s` is a period, not a rate: a larger
value means fewer measurements. Its old name, `geo_frequency_s`, is still accepted as an alias.

## `[synthetic]`

Read when `mode = "synthetic"`, and the only section that mode reads besides `[logging]`. `output` and `duration_s`
are required; the rest mirror the `syn` flags, with the IMU grade spelled `imu_quality` and the
initial state in a `[synthetic.initial_state]` table. The full key list and a worked example are
on [Synthetic Trajectories](./synthetic.md#the-synthetic-configuration-section).

## `[execution_limits]` and `[health_limits]`

The wall-clock and divergence limits of [Running Simulations](./simulations.md#execution-limits),
with the same defaults. Each key may be given alone.

| Section | Key | Default |
|---|---|---|
| `[execution_limits]` | `max_wall_clock_ratio` | `0.25` |
| | `max_wall_clock_s` | `1200.0` |
| | `max_no_progress_s` | `600.0` |
| `[health_limits]` | `lat_rad` | `[-π/2, π/2]` |
| | `lon_rad` | `[-π, π]` |
| | `alt_m` | `[-1e8, 1e8]` |
| | `speed_mps_max` | `500.0` |
| | `cov_diag_max` | `1e15` |
| | `nis_pos_max` | `100.0` |
| | `nis_pos_consec_fail` | `20` |

The latitude and longitude bands are **radians** here, as `[min, max]` pairs, where the command
line takes degrees (`--health-lat-min-deg`).

## `[logging]`

| Key | Default | Meaning |
|---|---|---|
| `level` | `"info"` | `off`, `error`, `warn`, `info`, `debug` or `trace` |
| `file` | none (stderr) | log file path, appended to |

`--log-file` on the command line overrides `file`. `--log-level` overrides `level` only when it
is something other than `info`: the flag's default is `info`, so `--log-level info` cannot be told
apart from not passing the flag, and the file's level wins. See [Logging](./logging.md).

## Seeds

A configuration file has up to three seeds:

| Key | Default | Seeds |
|---|---|---|
| `seed` (top level) | `42` | the particle filter's sampling, and the GNSS fault models unless `aiding.seed` is set |
| `aiding.seed` | the top-level `seed` | the GNSS fault models |
| `synthetic.seed` | `42` | the generated trajectory's noise |

So the top-level `seed` alone seeds the whole run, as `--seed` does on the command line: two
closed-loop files that differ only in it give two different fault realizations. Set
`aiding.seed` as well to separate the two, so that a particle-filter Monte Carlo study can vary
the filter's seed while holding the GNSS corruption fixed, or the reverse.

The same seeds with the same file and input produce identical output, which is what makes a
reported error meaningful.
