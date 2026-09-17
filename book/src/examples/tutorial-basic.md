# Tutorial: Basic INS Simulation

The runnable source for this tutorial is [`core/examples/basic_ins.rs`](https://github.com/jbrodovsky/strapdown-rs/blob/main/core/examples/basic_ins.rs):

```bash
cargo run -p strapdown-core --example basic_ins
```

It drives a vehicle due north at a constant 10 m/s on a level road for 60 seconds, with a
GNSS fix once a second. The trajectory is deliberately trivial so that every number it prints
can be checked by hand.

## The loop

Every application built on this crate has the same shape, whether the samples come from a
file, a hardware driver, or a generator:

1. Build an engine from an initial state.
2. For each IMU sample, `predict`.
3. When a GNSS fix arrives, `update_gnss`.
4. Read `nav_solution` whenever the current estimate is needed.

```rust,ignore
let mut engine = InsEngine::builder()
    // `InsEngineConfig` is `#[non_exhaustive]`, so neither a struct literal nor `..default()`
    // compiles outside `strapdown-core`. Start from `default()` and assign.
    .with_config({
        let mut config = InsEngineConfig::default();
        config.is_enu = false;
        config
    })
    .with_initial_state(initial_state)
    .build()?;

engine.predict(&ImuSample::from_rates(&imu, dt))?;
let outcome = engine.update_gnss(&fix)?;
let solution = engine.nav_solution();
```

`update_gnss` returns an [`UpdateOutcome`] rather than a bare `Ok(())`. It carries the
normalized innovation squared and whether the correction was applied. With no innovation gate
installed every fix is accepted; once you call `set_innovation_gate`, a rejected fix is
reported through this value rather than raised as an error, because a rejection is a normal
event and not a fault.

A gate also needs a way back, and the engine installs one by default. Rejecting a fix leaves
the state where it was, but the filter keeps propagating and keeps accumulating error -- so
without a recovery path the next fix disagrees by *more*, is rejected harder, and one
rejection quietly turns into dead reckoning for the rest of the run. `GateRecovery` is what
prevents that: every rejection inflates the filter's uncertainty in the directions that
measurement observed -- so the next fix of the same kind is judged against a covariance that
grew -- and after five consecutive rejections a measurement is applied regardless, on the
reasoning that a belief contradicted five times running is likelier to be wrong than the sensor
contradicting it. `set_gate_recovery`
tunes it, `GateRecovery::none()` switches it off, and `UpdateOutcome::forced` says whether an
accepted fix was accepted on its merits or by that escape.

## The sign convention that catches everyone

An accelerometer measures **specific force**, not acceleration. A vehicle sitting still is not
in free fall: the ground pushes up on it, and that is what the sensor reads. In NED the
vertical axis points *down*, so a level, stationary IMU reads:

```text
accel = [0.0, 0.0, -9.81]   // m/s², body frame, NED
```

**Negative**, on the down axis. It has to be for the mechanization to cancel it: the velocity
update adds gravity (positive down in NED) to the sensed increment, and a stationary vehicle's
velocity must not change.

In ENU the same physical reading is `+9.81` on the up axis. That is what the Sensor Logger
exports in `core/tests/test_data.csv` look like, which is why that file is loaded as ENU.

Getting this backwards does not fail loudly. The filter concludes the vehicle is upside down:
roll converges to 180°, and the vertical channel walks away even while GNSS is holding
altitude fixed. If a first integration produces those two symptoms together, check this first.

## Expected output

```text
start: NavSolution(t: 0.000 s, 39.9500000 deg, -75.1600000 deg, 12.00 m, ...)
after 60 s: NavSolution(t: 60.000 s, 39.9553996 deg, -75.1600000 deg, 12.14 m, ...)
travelled 600.0 m north; expected 600.0 m (difference 0.00 m)
position uncertainty (1-sigma): 3.01 m north, 2.32 m east, 1.08 m vertical
```

600 m in 60 s at 10 m/s, and the reported north uncertainty settles near the 3 m accuracy the
synthetic fixes advertise, which is what a correctly-tuned filter should do: with no vehicle
dynamics to exploit it cannot do better than its aiding source.

Next: [Tutorial: GPS Degradation](./tutorial-gps-degradation.md).
