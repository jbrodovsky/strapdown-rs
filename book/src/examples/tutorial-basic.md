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
    .with_config(InsEngineConfig { is_enu: false, ..InsEngineConfig::default() })
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
