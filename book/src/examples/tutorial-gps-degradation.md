# Tutorial: GPS Degradation

The runnable source is [`core/examples/gnss_outage.rs`](https://github.com/jbrodovsky/strapdown-rs/blob/main/core/examples/gnss_outage.rs):

```bash
cargo run -p strapdown-core --example gnss_outage
```

The scenario is the one every GNSS-denied study starts from: 60 s tracking normally, 120 s
with no fixes at all (a tunnel, an urban canyon, jamming), then aiding returns for 120 s. The
vehicle carries a 0.05 m/s² accelerometer bias on its forward axis, representative of a
consumer MEMS part after turn-on.

## The shape of the error

```text
   t (s)     error (m)        GNSS
      60           0.0          ok
      70           0.2      denied
     100           2.0      denied
     140          14.9      denied
     180          49.4      denied
     190           0.0          ok
```

Flat while aided, growing while coasting, and back to zero within a single fix of recovery.
The growth is quadratic because a constant acceleration error integrates twice into position.

## Why the drift is smaller than the textbook figure

Double-integrating the bias over the outage predicts 360 m:

```text
0.5 × 0.05 m/s² × (120 s)² = 360 m
```

The observed error is 49.4 m, seven times smaller. The reason is worth reading off the filter
state rather than guessing at, and the example prints it:

```text
accelerometer bias present:        0.0500 m/s^2 (forward axis)
bias the filter estimated:         0.0001 m/s^2  <- it barely moved
pitch error at outage start:       0.2987 deg
gravity that tilt leaks forward:   0.0511 m/s^2  <- nearly cancels the bias
```

The filter did **not** estimate the bias. It absorbed it into *attitude* instead: it believes
the vehicle is pitched nose-up by three tenths of a degree, and the gravity that tilt leaks
into the forward axis very nearly cancels the bias.

That is not a defect. A constant forward specific-force bias and a small pitch error produce
the same horizontal acceleration signature, so position and velocity aiding cannot separate
them — the pair is unobservable from this measurement set alone. The filter found an
explanation consistent with every fix it was given, and that explanation happens to coast
well.

The practical lesson is not the quadratic. It is that **which of the two the filter blames is
arbitrary, and it changes how well the solution coasts**. Aiding that does separate them — a
zero-velocity update, a second antenna, a magnetometer for heading — is what makes
dead-reckoning performance predictable rather than lucky.

## Turning the knobs

Two constants at the top of the example are worth changing:

- `OUTAGE_DURATION_S` — drift grows with the *square* of this, so doubling it roughly
  quadruples the peak error.
- `ACCEL_BIAS_MPS2` — set it to zero and the vehicle coasts almost perfectly. Note that the
  drift does not scale with it directly, for the reason above.

## The same scenario from a config file

To run this through the CLI instead of in code, see
[Example Configurations](./configurations.md); `simple_dropout.yaml` and
`extended_gnss_denied.yaml` describe outages of 30 s and 5 minutes respectively.
