# Example Configurations

Ready-to-run scenario files live in
[`examples/configs/`](https://github.com/jbrodovsky/strapdown-rs/tree/main/examples/configs).
For the schema itself — every field, and the naming pitfall that silently disables a scenario
— see [Configuration Files](../user-guide/configuration.md).

```bash
strapdown-sim -i data/input.csv -o results/output.csv closed-loop \
  --config examples/configs/simple_dropout.yaml
```

## Baseline

| File | What it does |
|---|---|
| `baseline.yaml` | No degradation. The reference every other scenario is compared against. |

## Availability

| File | What it does |
|---|---|
| `sched_10s.yaml` | One fix every 10 s, uncorrupted. A reduced update rate. |
| `duty_10on_2off.yaml` | 10 s available, 2 s denied, repeating. Brief periodic outages. |
| `simple_dropout.yaml` | 2 minutes available, 30 s denied. Bridges and tall buildings. |
| `extended_gnss_denied.yaml` | 1 minute available, 5 minutes denied. Tunnels and sustained jamming. |

`extended_gnss_denied.yaml` is the one that shows dead-reckoning quality most clearly: five
minutes is long enough for consumer-MEMS drift to dominate completely. Expect errors in the
hundreds of metres to kilometres, and see
[Tutorial: GPS Degradation](./tutorial-gps-degradation.md) for why the figure is usually
smaller than double-integrating the sensor's bias would predict.

## Accuracy

| File | What it does |
|---|---|
| `degraded_fullrate.yaml` | Full rate, AR(1)-correlated position and velocity error. |
| `degraded_5s.yaml` | Every 5 s *and* degraded. Scheduling and corruption together. |

## Spoofing

| File | What it does |
|---|---|
| `slowbias.yaml` | A slowly drifting offset. Soft spoofing — each fix looks plausible. |
| `slowbias_rot.yaml` | The same, with the drift direction rotating. |
| `hijack.yaml` | An abrupt offset during a fixed window. Hard spoofing. |

Spoofing scenarios are the ones worth pairing with an innovation gate: a `hijack` produces a
large normalized innovation squared the moment it starts, which a chi-squared gate rejects,
while a `slow_bias` is specifically designed to stay under that threshold.

## Combined

| File | What it does |
|---|---|
| `combo.yaml` | Reduced rate plus degraded accuracy. |
| `combo_duty_hijack.yaml` | Duty-cycled availability plus hard spoofing. |

## Other formats

`gnss_degradation.json` and `gnss_degradation.toml` are the same schema in the other two
supported formats. The `json/` subdirectory is a different thing: those files are
`{name, args}` CLI-invocation presets, not scenario configs, and they use the command line's
vocabulary rather than the config schema's.

## Running the set

```bash
for config in examples/configs/*.yaml; do
  name=$(basename "$config" .yaml)
  strapdown-sim -i data/input.csv -o "results/${name}.csv" closed-loop --config "$config"
done
```
