# Publications and Links

## Papers about this software

- **J. Brodovsky**, "strapdown-rs: Strapdown inertial navigation and GNSS-degradation
  simulation in Rust." Submitted to the *Journal of Open Source Software*; under review at
  [openjournals/joss-reviews#11377](https://github.com/openjournals/joss-reviews/issues/11377).
  The source is in `papers/joss/` in the repository. Until it is published, cite the software
  through the repository's `CITATION.cff`.

## Papers that use it

The geophysical-navigation work that `strapdown-geonav` supports (see
[Geophysical Navigation](../geonav/overview.md)):

- J. Brodovsky and P. Dames, "Navigation in GNSS-denied environments using MEMS-grade sensors
  and geophysical anomalies: A UKF approach," in *Proceedings of the 2026 International
  Technical Meeting of the Institute of Navigation*, Anaheim, California, Jan. 2026,
  pp. 155–164. [doi:10.33012/2026.20508](https://doi.org/10.33012/2026.20508)
- J. Brodovsky and P. Dames, "Navigation in GNSS-Denied Environments Using MEMS-Grade Sensors
  and Geophysical Anomalies: A Particle Filter Approach," in *Proceedings of the ION 2026
  Pacific PNT Meeting*, Honolulu, Hawaii, Apr. 2026, pp. 399–410.
  [doi:10.33012/2026.20618](https://doi.org/10.33012/2026.20618)

## References the implementation follows

- P. D. Groves, *Principles of GNSS, Inertial, and Multisensor Integrated Navigation Systems*,
  2nd ed. Boston: Artech House, 2013. ISBN 978-1-60807-005-3. The mechanization, the Earth
  model and the filters cite it by section and equation number throughout.
- A. Canciani and J. Raquet, "Airborne Magnetic Anomaly Navigation," *IEEE Transactions on
  Aerospace and Electronic Systems*, vol. 53, no. 1, pp. 67–80, 2017.
  [doi:10.1109/TAES.2017.2649238](https://doi.org/10.1109/TAES.2017.2649238). The design of
  the Rao-Blackwellized particle filter; see [Rao-Blackwellized Particle Filter](../filters/rbpf.md).
- National Geospatial-Intelligence Agency, *Department of Defense World Geodetic System 1984:
  Its Definition and Relationships with Local Geodetic Systems*, NGA.STND.0036_1.0.0_WGS84,
  2014. The ellipsoid in `strapdown::earth`.
- NOAA NCEI Geomagnetic Modeling Team and British Geological Survey, *World Magnetic Model
  2025*, 2024. [doi:10.25921/aqfd-sd83](https://doi.org/10.25921/aqfd-sd83). The reference
  field for magnetometer heading and magnetic anomalies.

## Links

| | |
|---|---|
| Source repository | <https://github.com/jbrodovsky/strapdown-rs> |
| This book | <https://jbrodovsky.github.io/strapdown-rs/> |
| API documentation | [published with this book](../api/index.md); on docs.rs as [strapdown-core](https://docs.rs/strapdown-core) and [strapdown-geonav](https://docs.rs/strapdown-geonav) once v1.0.0 is on crates.io |
| Issue tracker | <https://github.com/jbrodovsky/strapdown-rs/issues> |
| Sensor Logger (the app the input format comes from) | <https://www.tszheichoi.com/sensorlogger> |
| World Magnetic Model | <https://www.ncei.noaa.gov/products/world-magnetic-model> |
| `world_magnetic_model` crate (the WMM implementation used) | <https://crates.io/crates/world_magnetic_model> |
| `nav-types` crate (geodetic and ECEF types used by `earth`) | <https://github.com/nordmoen/nav-types> |
| GMT remote datasets (the source of the gravity and magnetic maps) | <https://www.generic-mapping-tools.org/remote-datasets/> |
