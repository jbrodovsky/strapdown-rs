# Product Specification

## Free Core (Open-Source / Academic)

The goal of the free core is to provide researchers, students, and developers with a lightweight but **reproducible simulation tool** for GNSS-denied scenarios.

**What it does:**

1. **Simulates a vehicle’s motion** in space (basic kinematics, point-mass model, following waypoints).
2. **Incorporates true data** pulls from a variety of open source real-world datasets (KITTI, nuScenes, Carla, MEMS-Nav, et cetera) that are are hosted by this project's maintainers and have been pre-processed to work with this model (a la PyTorch, TensorFlow, Scikit-learn).
3. **Allows simple GNSS degradation** (e.g., complete outages, increased noise, reduced satellite availability).
4. **Runs from a configuration file** (YAML/JSON) that describes the simulation scenario: path, sensors, noise levels, degradations, filter configuration, and output format.
5. **Exports results** in CSV/Parquet so researchers can analyze them in Python, MATLAB, or R.
6. **Provides a basic converter** to import third-party datasets into the required format.

**User interface (Core):**

* Command-line tool (`strapdown-sim -i test_data.csv -o out/baseline.csv closed-loop ...`) to run simulations from config files and/or command-line args.
* CLI or interactive tools for converting datasets into the required format.
* Python bindings of the core functionality.

**Definition of Done (Core):**

* A researcher can configure a scenario (input data file, INS filter configuration (e.g., noise models, sensor biases, Kalman or Particle filter), GNSS settings), run it, and get reproducible outputs for a navigation solution.
* A researcher can import their own dataset and have it be converted into the required format and run simulations with it.
* Particle filter results should be statistically consistent across runs.
* Results are identical every time if the same random seed is used.

---

## Professional Product (Commercial / Paid)

The pro tier builds on the free core with **more realistic models, automation, visualization tools, and advanced features** for serious users (defense labs, INS companies, autonomy startups).

**What it adds:**

1. **Produces synthetic sensor data** that looks like it came from real-world instruments and a vehicle platform following a trajectory:
   * Includes the following configurable sensors: 
     * **IMU** (accelerometer + gyroscope, with noise and bias errors)
     * **Magnetometer** (magnetic field with noise)
     * **Barometer** (altitude/pressure with noise)
     * **GNSS (basic)** — position/velocity fixes with configurable noise, at a fixed rate.
     * **Sonar** or **Altimeter** (gives depth below keel or altitude above ground)
   * Comes with an accompanying control module and waypoint following system to generate realistic vehicle trajectories (e.g., car, drone, boat).
     * Includes simple vehicle dynamics and waypoint following logic for at least six degrees of freedom (3D position + orientation) and configurable speed/acceleration limits.
     * Supports simulation of environmental effects like wind or water currents that affect the vehicle's motion.
     * Supports more complex noise models (e.g., IMU bias instability, correlated noise).
     * Includes a physics engine for realistic vehicle dynamics and sensor behavior.
2. **Advanced GNSS modeling**:
   * Simulated satellites, pseudoranges, DOP (dilution of precision).
   * Receiver clock error, satellite geometry effects.
   * **GNSS degradation models** includes more advanced and realistic GNSS fault modes:
     * **Spoofing** (pull-off, sudden jumps, false signals).
     * **Jamming** (noise raising error rates until signal is lost).
     * **Multipath** (extra delays due to reflections in urban canyons).
     * **Terrain and urban masking**: Using real DEM (digital elevation map) tiles to determine if satellites are blocked by mountains or buildings.
3. **Batch runner**: run hundreds of scenarios with parameter sweeps (e.g., outage lengths, spoofing rates). Outputs summary reports automatically.
4. **Professional reports**: PDF/HTML with plots, scenario summary, metrics tables, and visualizations.
5. **Graphical interface (Bevy UI)**:
   * Scenario editor (drag/drop parameters, timeline).
   * 3D playback of vehicle trajectories and sensor behavior.
   * Visualization of GNSS outages, spoofing events, error ellipses, etc.
6. **Data adapters**: import/export to ROS bag, RINEX, and standardized nav data formats.

**Definition of Done (Pro):**

* A user can load or design a scenario visually, run batch experiments, and receive a professional report with reproducible results.
* GNSS outages, spoofing, and terrain masking behave as expected and can be tuned by config or UI.
* All the above features are documented with examples and tutorials.
* The UI is user-friendly and stable, with no crashes or major bugs.

---

## Roadmap Pipeline for Advanced Products

This is how the product could evolve over 3–5 years.

### Phase 1 (Year 1–2) – Establish Software Platform

* Free core + pro SDK release.
* Open-source adoption in academic and research labs.
* Pro customers start using for GNSS-denied benchmarking.

### Phase 2 (Year 2–3) – Advanced Pro Features

* **Geophysical navigation modules**:

  * Gravity anomaly aiding (EGM models).
  * Magnetic anomaly aiding (regional grids).
* **Multi-vehicle simulation** (convoy, swarm scenarios).
* **Integration with ML/AI** frameworks (pytorch bindings for anomaly matching).
* **Enterprise collaboration features** (scenario sharing, version control, CI integration).

### Phase 3 (Year 3–4) – Hardware Reference Kit

* Develop a **low-cost INS dev board** (MEMS IMU + baro + mag + GNSS front-end).
* Bundled with SDK for anomaly-aided navigation.
* Sold as a **reference kit** for labs and early adopters.

### Phase 4 (Year 4–5) – Product Diversification

* **Full INS device** (robust, plug-and-play box for drones, ships, ground robots).
* **Enterprise cloud portal** (optional): large-scale scenario sweeps in the cloud, collaboration across teams, dataset marketplace.
* **Custom anomaly packs**: proprietary regional gravity/mag maps for defense/enterprise customers.

---

✅ **Summary**:

* Start with **Core (free)**: deterministic simulator, basic sensors, GNSS outages.
* Build **Pro (paid)**: advanced GNSS modeling, spoofing/jamming, terrain masking, batch runner, UI, pro reports.
* Roadmap toward **anomaly-aided SDK** and eventually a **hardware INS product**.

