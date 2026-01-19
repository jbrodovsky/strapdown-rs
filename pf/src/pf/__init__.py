"""
Geophysical Particle Filter (pf)

A Python package for velocity-based particle filter navigation using geophysical
anomaly measurements (gravity and magnetic fields) for map-matching.

This package provides an alternative positioning solution for GNSS-denied scenarios
by propagating particles using velocity estimates from an INS/GNSS system and
updating particle weights based on geophysical map-matching.

Main Components:
    - GeophysicalParticleFilter: High-level filter interface
    - GeoMap: Container for geophysical anomaly map data
    - ParticleFilterConfig: Configuration parameters
    - CLI tools:
        - pf-sim: Run particle filter simulations
        - pf-postprocess: Generate plots and summary statistics

Example:
    >>> from pf import GeophysicalParticleFilter, ParticleFilterConfig, GeoMap
    >>> from pf import GeophysicalMeasurementType
    >>>
    >>> # Load gravity map from PyGMT (1 arc-minute resolution)
    >>> gravity_map = GeoMap.load_gravity(region=(-74, -73, 40, 41))
    >>>
    >>> # Create filter
    >>> config = ParticleFilterConfig(
    ...     num_particles=1000,
    ...     measurement_type=GeophysicalMeasurementType.GRAVITY,
    ...     seed=42
    ... )
    >>> pf = GeophysicalParticleFilter(config, gravity_map=gravity_map)
    >>>
    >>> # Initialize and run
    >>> pf.initialize((40.5, -73.5, 100.0))
    >>> est_lat, est_lon, est_alt = pf.step(nav_state, measurement, dt=1.0)
"""

from pf.geopf import (
    # Constants
    DEG_TO_RAD,
    EARTH_ROTATION_RATE,
    ECCENTRICITY_SQUARED,
    EQUATORIAL_RADIUS,
    FLATTENING,
    GP,
    GRAVITY_EQUATOR,
    GRAVITY_POLE,
    POLAR_RADIUS,
    RAD_TO_DEG,
    # Configuration classes
    AveragingStrategy,
    # Map interface
    GeoMap,
    # State containers
    GeophysicalMeasurement,
    GeophysicalMeasurementType,
    # Main filter class
    GeophysicalParticleFilter,
    NavigationState,
    ParticleFilterConfig,
    ParticleState,
    ResamplingStrategy,
    # Core functions
    build_pf_dataset,
    compute_position_error,
    effective_sample_size,
    # Earth model functions
    eotvos_correction,
    estimate_covariance,
    estimate_state,
    gravity_anomaly,
    gravity_with_altitude,
    initialize_particles,
    predict,
    principal_radii,
    resample_particles,
    somigliana_gravity,
    update_weights_combined,
    update_weights_gravity,
    update_weights_magnetic,
)

__version__ = "0.1.0"
__author__ = "James Brodovsky"

__all__ = [
    # Version info
    "__version__",
    "__author__",
    # Configuration enums
    "AveragingStrategy",
    "GeophysicalMeasurementType",
    "ResamplingStrategy",
    # Configuration classes
    "ParticleFilterConfig",
    # State containers
    "GeophysicalMeasurement",
    "NavigationState",
    "ParticleState",
    # Map interface
    "GeoMap",
    # Main filter class
    "GeophysicalParticleFilter",
    # Core functions
    "build_pf_dataset",
    "compute_position_error",
    "effective_sample_size",
    "estimate_covariance",
    "estimate_state",
    "gravity_anomaly",
    "initialize_particles",
    "predict",
    "resample_particles",
    "update_weights_combined",
    "update_weights_gravity",
    "update_weights_magnetic",
    # Earth model functions
    "eotvos_correction",
    "gravity_with_altitude",
    "principal_radii",
    "somigliana_gravity",
    # Constants
    "DEG_TO_RAD",
    "EARTH_ROTATION_RATE",
    "ECCENTRICITY_SQUARED",
    "EQUATORIAL_RADIUS",
    "FLATTENING",
    "GP",
    "GRAVITY_EQUATOR",
    "GRAVITY_POLE",
    "POLAR_RADIUS",
    "RAD_TO_DEG",
    # Post-processing
    "PerformanceStats",
    "compute_performance_stats",
    "plot_pf_performance",
    "plot_pf_vs_ins",
    "plot_trajectory_map",
    "run_postprocessing",
]

# Import post-processing functions
from pf.postprocess import (
    PerformanceStats,
    compute_performance_stats,
    plot_pf_performance,
    plot_pf_vs_ins,
    plot_trajectory_map,
    run_postprocessing,
)
