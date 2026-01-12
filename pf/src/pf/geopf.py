"""
Geophysical particle filter implementation using velocity-based propagation.

This module implements a velocity-based particle filter for geophysical navigation.
It uses velocity estimates from an INS/GNSS system to propagate particles forward
in time, and updates particle weights using geophysical anomaly measurements
(gravity and/or magnetic) matched against pre-loaded maps.

The filter provides open-loop position estimates through geophysical map-matching,
serving as an alternative positioning sensor for GNSS-denied scenarios.

Data Sources:
    1) INS output: Navigation solution containing position, velocity, and attitude
       Required columns: timestamp, latitude, longitude, altitude,
       velocity_north, velocity_east, velocity_vertical
    2) IMU/GNSS output: Raw sensor data with geophysical measurements
       Required columns: time, latitude, longitude, altitude, speed, bearing,
       grav_x, grav_y, grav_z, mag_x, mag_y, mag_z

Reference:
    Geophysical anomaly calculations follow the implementation in geonav/src/lib.rs.
    Gravity anomaly uses Eötvös correction; magnetic uses World Magnetic Model (WMM).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import pandas as pd
import xarray as xr
from filterpy.monte_carlo import residual_resample, stratified_resample, systematic_resample
from haversine import Unit, haversine
from numpy.typing import NDArray

# ============================================================================
# Constants
# ============================================================================

# WGS84 Earth ellipsoid constants
EQUATORIAL_RADIUS: float = 6_378_137.0  # meters
POLAR_RADIUS: float = 6_356_752.314_245  # meters
FLATTENING: float = 1.0 / 298.257_223_563
ECCENTRICITY_SQUARED: float = 2 * FLATTENING - FLATTENING**2

# Gravity model constants (Somigliana formula)
GRAVITY_EQUATOR: float = 9.780_325_335_903  # m/s^2
GRAVITY_POLE: float = 9.832_184_937_863  # m/s^2
GP: float = 9.806_65  # Standard gravity (m/s^2)

# Earth rotation rate (rad/s)
EARTH_ROTATION_RATE: float = 7.292_115e-5

# Conversion factors
DEG_TO_RAD: float = np.pi / 180.0
RAD_TO_DEG: float = 180.0 / np.pi


# ============================================================================
# Enums and Configuration
# ============================================================================


class ResamplingStrategy(Enum):
    """Available resampling strategies for the particle filter."""

    SYSTEMATIC = "systematic"
    STRATIFIED = "stratified"
    RESIDUAL = "residual"


class AveragingStrategy(Enum):
    """Available averaging strategies for state estimation."""

    MEAN = "mean"
    WEIGHTED_MEAN = "weighted_mean"
    MAP = "map"  # Maximum a posteriori (highest weight particle)


class GeophysicalMeasurementType(Enum):
    """Types of geophysical measurements supported."""

    GRAVITY = "gravity"
    MAGNETIC = "magnetic"
    COMBINED = "combined"


@dataclass
class ParticleFilterConfig:
    """Configuration parameters for the geophysical particle filter.

    Attributes:
        num_particles: Number of particles in the filter.
        measurement_type: Type of geophysical measurement to use.
        resampling_strategy: Strategy for particle resampling.
        averaging_strategy: Strategy for computing state estimates.
        gravity_noise_std: Measurement noise std for gravity (mGal).
        magnetic_noise_std: Measurement noise std for magnetic (nT).
        position_jitter_std: Position jitter std for prediction (degrees).
        altitude_jitter_std: Altitude jitter std for prediction (meters).
        effective_sample_threshold: Threshold ratio for resampling trigger.
        seed: Random seed for reproducibility.
    """

    num_particles: int = 1000
    measurement_type: GeophysicalMeasurementType = GeophysicalMeasurementType.GRAVITY
    resampling_strategy: ResamplingStrategy = ResamplingStrategy.SYSTEMATIC
    averaging_strategy: AveragingStrategy = AveragingStrategy.WEIGHTED_MEAN
    gravity_noise_std: float = 100.0  # mGal
    magnetic_noise_std: float = 150.0  # nT
    position_jitter_std: float = 0.0001  # degrees (~11m)
    altitude_jitter_std: float = 5.0  # meters
    effective_sample_threshold: float = 0.5
    seed: int | None = None


@dataclass
class ParticleState:
    """Container for particle filter state.

    Attributes:
        particles: (N, 3) array of particle positions [lat, lon, alt] in degrees/meters.
        weights: (N,) array of normalized particle weights.
        rng: Random number generator for reproducibility.
    """

    particles: NDArray[np.float64]
    weights: NDArray[np.float64]
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng())


@dataclass
class NavigationState:
    """Navigation state from INS solution.

    Attributes:
        timestamp: Unix timestamp or datetime.
        latitude: Latitude in degrees.
        longitude: Longitude in degrees.
        altitude: Altitude in meters.
        velocity_north: Northward velocity in m/s.
        velocity_east: Eastward velocity in m/s.
        velocity_vertical: Vertical velocity in m/s (positive up).
    """

    timestamp: float
    latitude: float
    longitude: float
    altitude: float
    velocity_north: float
    velocity_east: float
    velocity_vertical: float


@dataclass
class GeophysicalMeasurement:
    """Geophysical measurement from IMU sensors.

    Attributes:
        timestamp: Unix timestamp or datetime.
        gravity_x: Gravity x-component (m/s^2).
        gravity_y: Gravity y-component (m/s^2).
        gravity_z: Gravity z-component (m/s^2).
        magnetic_x: Magnetic field x-component (uT).
        magnetic_y: Magnetic field y-component (uT).
        magnetic_z: Magnetic field z-component (uT).
    """

    timestamp: float
    gravity_x: float = np.nan
    gravity_y: float = np.nan
    gravity_z: float = np.nan
    magnetic_x: float = np.nan
    magnetic_y: float = np.nan
    magnetic_z: float = np.nan

    @property
    def gravity_magnitude(self) -> float:
        """Compute scalar gravity magnitude."""
        return float(np.sqrt(self.gravity_x**2 + self.gravity_y**2 + self.gravity_z**2))

    @property
    def magnetic_magnitude(self) -> float:
        """Compute scalar magnetic field magnitude."""
        return float(np.sqrt(self.magnetic_x**2 + self.magnetic_y**2 + self.magnetic_z**2))


# ============================================================================
# Earth Model Functions (Vectorized)
# ============================================================================


def principal_radii_vec(
    latitude_rad: NDArray[np.float64], altitude: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute principal radii of curvature for WGS84 ellipsoid (vectorized).

    Args:
        latitude_rad: Array of latitudes in radians.
        altitude: Array of altitudes in meters.

    Returns:
        Tuple of (R_N, R_E) arrays - meridian and transverse radii in meters.
    """
    sin_lat = np.sin(latitude_rad)
    sin_lat_sq = sin_lat**2

    # Meridian radius of curvature (Groves eq. 2.105)
    denom = np.sqrt(1 - ECCENTRICITY_SQUARED * sin_lat_sq)
    r_n = EQUATORIAL_RADIUS * (1 - ECCENTRICITY_SQUARED) / (denom**3)

    # Transverse radius of curvature (Groves eq. 2.106)
    r_e = EQUATORIAL_RADIUS / denom

    return r_n + altitude, r_e + altitude


def somigliana_gravity_vec(latitude_deg: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute normal gravity using Somigliana's formula (vectorized).

    Args:
        latitude_deg: Array of latitudes in degrees.

    Returns:
        Array of normal gravity at sea level in m/s^2.
    """
    sin_lat = np.sin(latitude_deg * DEG_TO_RAD)
    sin_lat_sq = sin_lat**2

    # Somigliana formula (Groves eq. 2.134)
    k = (POLAR_RADIUS * GRAVITY_POLE - EQUATORIAL_RADIUS * GRAVITY_EQUATOR) / (
        EQUATORIAL_RADIUS * GRAVITY_EQUATOR
    )
    numerator = 1 + k * sin_lat_sq
    denominator = np.sqrt(1 - ECCENTRICITY_SQUARED * sin_lat_sq)

    return GRAVITY_EQUATOR * numerator / denominator


def eotvos_correction_vec(
    latitude_deg: NDArray[np.float64],
    altitude: NDArray[np.float64],
    velocity_north: NDArray[np.float64],
    velocity_east: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute Eötvös correction for moving platform (vectorized).

    Args:
        latitude_deg: Array of latitudes in degrees.
        altitude: Array of altitudes in meters.
        velocity_north: Array of northward velocities in m/s.
        velocity_east: Array of eastward velocities in m/s.

    Returns:
        Array of Eötvös corrections in m/s^2.
    """
    lat_rad = latitude_deg * DEG_TO_RAD
    cos_lat = np.cos(lat_rad)

    # Get radii of curvature
    r_n, r_e = principal_radii_vec(lat_rad, altitude)

    # Eötvös correction (Groves eq. 5.18)
    correction = (
        2 * EARTH_ROTATION_RATE * velocity_east * cos_lat
        + velocity_east**2 / r_e
        + velocity_north**2 / r_n
    )

    return correction


def gravity_anomaly_vec(
    latitude_deg: NDArray[np.float64],
    altitude: NDArray[np.float64],
    velocity_north: NDArray[np.float64],
    velocity_east: NDArray[np.float64],
    gravity_observed: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute free-air gravity anomaly (vectorized).

    Args:
        latitude_deg: Array of latitudes in degrees.
        altitude: Array of altitudes in meters.
        velocity_north: Array of northward velocities in m/s.
        velocity_east: Array of eastward velocities in m/s.
        gravity_observed: Array of measured gravity magnitudes in m/s^2.

    Returns:
        Array of gravity anomalies in m/s^2.
    """
    # Normal gravity at sea level
    normal_gravity = somigliana_gravity_vec(latitude_deg)

    # Eötvös correction
    eotvos = eotvos_correction_vec(latitude_deg, altitude, velocity_north, velocity_east)

    return gravity_observed - normal_gravity - eotvos


# Scalar versions for backward compatibility
def principal_radii(latitude_rad: float, altitude: float) -> tuple[float, float]:
    """Compute principal radii of curvature for WGS84 ellipsoid (scalar)."""
    r_n, r_e = principal_radii_vec(np.array([latitude_rad]), np.array([altitude]))
    return float(r_n[0]), float(r_e[0])


def somigliana_gravity(latitude_deg: float) -> float:
    """Compute normal gravity using Somigliana's formula (scalar)."""
    return float(somigliana_gravity_vec(np.array([latitude_deg]))[0])


def gravity_with_altitude(latitude_deg: float, altitude: float) -> float:
    """Compute gravity at a given latitude and altitude (scalar)."""
    g0 = somigliana_gravity(latitude_deg)
    free_air_correction = -3.086e-6 * altitude
    return g0 + free_air_correction


def eotvos_correction(
    latitude_deg: float, altitude: float, velocity_north: float, velocity_east: float
) -> float:
    """Compute Eötvös correction for moving platform (scalar)."""
    return float(
        eotvos_correction_vec(
            np.array([latitude_deg]),
            np.array([altitude]),
            np.array([velocity_north]),
            np.array([velocity_east]),
        )[0]
    )


def gravity_anomaly(
    latitude_deg: float,
    altitude: float,
    velocity_north: float,
    velocity_east: float,
    gravity_observed: float,
) -> float:
    """Compute free-air gravity anomaly (scalar)."""
    return float(
        gravity_anomaly_vec(
            np.array([latitude_deg]),
            np.array([altitude]),
            np.array([velocity_north]),
            np.array([velocity_east]),
            np.array([gravity_observed]),
        )[0]
    )


# ============================================================================
# Geophysical Map Interface using xarray
# ============================================================================


class GeoMap:
    """Container for geophysical anomaly map data using xarray DataArray.

    Provides fast interpolation for map lookups using xarray's interp() method.

    Attributes:
        data: xarray DataArray with 'lat' and 'lon' coordinates.
        map_type: Type of geophysical data in the map.
    """

    def __init__(self, data: xr.DataArray, map_type: GeophysicalMeasurementType) -> None:
        """Initialize GeoMap with an xarray DataArray.

        Args:
            data: xarray DataArray with 'lat' and 'lon' coordinates.
            map_type: Type of geophysical data.
        """
        self._data = data
        self.map_type = map_type

        # Validate coordinates exist
        if "lat" not in data.coords or "lon" not in data.coords:
            msg = "DataArray must have 'lat' and 'lon' coordinates"
            raise ValueError(msg)

    @property
    def data(self) -> xr.DataArray:
        """Return the underlying xarray DataArray."""
        return self._data

    @property
    def latitudes(self) -> NDArray[np.float64]:
        """Return latitude coordinate values as numpy array."""
        return np.array(self._data.lat.values)

    @property
    def longitudes(self) -> NDArray[np.float64]:
        """Return longitude coordinate values as numpy array."""
        return np.array(self._data.lon.values)

    @property
    def lat_bounds(self) -> tuple[float, float]:
        """Return latitude bounds of the map."""
        lats = self.latitudes
        return float(np.min(lats)), float(np.max(lats))

    @property
    def lon_bounds(self) -> tuple[float, float]:
        """Return longitude bounds of the map."""
        lons = self.longitudes
        return float(np.min(lons)), float(np.max(lons))

    def in_bounds(self, lat: float, lon: float) -> bool:
        """Check if a point is within the map bounds."""
        lat_min, lat_max = self.lat_bounds
        lon_min, lon_max = self.lon_bounds
        return lat_min <= lat <= lat_max and lon_min <= lon <= lon_max

    def get_point(self, lat: float, lon: float) -> float:
        """Get interpolated map value at a single point.

        Args:
            lat: Latitude in degrees.
            lon: Longitude in degrees.

        Returns:
            Interpolated map value at the specified point.
        """
        result = self._data.interp(lat=lat, lon=lon, method="linear")
        value = float(result.values)
        return value if np.isfinite(value) else np.nan

    def get_points(
        self, lats: NDArray[np.float64], lons: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Get interpolated map values for multiple points using vectorized xarray interp.

        Args:
            lats: Array of latitudes in degrees.
            lons: Array of longitudes in degrees.

        Returns:
            Array of interpolated map values.
        """
        # Convert to xarray DataArrays for efficient interpolation
        lat_da = xr.DataArray(lats, dims=["points"])
        lon_da = xr.DataArray(lons, dims=["points"])

        # Vectorized interpolation using xarray
        result = self._data.interp(lat=lat_da, lon=lon_da, method="linear")

        return np.array(result.values)

    @classmethod
    def load_gravity(
        cls,
        region: tuple[float, float, float, float] | None = None,
        resolution: str = "01m",
    ) -> GeoMap:
        """Load gravity free-air anomaly map from PyGMT.

        Args:
            region: Optional (west, east, south, north) bounds in degrees.
            resolution: Resolution string (default: "01m" = 1 arc-minute).

        Returns:
            GeoMap instance with gravity anomaly data.
        """
        import pygmt

        data = pygmt.datasets.load_earth_free_air_anomaly(resolution=resolution, region=region)

        return cls(data=data, map_type=GeophysicalMeasurementType.GRAVITY)

    @classmethod
    def load_magnetic(
        cls,
        region: tuple[float, float, float, float] | None = None,
        resolution: str = "03m",
        data_source: str = "wdmam",
    ) -> GeoMap:
        """Load magnetic anomaly map from PyGMT.

        Args:
            region: Optional (west, east, south, north) bounds in degrees.
            resolution: Resolution string (default: "03m" = 3 arc-minutes).
            data_source: Data source (default: "wdmam" = World Digital Magnetic Anomaly Map).

        Returns:
            GeoMap instance with magnetic anomaly data.
        """
        import pygmt

        data = pygmt.datasets.load_earth_magnetic_anomaly(
            resolution=resolution, region=region, data_source=data_source
        )

        return cls(data=data, map_type=GeophysicalMeasurementType.MAGNETIC)

    @classmethod
    def from_dataarray(cls, data: xr.DataArray, map_type: GeophysicalMeasurementType) -> GeoMap:
        """Create GeoMap from an existing xarray DataArray.

        Args:
            data: xarray DataArray with 'lat' and 'lon' coordinates.
            map_type: Type of geophysical data.

        Returns:
            GeoMap instance.
        """
        return cls(data=data, map_type=map_type)

    def __repr__(self) -> str:
        """Return string representation."""
        lat_min, lat_max = self.lat_bounds
        lon_min, lon_max = self.lon_bounds
        return (
            f"GeoMap({self.map_type.value}, "
            f"lat=[{lat_min:.2f}, {lat_max:.2f}], "
            f"lon=[{lon_min:.2f}, {lon_max:.2f}], "
            f"shape={self._data.shape})"
        )


# ============================================================================
# Particle Filter Core Functions (Vectorized)
# ============================================================================


def initialize_particles(
    initial_position: tuple[float, float, float],
    num_particles: int,
    position_std: float = 0.001,
    altitude_std: float = 10.0,
    rng: np.random.Generator | None = None,
) -> ParticleState:
    """Initialize particles around an initial position estimate.

    Args:
        initial_position: (latitude, longitude, altitude) in degrees/meters.
        num_particles: Number of particles to create.
        position_std: Standard deviation for position spread (degrees).
        altitude_std: Standard deviation for altitude spread (meters).
        rng: Random number generator for reproducibility.

    Returns:
        Initialized ParticleState.
    """
    if rng is None:
        rng = np.random.default_rng()

    lat0, lon0, alt0 = initial_position

    # Initialize particles with Gaussian spread (vectorized)
    particles = np.column_stack(
        [
            lat0 + rng.normal(0, position_std, num_particles),
            lon0 + rng.normal(0, position_std, num_particles),
            alt0 + rng.normal(0, altitude_std, num_particles),
        ]
    )

    # Initialize uniform weights
    weights = np.ones(num_particles) / num_particles

    return ParticleState(particles=particles, weights=weights, rng=rng)


def predict(
    state: ParticleState,
    velocity_north: float,
    velocity_east: float,
    velocity_vertical: float,
    dt: float,
    position_jitter_std: float = 0.0001,
    altitude_jitter_std: float = 5.0,
) -> ParticleState:
    """Predict particle positions forward in time using velocity inputs (vectorized).

    Uses simplified local-level frame position update equations.

    Args:
        state: Current particle state.
        velocity_north: Northward velocity in m/s.
        velocity_east: Eastward velocity in m/s.
        velocity_vertical: Vertical velocity in m/s (positive up).
        dt: Time step in seconds.
        position_jitter_std: Process noise for position (degrees).
        altitude_jitter_std: Process noise for altitude (meters).

    Returns:
        Updated ParticleState with predicted positions.
    """
    n_particles = len(state.particles)

    # Extract current positions
    lats = state.particles[:, 0]
    lons = state.particles[:, 1]
    alts = state.particles[:, 2]

    # Convert latitude to radians for calculations
    lat_rad = lats * DEG_TO_RAD

    # Get radii of curvature at each particle position (vectorized)
    r_n, r_e = principal_radii_vec(lat_rad, alts)
    cos_lat = np.cos(lat_rad)

    # Compute position deltas (vectorized)
    dlat_deg = (velocity_north * dt / r_n) * RAD_TO_DEG

    # Handle near-pole case
    dlon_deg = np.where(
        np.abs(cos_lat) > 1e-6,
        (velocity_east * dt / (r_e * cos_lat)) * RAD_TO_DEG,
        0.0,
    )

    dalt = velocity_vertical * dt

    # Generate process noise (vectorized)
    lat_noise = state.rng.normal(0, position_jitter_std, n_particles)
    lon_noise = state.rng.normal(0, position_jitter_std, n_particles)
    alt_noise = state.rng.normal(0, altitude_jitter_std, n_particles)

    # Apply position updates with process noise
    new_particles = np.column_stack(
        [
            lats + dlat_deg + lat_noise,
            lons + dlon_deg + lon_noise,
            alts + dalt + alt_noise,
        ]
    )

    return ParticleState(particles=new_particles, weights=state.weights.copy(), rng=state.rng)


def gaussian_likelihood(innovation: NDArray[np.float64], noise_std: float) -> NDArray[np.float64]:
    """Compute Gaussian likelihood (vectorized, numerically stable).

    Args:
        innovation: Array of measurement innovations.
        noise_std: Measurement noise standard deviation.

    Returns:
        Array of likelihood values.
    """
    # Use log-likelihood for numerical stability, then exponentiate
    log_likelihood = -0.5 * (innovation / noise_std) ** 2
    # Subtract max for numerical stability before exponentiating
    log_likelihood_shifted = log_likelihood - np.max(log_likelihood)
    return np.exp(log_likelihood_shifted)


def update_weights_gravity(
    state: ParticleState,
    gravity_map: GeoMap,
    observed_anomaly: float,
    velocity_north: float,
    velocity_east: float,
    noise_std: float = 100.0,
) -> ParticleState:
    """Update particle weights based on gravity anomaly measurement (vectorized).

    Args:
        state: Current particle state.
        gravity_map: Gravity anomaly map for lookups.
        observed_anomaly: Observed gravity anomaly (mGal or m/s^2).
        velocity_north: Northward velocity for Eötvös correction.
        velocity_east: Eastward velocity for Eötvös correction.
        noise_std: Measurement noise standard deviation.

    Returns:
        Updated ParticleState with new weights.
    """
    lats = state.particles[:, 0]
    lons = state.particles[:, 1]

    # Get predicted anomalies from map at all particle positions (vectorized)
    predicted_anomalies = gravity_map.get_points(lats, lons)

    # Compute innovations
    innovations = observed_anomaly - predicted_anomalies

    # Compute likelihoods (handle NaN from out-of-bounds particles)
    likelihoods = gaussian_likelihood(innovations, noise_std)
    likelihoods = np.where(np.isfinite(innovations), likelihoods, 1e-300)

    # Update weights
    new_weights = state.weights * likelihoods

    # Normalize weights
    weight_sum = np.sum(new_weights)
    if weight_sum > 1e-300:
        new_weights /= weight_sum
    else:
        # Weights collapsed - reinitialize to uniform
        new_weights = np.ones(len(state.particles)) / len(state.particles)

    return ParticleState(particles=state.particles.copy(), weights=new_weights, rng=state.rng)


def update_weights_magnetic(
    state: ParticleState,
    magnetic_map: GeoMap,
    observed_anomaly: float,
    noise_std: float = 150.0,
) -> ParticleState:
    """Update particle weights based on magnetic anomaly measurement (vectorized).

    Args:
        state: Current particle state.
        magnetic_map: Magnetic anomaly map for lookups.
        observed_anomaly: Observed magnetic anomaly (nT).
        noise_std: Measurement noise standard deviation.

    Returns:
        Updated ParticleState with new weights.
    """
    lats = state.particles[:, 0]
    lons = state.particles[:, 1]

    # Get predicted anomalies from map at all particle positions (vectorized)
    predicted_anomalies = magnetic_map.get_points(lats, lons)

    # Compute innovations
    innovations = observed_anomaly - predicted_anomalies

    # Compute likelihoods (handle NaN from out-of-bounds particles)
    likelihoods = gaussian_likelihood(innovations, noise_std)
    likelihoods = np.where(np.isfinite(innovations), likelihoods, 1e-300)

    # Update weights
    new_weights = state.weights * likelihoods

    # Normalize weights
    weight_sum = np.sum(new_weights)
    if weight_sum > 1e-300:
        new_weights /= weight_sum
    else:
        new_weights = np.ones(len(state.particles)) / len(state.particles)

    return ParticleState(particles=state.particles.copy(), weights=new_weights, rng=state.rng)


def update_weights_combined(
    state: ParticleState,
    gravity_map: GeoMap | None,
    magnetic_map: GeoMap | None,
    gravity_anomaly_obs: float | None,
    magnetic_anomaly_obs: float | None,
    velocity_north: float,
    velocity_east: float,
    gravity_noise_std: float = 100.0,
    magnetic_noise_std: float = 150.0,
) -> ParticleState:
    """Update particle weights using combined gravity and magnetic measurements.

    Args:
        state: Current particle state.
        gravity_map: Gravity anomaly map (optional).
        magnetic_map: Magnetic anomaly map (optional).
        gravity_anomaly_obs: Observed gravity anomaly (optional).
        magnetic_anomaly_obs: Observed magnetic anomaly (optional).
        velocity_north: Northward velocity for gravity Eötvös correction.
        velocity_east: Eastward velocity for gravity Eötvös correction.
        gravity_noise_std: Gravity measurement noise std.
        magnetic_noise_std: Magnetic measurement noise std.

    Returns:
        Updated ParticleState with new weights.
    """
    current_state = state

    if gravity_map is not None and gravity_anomaly_obs is not None:
        current_state = update_weights_gravity(
            current_state,
            gravity_map,
            gravity_anomaly_obs,
            velocity_north,
            velocity_east,
            gravity_noise_std,
        )

    if magnetic_map is not None and magnetic_anomaly_obs is not None:
        current_state = update_weights_magnetic(
            current_state, magnetic_map, magnetic_anomaly_obs, magnetic_noise_std
        )

    return current_state


def effective_sample_size(weights: NDArray[np.float64]) -> float:
    """Compute the effective sample size of the particle distribution.

    Args:
        weights: Normalized particle weights.

    Returns:
        Effective sample size (1 to N).
    """
    return 1.0 / np.sum(weights**2)


def resample_particles(
    state: ParticleState,
    strategy: ResamplingStrategy = ResamplingStrategy.SYSTEMATIC,
) -> ParticleState:
    """Resample particles based on their weights.

    Args:
        state: Current particle state.
        strategy: Resampling strategy to use.

    Returns:
        Resampled ParticleState with uniform weights.
    """
    n_particles = len(state.particles)

    # Select resampling function
    if strategy == ResamplingStrategy.SYSTEMATIC:
        indices = systematic_resample(state.weights)
    elif strategy == ResamplingStrategy.STRATIFIED:
        indices = stratified_resample(state.weights)
    else:  # RESIDUAL
        indices = residual_resample(state.weights)

    # Resample particles
    new_particles = state.particles[indices]
    new_weights = np.ones(n_particles) / n_particles

    return ParticleState(particles=new_particles, weights=new_weights, rng=state.rng)


def estimate_state(
    state: ParticleState,
    strategy: AveragingStrategy = AveragingStrategy.WEIGHTED_MEAN,
) -> tuple[float, float, float]:
    """Estimate the navigation state from particles.

    Args:
        state: Current particle state.
        strategy: Averaging strategy to use.

    Returns:
        Tuple of (latitude, longitude, altitude) estimates.
    """
    if strategy == AveragingStrategy.MEAN:
        lat = np.mean(state.particles[:, 0])
        lon = np.mean(state.particles[:, 1])
        alt = np.mean(state.particles[:, 2])
    elif strategy == AveragingStrategy.WEIGHTED_MEAN:
        lat = np.dot(state.weights, state.particles[:, 0])
        lon = np.dot(state.weights, state.particles[:, 1])
        alt = np.dot(state.weights, state.particles[:, 2])
    else:  # MAP
        max_idx = np.argmax(state.weights)
        lat = state.particles[max_idx, 0]
        lon = state.particles[max_idx, 1]
        alt = state.particles[max_idx, 2]

    return float(lat), float(lon), float(alt)


def estimate_covariance(state: ParticleState) -> NDArray[np.float64]:
    """Estimate the state covariance from weighted particles (vectorized).

    Args:
        state: Current particle state.

    Returns:
        3x3 covariance matrix for [lat, lon, alt].
    """
    # Weighted mean
    mean = np.array(
        [
            np.dot(state.weights, state.particles[:, 0]),
            np.dot(state.weights, state.particles[:, 1]),
            np.dot(state.weights, state.particles[:, 2]),
        ]
    )

    # Weighted covariance (vectorized using einsum)
    deviations = state.particles - mean  # (N, 3)
    # cov = sum_i w_i * outer(dev_i, dev_i)
    # Using einsum: cov_jk = sum_i w_i * dev_ij * dev_ik
    cov = np.einsum("i,ij,ik->jk", state.weights, deviations, deviations)

    return cov


# ============================================================================
# High-Level Filter Interface
# ============================================================================


class GeophysicalParticleFilter:
    """High-level interface for the geophysical particle filter.

    This class encapsulates the full particle filter workflow including
    initialization, prediction, update, and resampling.
    """

    def __init__(
        self,
        config: ParticleFilterConfig,
        gravity_map: GeoMap | None = None,
        magnetic_map: GeoMap | None = None,
    ) -> None:
        """Initialize the particle filter.

        Args:
            config: Filter configuration parameters.
            gravity_map: Gravity anomaly map for updates (optional).
            magnetic_map: Magnetic anomaly map for updates (optional).
        """
        self.config = config
        self.gravity_map = gravity_map
        self.magnetic_map = magnetic_map
        self.state: ParticleState | None = None
        self.rng = np.random.default_rng(config.seed)

        # Validate configuration
        if config.measurement_type == GeophysicalMeasurementType.GRAVITY and gravity_map is None:
            msg = "Gravity map required for gravity measurement type"
            raise ValueError(msg)
        if config.measurement_type == GeophysicalMeasurementType.MAGNETIC and magnetic_map is None:
            msg = "Magnetic map required for magnetic measurement type"
            raise ValueError(msg)
        if (
            config.measurement_type == GeophysicalMeasurementType.COMBINED
            and gravity_map is None
            and magnetic_map is None
        ):
            msg = "At least one map required for combined measurement type"
            raise ValueError(msg)

    def initialize(
        self,
        initial_position: tuple[float, float, float],
        position_std: float | None = None,
        altitude_std: float | None = None,
    ) -> None:
        """Initialize particles around an initial position.

        Args:
            initial_position: (latitude, longitude, altitude) in degrees/meters.
            position_std: Optional override for position spread.
            altitude_std: Optional override for altitude spread.
        """
        pos_std = position_std if position_std is not None else self.config.position_jitter_std
        alt_std = altitude_std if altitude_std is not None else self.config.altitude_jitter_std

        self.state = initialize_particles(
            initial_position=initial_position,
            num_particles=self.config.num_particles,
            position_std=pos_std * 10,  # Initial spread is larger
            altitude_std=alt_std * 2,
            rng=self.rng,
        )

    def step(
        self,
        nav_state: NavigationState,
        measurement: GeophysicalMeasurement,
        dt: float,
    ) -> tuple[float, float, float]:
        """Run one filter step: predict, update, resample.

        Args:
            nav_state: Current navigation state with velocities.
            measurement: Current geophysical measurement.
            dt: Time step in seconds.

        Returns:
            Estimated (latitude, longitude, altitude).
        """
        if self.state is None:
            msg = "Filter not initialized. Call initialize() first."
            raise RuntimeError(msg)

        # Predict step
        self.state = predict(
            self.state,
            velocity_north=nav_state.velocity_north,
            velocity_east=nav_state.velocity_east,
            velocity_vertical=nav_state.velocity_vertical,
            dt=dt,
            position_jitter_std=self.config.position_jitter_std,
            altitude_jitter_std=self.config.altitude_jitter_std,
        )

        # Update step based on measurement type
        gravity_obs = None
        magnetic_obs = None

        if self.config.measurement_type in (
            GeophysicalMeasurementType.GRAVITY,
            GeophysicalMeasurementType.COMBINED,
        ) and not np.isnan(measurement.gravity_magnitude):
            # Compute gravity anomaly
            gravity_obs = gravity_anomaly(
                nav_state.latitude,
                nav_state.altitude,
                nav_state.velocity_north,
                nav_state.velocity_east,
                measurement.gravity_magnitude,
            )

        if self.config.measurement_type in (
            GeophysicalMeasurementType.MAGNETIC,
            GeophysicalMeasurementType.COMBINED,
        ) and not np.isnan(measurement.magnetic_magnitude):
            # For magnetic, we directly use the magnitude as the observation
            # In practice, you'd subtract the WMM reference field
            magnetic_obs = measurement.magnetic_magnitude

        self.state = update_weights_combined(
            self.state,
            gravity_map=self.gravity_map,
            magnetic_map=self.magnetic_map,
            gravity_anomaly_obs=gravity_obs,
            magnetic_anomaly_obs=magnetic_obs,
            velocity_north=nav_state.velocity_north,
            velocity_east=nav_state.velocity_east,
            gravity_noise_std=self.config.gravity_noise_std,
            magnetic_noise_std=self.config.magnetic_noise_std,
        )

        # Resample if effective sample size is too low
        n_eff = effective_sample_size(self.state.weights)
        threshold = self.config.effective_sample_threshold * self.config.num_particles
        if n_eff < threshold:
            self.state = resample_particles(self.state, self.config.resampling_strategy)

        # Return state estimate
        return estimate_state(self.state, self.config.averaging_strategy)

    def get_state_estimate(self) -> tuple[float, float, float]:
        """Get the current state estimate.

        Returns:
            Estimated (latitude, longitude, altitude).
        """
        if self.state is None:
            msg = "Filter not initialized"
            raise RuntimeError(msg)
        return estimate_state(self.state, self.config.averaging_strategy)

    def get_covariance(self) -> NDArray[np.float64]:
        """Get the current state covariance estimate.

        Returns:
            3x3 covariance matrix.
        """
        if self.state is None:
            msg = "Filter not initialized"
            raise RuntimeError(msg)
        return estimate_covariance(self.state)

    def get_effective_sample_size(self) -> float:
        """Get the current effective sample size.

        Returns:
            Effective sample size.
        """
        if self.state is None:
            msg = "Filter not initialized"
            raise RuntimeError(msg)
        return effective_sample_size(self.state.weights)


# ============================================================================
# Data Processing Utilities
# ============================================================================


def build_pf_dataset(nav_df: pd.DataFrame, imu_df: pd.DataFrame) -> pd.DataFrame:
    """Build a combined dataset for particle filter from navigation and IMU data.

    Synchronizes INS navigation output with IMU/GNSS measurements based on
    timestamps. Uses nearest-neighbor interpolation for alignment.

    Args:
        nav_df: Navigation data with columns: timestamp, latitude, longitude,
                altitude, velocity_north, velocity_east, velocity_vertical
        imu_df: IMU/GNSS data with columns: time, latitude, longitude, altitude,
                speed, bearing, grav_x, grav_y, grav_z, mag_x, mag_y, mag_z

    Returns:
        Merged DataFrame with synchronized navigation and measurement data.
    """
    import pandas as pd

    # Ensure timestamp columns are datetime
    if not pd.api.types.is_datetime64_any_dtype(nav_df["timestamp"]):
        nav_df = nav_df.copy()
        nav_df["timestamp"] = pd.to_datetime(nav_df["timestamp"])

    if not pd.api.types.is_datetime64_any_dtype(imu_df["time"]):
        imu_df = imu_df.copy()
        imu_df["time"] = pd.to_datetime(imu_df["time"])

    # Rename IMU time column for merge
    imu_df = imu_df.rename(columns={"time": "timestamp"})

    # Sort both by timestamp
    nav_df = nav_df.sort_values("timestamp").reset_index(drop=True)
    imu_df = imu_df.sort_values("timestamp").reset_index(drop=True)

    # Use merge_asof for nearest-time matching
    merged = pd.merge_asof(
        nav_df,
        imu_df,
        on="timestamp",
        direction="nearest",
        tolerance=pd.Timedelta("1s"),
        suffixes=("_nav", "_imu"),
    )

    # Use navigation solution positions but keep IMU measurements
    # Rename columns for clarity
    column_mapping = {
        "latitude_nav": "latitude",
        "longitude_nav": "longitude",
        "altitude_nav": "altitude",
    }
    merged = merged.rename(columns=column_mapping)

    # Keep ground truth positions from IMU for comparison
    if "latitude_imu" in merged.columns:
        merged = merged.rename(
            columns={
                "latitude_imu": "latitude_truth",
                "longitude_imu": "longitude_truth",
                "altitude_imu": "altitude_truth",
            }
        )

    return merged


def compute_position_error(
    estimated: tuple[float, float, float],
    truth: tuple[float, float, float],
) -> tuple[float, float, float]:
    """Compute position error between estimated and true positions.

    Args:
        estimated: (lat, lon, alt) estimated position.
        truth: (lat, lon, alt) true position.

    Returns:
        Tuple of (horizontal_error_m, vertical_error_m, 3d_error_m).
    """
    # Horizontal error using Haversine formula
    horiz_error = haversine(
        (estimated[0], estimated[1]),
        (truth[0], truth[1]),
        unit=Unit.METERS,
    )

    # Vertical error
    vert_error = abs(estimated[2] - truth[2])

    # 3D error
    error_3d = np.sqrt(horiz_error**2 + vert_error**2)

    return float(horiz_error), float(vert_error), float(error_3d)
