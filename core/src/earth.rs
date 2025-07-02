//! Earth-related constants and functions
//!
//! This module contains constants and functions related to the Earth's shape and other
//! geophysical features (gravity and magnetic field). The Earth is modeled as an ellipsoid
//! (WGS84) with a semi-major axis and a semi-minor axis. The Earth's gravity is modeled as
//! a function of the latitude and altitude using the Somigliana method. The Earth's
//! rotation rate is also included in this module. This module relies on the `nav-types`
//! crate for the coordinate types and conversions, but provides additional functionality
//! for calculating rotations for the strapdown navigation filters. This permits the
//! transformation of additional quantities (velocity, acceleration, etc.) between the
//! Earth-centered Earth-fixed (ECEF) frame and the local-level frame.
//!
//! # Coordinate Systems
//! The WGS84 ellipsoidal model is the primary model used for the Earth's shape. This crate
//! is primarily concerned with the ECEF and local-level frames, in addition to the basic
//! body frame of the vehicle. The ECEF frame is a right-handed Cartesian coordinate system
//! with the origin at the Earth's center. The local-level frame is a right-handed Cartesian
//! coordinate system with the origin at the sensor's position. The local-level frame is
//! defined by the tangent to the ellipsoidal surface at the sensor's position. The body
//! frame is a right-handed Cartesian coordinate system with the origin at the sensor's
//! center of mass. The body frame is defined by the sensor's orientation.
//!
//! For basic positional conversions, the [`nav-types`](https://crates.io/crates/nav-types)
//! crate is used. This crate provides the `WGS84` and `ECEF` types for representing the
//! Earth's position in geodetic and Cartesian coordinates, respectively. The `nav-types`
//! crate also provides the necessary conversions between the two coordinate systems.
//!
//! # Rotation Functions
//! The rotations needed for the strapdown navigation filters are not directly supported
//! by the `nav-types` crate. These functions provide the necessary rotations that are
//! primarily used for projecting the velocity and acceleration vectors. The rotations
//! are primarily used to convert between the ECEF and local-level frames. The rotations
//! from the local level frame to the body frame can be taken care of by the `nalgebra`
//! crate, which provides the necessary rotation matrices using the Rotation3 type.

// ----------
// Working notes:
// The canonical strapdown navigation state vector is WGS84 geodetic position (latitude, longitude, altitude) and local tangent plane (NED) velocities (north, east,
// down). The state vector is updated by integrating the IMU measurements (body frame) to estimate the position and velocity of the sensor. Velocities are updated
// in NED, whereas the positions are updated in WGS84.
// ----------
// Rotations can be handled using nalgebra's Rotation3 type, which can be converted to a DCM using the into() method. The Rotation3 type can be created from
// Euler angles for the body to local-level frame rotation. The inverse of the Rotation3 type can be used to convert from the local-level frame to the body frame.
// ----------
use crate::{wrap_latitude, wrap_to_180};
use ::nalgebra::{Matrix3, Vector3};
use ::nav_types::{ECEF, WGS84};

// Earth constants (WGS84)
/// Earth's rotation rate rad/s ($\omega_{ie}$)
pub const RATE: f64 = 7.2921159e-5;
/// Earth's rotation rate rad/s ($\omega_{ie}$) in a vector form
pub const RATE_VECTOR: Vector3<f64> = Vector3::new(0.0, 0.0, RATE);
/// Earth's equitorial radius in meters
pub const EQUATORIAL_RADIUS: f64 = 6378137.0; // meters
/// Earth's polar radius in meters
pub const POLAR_RADIUS: f64 = 6356752.31425; // meters
/// Earth's mean radius in meters
pub const MEAN_RADIUS: f64 = 6371000.0; // meters
/// Earth's eccentricity ($e$)
pub const ECCENTRICITY: f64 = 0.0818191908425; // unit-less
/// Earth's eccentricity squared ($e^2$)
pub const ECCENTRICITY_SQUARED: f64 = ECCENTRICITY * ECCENTRICITY;
/// Earth's gravitational acceleration at the equator ($g_e$) in $m/s^2$
pub const GE: f64 = 9.7803253359; // m/s^2, equatorial radius
/// Earth's gravitational acceleration at the poles ($g_p$) in $m/s^2$
pub const GP: f64 = 9.8321849378; // $m/s^2$, polar radius
/// Earth's flattening factor ($f$)
pub const F: f64 = 1.0 / 298.257223563; // Flattening factor
/// Somigliana's constant ($K$)
pub const K: f64 = (POLAR_RADIUS * GP - EQUATORIAL_RADIUS * GE) / (EQUATORIAL_RADIUS * GE); // Somigliana's constant
// Earth magnetic field constants (dipole model)
/// Earth's magnetic north pole latitude, degrees (2025, International Geomagnetic Reference Field)
pub const MAGNETIC_NORTH_LATITUDE: f64 = 80.8; // degrees, geomagnetic north pole latitude  
/// Earth's magnetic north pole longitude, degrees (2025, International Geomagnetic Reference Field)
pub const MAGNETIC_NORTH_LONGITUDE: f64 = -72.8; // degrees, geomagnetic north pole longitude
/// Earth's magnetic reference radius, meters (2025, International Geomagnetic Reference Field)
pub const MAGNETIC_REFERENCE_RADIUS: f64 = 6371200.0; // meters, reference radius for magnetic field calculations
/// Earth's magnetic field strength ($B_0$), teslas (2025, International Geomagnetic Reference Field)
pub const MAGNETIC_FIELD_STRENGTH: f64 = 3.12e-5; // T, reference mean magnetic field strength
/// Rough conversion factor from meters to degrees for latitude/longitude via nautical miles (1 degree ~ 60 nautical miles; 1 nautical mile ~ 1852 meters)
pub const METERS_TO_DEGREES: f64 = 1.0 / (60.0 * 1852.0);
/// Rough conversion factor from degrees to meters for latitude/longitude via nautical miles (1 degree ~ 60 nautical miles; 1 nautical mile ~ 1852 meters)
pub const DEGREES_TO_METERS: f64 = 60.0 * 1852.0;

/// Convert a three-element vector to a skew-symmetric matrix
///
/// Groves' notation uses a lot of skew-symmetric matrices to represent cross products
/// and to perform more concise matrix operations (particularly involving rotations).
/// This function converts a three-element vector to a skew-symmetric matrix.
///
/// $$
/// x = \begin{bmatrix} a \\\\ b \\\\ c \end{bmatrix} \rightarrow X = \begin{bmatrix} 0 & -c & b \\\\ c & 0 & -a \\\\ -b & a & 0 \end{bmatrix}
/// $$
///
/// # Example
/// ```rust
/// use nalgebra::{Vector3, Matrix3};
/// use strapdown::earth;
/// let v: Vector3<f64> = Vector3::new(1.0, 2.0, 3.0);
/// let skew: Matrix3<f64> = earth::vector_to_skew_symmetric(&v);
/// ```
pub fn vector_to_skew_symmetric(v: &Vector3<f64>) -> Matrix3<f64> {
    let mut skew: Matrix3<f64> = Matrix3::zeros();
    skew[(0, 1)] = -v[2];
    skew[(0, 2)] = v[1];
    skew[(1, 0)] = v[2];
    skew[(1, 2)] = -v[0];
    skew[(2, 0)] = -v[1];
    skew[(2, 1)] = v[0];
    skew
}
/// Convert a skew-symmetric matrix to a three-element vector
///
/// This function converts a skew-symmetric matrix to a three-element vector. This is the
/// inverse operation of the `vector_to_skew_symmetric` function.
///
/// $$
/// X = \begin{bmatrix} 0 & -c & b \\\\ c & 0 & -a \\\\ -b & a & 0 \end{bmatrix} \rightarrow x = \begin{bmatrix} a \\\\ b \\\\ c \end{bmatrix}
/// $$
///
/// # Example
/// ```rust
/// use strapdown::earth;
/// use nalgebra::{Vector3, Matrix3};
/// let skew: Matrix3<f64> = Matrix3::new(0.0, -3.0, 2.0, 3.0, 0.0, -1.0, -2.0, 1.0, 0.0);
/// let v: Vector3<f64> = earth::skew_symmetric_to_vector(&skew);
/// ```
pub fn skew_symmetric_to_vector(skew: &Matrix3<f64>) -> Vector3<f64> {
    let mut v: Vector3<f64> = Vector3::zeros();
    v[0] = skew[(2, 1)];
    v[1] = skew[(0, 2)];
    v[2] = skew[(1, 0)];
    v
}
/// Coordinate conversion from the Earth-centered Inertial (ECI) frame to the Earth-centered
/// Earth-fixed (ECEF) frame.
///
/// The ECI frame is a right-handed Cartesian coordinate system with
/// the origin at the Earth's center. The ECEF frame is a right-handed Cartesian coordinate
/// system with the origin at the Earth's center. The ECI frame is fixed with respect to the
/// stars, whereas the ECEF frame rotates with the Earth.
///
/// # Parameters
/// - `time` - The time in seconds that define the rotation period
///
/// # Returns
/// A 3x3 rotation matrix that converts from the ECI frame to the ECEF frame
///
/// # Example
/// ```rust
/// use nalgebra::Matrix3;
/// use strapdown::earth;
/// let time: f64 = 30.0;
/// let rot: Matrix3<f64> = earth::eci_to_ecef(time);
/// ```
pub fn eci_to_ecef(time: f64) -> Matrix3<f64> {
    let mut rot: Matrix3<f64> = Matrix3::zeros();
    rot[(0, 0)] = (RATE * time).cos();
    rot[(0, 1)] = (RATE * time).sin();
    rot[(1, 0)] = -(RATE * time).sin();
    rot[(1, 1)] = (RATE * time).cos();
    rot[(2, 2)] = 1.0;
    rot
}
/// Coordinate conversion from the Earth-centered Earth-fixed (ECEF) frame to the Earth-centered
/// Inertial (ECI) frame.
///
/// The ECI frame is a right-handed Cartesian coordinate system with
/// the origin at the Earth's center. The ECEF frame is a right-handed Cartesian coordinate
/// system with the origin at the Earth's center. The ECI frame is fixed with respect to the
/// stars, whereas the ECEF frame rotates with the Earth.
///     
/// # Parameters
/// - `time` - The time in seconds that define the rotation period
///
/// # Returns
/// A 3x3 rotation matrix that converts from the ECEF frame to the ECI frame
///
/// # Example
/// ```rust
/// use nalgebra::Matrix3;
/// use strapdown::earth;
/// let time: f64 = 30.0;
/// let rot: Matrix3<f64> = earth::ecef_to_eci(time);
/// ```
pub fn ecef_to_eci(time: f64) -> Matrix3<f64> {
    eci_to_ecef(time).transpose()
}
/// Coordinate conversion from the Earth-centered Earth-fixed (ECEF) frame to the local-level frame.
///
/// The ECEF frame is a right-handed Cartesian coordinate system with the origin at the Earth's center.
/// The local-level frame is a right-handed Cartesian coordinate system with the origin at the sensor's
/// position. The local-level frame is defined by the tangent to the ellipsoidal surface at the sensor's
/// position. The local level frame is defined by the WGS84 latitude and longitude.
///
/// # Parameters
/// - `latitude` - The WGS84 latitude in degrees
/// - `longitude` - The WGS84 longitude in degrees
///
/// # Returns
/// A 3x3 rotation matrix that converts from the ECEF frame to the local-level frame
///
/// # Example
/// ```rust
/// use nalgebra::Matrix3;
/// use strapdown::earth;
/// let latitude: f64 = 45.0;
/// let longitude: f64 = 90.0;
/// let rot: Matrix3<f64> = earth::ecef_to_lla(&latitude, &longitude);
/// ```
pub fn ecef_to_lla(latitude: &f64, longitude: &f64) -> Matrix3<f64> {
    let lat: f64 = (*latitude).to_radians();
    let lon: f64 = (*longitude).to_radians();

    let mut rot: Matrix3<f64> = Matrix3::zeros();
    rot[(0, 0)] = -lon.sin() * lat.cos();
    rot[(0, 1)] = -lon.sin() * lat.sin();
    rot[(0, 2)] = lat.cos();
    rot[(1, 0)] = -lon.sin();
    rot[(1, 1)] = lon.cos();
    rot[(2, 0)] = -lat.cos() * lon.cos();
    rot[(2, 1)] = -lat.cos() * lon.sin();
    rot[(2, 2)] = -lat.sin();
    rot
}
/// Coordinate conversion from the local-level frame to the Earth-centered Earth-fixed (ECEF) frame.
///
/// The ECEF frame is a right-handed Cartesian coordinate system with the origin at the Earth's center.
/// The local-level frame is a right-handed Cartesian coordinate system with the origin at the sensor's
/// position. The local-level frame is defined by the tangent to the ellipsoidal surface at the sensor's
/// position. The local level frame is defined by the WGS84 latitude and longitude.
///
/// # Parameters
/// - `latitude` - The WGS84 latitude in degrees
/// - `longitude` - The WGS84 longitude in degrees
///
/// # Returns
/// A 3x3 rotation matrix that converts from the local-level frame to the ECEF frame
///
/// # Example
/// ```rust
/// use nalgebra::Matrix3;
/// use strapdown::earth;
/// let latitude: f64 = 45.0;
/// let longitude: f64 = 90.0;
/// let rot: Matrix3<f64> = earth::lla_to_ecef(&latitude, &longitude);
/// ```
pub fn lla_to_ecef(latitude: &f64, longitude: &f64) -> Matrix3<f64> {
    ecef_to_lla(latitude, longitude).transpose()
}
/// Calculate principal radii of curvature
///
/// The [principal radii of curvature](https://en.wikipedia.org/wiki/Earth_radius) are used to
/// calculate and convert Cartesian body frame quantities to the local-level frame WGS84 coordinates.
///
/// # Parameters
/// - `latitude` - The WGS84 latitude in degrees
/// - `altitude` - The WGS84 altitude in meters
///
/// # Returns
/// A tuple of the principal radii of curvature (r_n, r_e, r_p) in meters where r_n is the radius
/// of curvature in the prime vertical (alternatively as _N_ or R_N), r_e is the radius of curvature
/// in the meridian (alternatively _M_ or R_M), and r_p is the radius of curvature in the local
/// normal direction.
///
/// # Example
/// ```rust
/// use strapdown::earth;
/// let latitude: f64 = 45.0;
/// let altitude: f64 = 1000.0;
/// let (r_n, r_e, r_p) = earth::principal_radii(&latitude, &altitude);
/// ```
pub fn principal_radii(latitude: &f64, altitude: &f64) -> (f64, f64, f64) {
    let latitude_rad: f64 = (latitude).to_radians();
    let sin_lat: f64 = latitude_rad.sin();
    let sin_lat_sq: f64 = sin_lat * sin_lat;
    let r_n: f64 = (EQUATORIAL_RADIUS * (1.0 - ECCENTRICITY_SQUARED))
        / (1.0 - ECCENTRICITY_SQUARED * sin_lat_sq).powf(3.0 / 2.0);
    let r_e: f64 = EQUATORIAL_RADIUS / (1.0 - ECCENTRICITY_SQUARED * sin_lat_sq).sqrt();
    let r_p: f64 = r_e * latitude_rad.cos() + altitude;
    (r_n, r_e, r_p)
}
/// Calculate the WGS84 gravity scalar
///
/// The [gravity model](https://en.wikipedia.org/wiki/Gravity_of_Earth) is based on the [Somigliana
/// method](https://en.wikipedia.org/wiki/Theoretical_gravity#Somigliana_equation), which models
/// the Earth's gravity as a function of the latitude and altitude. The gravity model is used to
/// calculate the gravitational force scalar in the local-level frame. Free-air correction is applied.
///
/// # Parameters
/// - `latitude` - The WGS84 latitude in degrees
/// - `altitude` - The WGS84 altitude in meters
///
/// # Returns
/// The gravitational force scalar in m/s^2
///
/// # Example
/// ```rust
/// use strapdown::earth;
/// let latitude: f64 = 45.0;
/// let altitude: f64 = 1000.0;
/// let grav = earth::gravity(&latitude, &altitude);
/// ```
pub fn gravity(latitude: &f64, altitude: &f64) -> f64 {
    let sin_lat: f64 = (latitude).to_radians().sin();
    let g0: f64 = (GE * (1.0 + K * sin_lat * sin_lat))
        / (1.0 - ECCENTRICITY_SQUARED * sin_lat * sin_lat).sqrt();
    g0 - 3.08e-6 * altitude
}
/// Calculate the gravitational force vector in the local-level frame
///
/// The [gravity model](https://en.wikipedia.org/wiki/Gravity_of_Earth) is based on the [Somigliana
/// method](https://en.wikipedia.org/wiki/Theoretical_gravity#Somigliana_equation), which models
/// the Earth's gravity as a function of the latitude and altitude. The gravity model is used to
/// calculate the gravitational force vector in the local-level frame. This is then combined
/// with the rotational effects of the Earth to calculate the effective gravity vector. This
/// differs from the gravity scalar in that it includes the centrifugal effects of the Earth's
/// rotation.
///
/// *Note:* Local level frame coordintaes are odd and mixed and can be defined as North, East,
/// Down (NED) or East, North, Up (ENU). This function uses the ENU convention, thus gravity acts
/// along the negative Z-axis (downward) in the local-level frame.
///
/// # Parameters
/// - `latitude` - The WGS84 latitude in degrees
/// - `longitude` - The WGS84 longitude in degrees
/// - `altitude` - The WGS84 altitude in meters
///
/// # Returns
/// The gravitational force vector in m/s^2 in the local-level frame
///
/// # Example
/// ```rust
/// use strapdown::earth;
/// let latitude: f64 = 45.0;
/// let longitude: f64 = 90.0;
/// let altitude: f64 = 1000.0;
/// let grav = earth::gravitation(&latitude, &longitude, &altitude);
/// ```
pub fn gravitation(latitude: &f64, longitude: &f64, altitude: &f64) -> Vector3<f64> {
    let latitude = wrap_latitude(*latitude);
    let longitude = wrap_to_180(*longitude);
    let wgs84: WGS84<f64> = WGS84::from_degrees_and_meters(latitude, longitude, *altitude);
    let ecef: ECEF<f64> = ECEF::from(wgs84);
    // Get centrifugal terms in ECEF
    let ecef_vec: Vector3<f64> = Vector3::new(ecef.x(), ecef.y(), ecef.z());
    let omega_ie: Matrix3<f64> = vector_to_skew_symmetric(&RATE_VECTOR);
    // Get rotation and gravity in LLA
    let rot: Matrix3<f64> = ecef_to_lla(&latitude, &longitude);
    let gravity: Vector3<f64> = Vector3::new(0.0, 0.0, gravity(&latitude, altitude));
    // Calculate the effective gravity vector combining gravity and centrifugal terms
    gravity + rot * omega_ie * omega_ie * ecef_vec
}
/// Calculate local gravity anomaly from IMU accelerometer measurements
///
/// This function calculates the local gravity anomaly by comparing the observed gravity from the
/// IMU accelerometer measurements (eg: $\sqrt(a_x^2 + a_y^2 + a_z^2)$) with the normal gravity
/// at the given latitude and altitude via the Somigliana method. Additionally, this function
/// compensates for the motion of the platform (if any) using the Eötvös correction.
///
/// # Parameters
/// - `latitude` - The WGS84 latitude in degrees
/// - `altitude` - The WGS84 altitude in meters
/// - `north_velocity` - The northward velocity component in m/s
/// - `east_velocity` - The eastward velocity component in m/s
/// - `gravity_observed` - The observed gravity from the IMU accelerometer measurements in m/s^2
///
/// # Returns
/// The local gravity anomaly in m/s^2, which is the difference between the observed gravity and the normal gravity at the given latitude and altitude, adjusted for the Eötvös correction.
pub fn gravity_anomaly(
    latitude: &f64,
    altitude: &f64,
    north_velocity: &f64,
    east_velocity: &f64,
    gravity_observed: &f64,
) -> f64 {
    let normal_gravity: f64 = gravity(latitude, &0.0);
    let eotvos_correction: f64 = eotvos(latitude, altitude, north_velocity, east_velocity);
    *gravity_observed - normal_gravity - eotvos_correction
}
/// Calculate the Eotvos correction for the local-level frame
///
/// The Eötvös correction accounts for the centrifugal acceleration caused by the vehicle's motion
/// relative to the Earth's rotation. It depends on the platform's velocity, latitude, and Earth's
/// angular velocity. The correction is generally added to the observed gravity measurement to
/// account for this effect. The formula can be complex, involving latitude, velocity components
/// (East-West), and Earth's rotation rate.
///
/// # Arguments
/// - `latitude` - The WGS84 latitude in radians
/// - `altitude` - The WGS84 altitude in meters
/// - `north_velocity` - The northward velocity component in m/s
/// - `east_velocity` - The eastward velocity component in m/s
///
/// # Returns
/// The Eötvös correction in m/s^2
pub fn eotvos(latitude: &f64, altitude: &f64, north_velocity: &f64, east_velocity: &f64) -> f64 {
    let (_, _, r_p) = principal_radii(latitude, altitude);
    2.0 * RATE * *east_velocity * latitude.cos()
        + (north_velocity.powi(2) + east_velocity.powi(2)) / r_p
}

/// Calculate the Earth rotation rate vector in the local-level frame
///
/// The Earth's rotation rate modeled as a vector in the local-level frame. This vector
/// is used to calculate the Coriolis and centrifugal effects in the local-level frame.
///
/// # Parameters
/// - `latitude` - The WGS84 latitude in degrees
///
/// # Returns
/// The Earth's rotation rate vector in rad/s in the local-level frame
///
/// # Example
/// ```rust
/// use strapdown::earth;
/// let latitude: f64 = 45.0;
/// let omega_ie = earth::earth_rate_lla(&latitude);
/// ```
pub fn earth_rate_lla(latitude: &f64) -> Vector3<f64> {
    let sin_lat: f64 = (latitude).to_radians().sin();
    let cos_lat: f64 = (latitude).to_radians().cos();
    let omega_ie: Vector3<f64> = Vector3::new(RATE * cos_lat, 0.0, -RATE * sin_lat);
    omega_ie
}
/// Calculate the transport rate vector in the local-level frame
///
/// The transport rate is used to calculate the rate of change of the local-level frame
/// with respect to the ECEF frame since the origin point of the local-level frame is
/// always tangential to the WGS84 ellipsoid and thus constantly moving in the ECEF frame.
///
/// # Parameters
/// - `latitude` - The WGS84 latitude in degrees
/// - `altitude` - The WGS84 altitude in meters
/// - `velocities` - The velocity vector in the local-level frame (northward, eastward, downward)
///
/// # Returns
/// The transport rate vector in m/s^2 in the local-level frame
///
/// # Example
/// ```rust
/// use nalgebra::Vector3;
/// use strapdown::earth;
/// let latitude: f64 = 45.0;
/// let altitude: f64 = 1000.0;
/// let velocities: Vector3<f64> = Vector3::new(10.0, 0.0, 0.0);
/// let omega_en_n = earth::transport_rate(&latitude, &altitude, &velocities);
/// ```
pub fn transport_rate(latitude: &f64, altitude: &f64, velocities: &Vector3<f64>) -> Vector3<f64> {
    let (r_n, r_e, _) = principal_radii(latitude, altitude);
    let lat_rad = latitude.to_radians();
    let omega_en_n: Vector3<f64> = Vector3::new(
        -velocities[1] / (r_n + *altitude),
        velocities[0] / (r_e + *altitude),
        velocities[0] * lat_rad.tan() / (r_n + *altitude),
    );
    omega_en_n
}
/// Calculate the magnetic field using the Earth's dipole model in the local-level frame
///
/// This function computes the Earth's magnetic field at a given position using a simple
/// dipole model. The dipole model approximates the Earth's magnetic field as a magnetic
/// dipole with the axis through the geographic poles.
///
/// # Parameters
/// - `latitude` - The WGS84 latitude in degrees
/// - `longitude` - The WGS84 longitude in degrees  
/// - `altitude` - The WGS84 altitude in meters
///
/// # Returns
/// The magnetic field vector in nanoteslas (nT) in the local-level frame (North, East, Down)
///
/// # Example
/// ```rust
/// use nalgebra::Vector3;
/// use strapdown::earth;
/// let latitude: f64 = 45.0;
/// let longitude: f64 = -75.0;
/// let altitude: f64 = 0.0;
/// let magnetic_field = earth::calculate_magnetic_field(&latitude, &longitude, &altitude);
/// ```
pub fn calculate_magnetic_field(latitude: &f64, longitude: &f64, altitude: &f64) -> Vector3<f64> {
    // Calculate the magnetic colatitude and longitude
    let (mag_colatitude, _) = wgs84_to_magnetic(latitude, longitude);

    // Calculate the radial and latitudinal components of the magnetic field
    let radial_field = calculate_radial_magnetic_field(mag_colatitude.to_radians(), *altitude);
    let lat_field = calculate_latitudinal_magnetic_field(mag_colatitude.to_radians(), *altitude);

    // Create the magnetic field vector in the local-level frame (NED)
    let b_vector: Vector3<f64> = Vector3::new(radial_field, lat_field, 0.0);

    b_vector
}

/// Calculate the radial component of Earth's magnetic field using the dipole model
///
/// # Parameters
/// - `colatitude` - The magnetic *colatitude* in radians (angle from magnetic north pole)
/// - `radius` - The distance from Earth's center in meters
///
/// # Returns
/// The radial component of the magnetic field in nanoteslas (nT)
///
/// # Example
/// ```rust
/// use strapdown::earth;
/// let colatitude: f64 = 1.0;
/// let radius: f64 = 6371000.0;
/// let radial_field = earth::calculate_radial_magnetic_field(colatitude, radius);
/// ```
pub fn calculate_radial_magnetic_field(colatitude: f64, radius: f64) -> f64 {
    // let radius_ratio = radius / MAGNETIC_REFERENCE_RADIUS;
    // return -2.0 * MAGNETIC_FIELD_STRENGTH / radius_ratio.powi(3) * latitude.sin();
    -2.0 * MAGNETIC_FIELD_STRENGTH * (MEAN_RADIUS / radius).powi(3) * colatitude.cos()
}

/// Calculate the latitudinal component of Earth's magnetic field using the dipole model
///
/// # Parameters
/// - `colatitude` - The magnetic *colatitude* in radians (angle from magnetic north pole)
/// - `radius` - The distance from Earth's center in meters
///
/// # Returns
/// The latitudinal component of the magnetic field in nanoteslas (nT)
///
/// # Example
/// ```rust
/// use strapdown::earth;
/// let colatitude: f64 = 1.0;
/// let radius: f64 = 6371000.0;
/// let lat_field = earth::calculate_latitudinal_magnetic_field(colatitude, radius);
/// ```
pub fn calculate_latitudinal_magnetic_field(colatitude: f64, radius: f64) -> f64 {
    -MAGNETIC_FIELD_STRENGTH * (MEAN_RADIUS / radius).powi(3) * colatitude.sin()
}
/// Calculate magnetic colatitude and longitude from WGS84 coordinates
///
/// This function transforms WGS84 geographic coordinates to geomagnetic coordinates
/// using the dipole model of Earth's magnetic field. The transformation is based on
/// the location of the geomagnetic north pole.
///
/// # Parameters
/// - `latitude` - The WGS84 latitude in degrees
/// - `longitude` - The WGS84 longitude in degrees
///
/// # Returns
/// A tuple containing (magnetic_colatitude, magnetic_longitude) in degrees. Colatitude
/// is the angle from the magnetic north pole [0, 180], and longitude is the angle from the
/// magnetic meridian.
///
/// # Example
/// ```rust
/// use strapdown::earth;
/// let latitude = 45.0;
/// let longitude = -75.0;
/// let (mag_lat, mag_lon) = earth::wgs84_to_magnetic(&latitude, &longitude);
/// ```
pub fn wgs84_to_magnetic(latitude: &f64, longitude: &f64) -> (f64, f64) {
    // Convert all angles to radians for calculations
    let lat_rad = latitude.to_radians();
    let lon_rad = longitude.to_radians();
    let mag_lat_rad = MAGNETIC_NORTH_LATITUDE.to_radians();
    let mag_lon_rad = MAGNETIC_NORTH_LONGITUDE.to_radians();

    // Calculate the magnetic latitude
    // This is based on the spherical angle between the point and the geomagnetic pole
    let cos_theta = lat_rad.sin() * mag_lat_rad.sin()
        + lat_rad.cos() * mag_lat_rad.cos() * (lon_rad - mag_lon_rad).cos();
    // Calculate magnetic longitude
    // This requires finding the azimuth from the magnetic pole to the point
    let y = (lon_rad - mag_lon_rad).sin() * lat_rad.cos();
    let x = mag_lat_rad.cos() * lat_rad.sin()
        - mag_lat_rad.sin() * lat_rad.cos() * (lon_rad - mag_lon_rad).cos();

    // Magnetic latitude is defined as the angle from the magnetic pole
    let mag_latitude = cos_theta.acos().to_degrees();
    // Magnetic longitude is defined relative to the magnetic meridian (0°)
    let mag_longitude = y.atan2(x).to_degrees();

    (mag_latitude, mag_longitude)
}
/// Calculate the magnetic inclination (dip angle) at a given location
///
/// The magnetic inclination is the angle between the horizontal plane and the
/// magnetic field vector, positive downward. At the magnetic equator, the inclination
/// is zero. At the magnetic poles, the inclination is ±90°.
///
/// # Parameters
/// - `latitude` - The WGS84 latitude in degrees
/// - `longitude` - The WGS84 longitude in degrees
/// - `altitude` - The WGS84 altitude in meters
///
/// # Returns
/// The magnetic inclination angle in degrees
///
/// # Example
/// ```rust
/// use strapdown::earth;
/// let latitude = 45.0;
/// let longitude = -75.0;
/// let altitude = 0.0;
/// let inclination = earth::magnetic_inclination(&latitude, &longitude, &altitude);
/// ```
pub fn magnetic_inclination(latitude: &f64, longitude: &f64, altitude: &f64) -> f64 {
    // Get the magnetic field vector in NED coordinates
    let b_vector = calculate_magnetic_field(latitude, longitude, altitude);

    // Horizontal component magnitude (North-East plane)
    let b_h = (b_vector[0].powi(2) + b_vector[1].powi(2)).sqrt();

    // Calculate inclination (dip angle)
    // Positive downward, negative upward

    (b_vector[2] / b_h).atan().to_degrees()
}
/// Calculate the magnetic declination (variation) at a given location
///
/// The magnetic declination is the angle between true north and magnetic north,
/// positive eastward. This is essential for navigation as it represents the correction
/// needed to convert between magnetic compass readings and true bearings.
///
/// # Parameters
/// - `latitude` - The WGS84 latitude in degrees
/// - `longitude` - The WGS84 longitude in degrees
/// - `altitude` - The WGS84 altitude in meters
///
/// # Returns
/// The magnetic declination angle in degrees
///
/// # Example
/// ```rust
/// use strapdown::earth;
/// let latitude = 45.0;
/// let longitude = -75.0;
/// let altitude = 0.0;
/// let declination = earth::magnetic_declination(&latitude, &longitude, &altitude);
/// ```
pub fn magnetic_declination(latitude: &f64, longitude: &f64, altitude: &f64) -> f64 {
    // Get the magnetic field vector in NED coordinates
    let b_vector = calculate_magnetic_field(latitude, longitude, altitude);

    // Calculate declination (variation)
    // Positive eastward, negative westward

    (b_vector[1] / b_vector[0]).atan().to_degrees()
}
/// Calculate the magnetic anomaly at a given location
pub fn magnetic_anomaly(
    latitude: f64,
    longitude: f64,
    altitude: f64,
    mag_x: f64,
    mag_y: f64,
    mag_z: f64,
) -> f64 {
    0.0 * latitude * longitude * altitude * mag_x * mag_y * mag_z // Placeholder for magnetic anomaly calculation
}
// === Unit tests ===
#[cfg(test)]
mod tests {
    use super::*;
    use assert_approx_eq::assert_approx_eq;
    #[test]
    fn vector_to_skew_symmetric() {
        let v: Vector3<f64> = Vector3::new(1.0, 2.0, 3.0);
        let skew: Matrix3<f64> = super::vector_to_skew_symmetric(&v);
        assert_eq!(skew[(0, 1)], -v[2]);
        assert_eq!(skew[(0, 2)], v[1]);
        assert_eq!(skew[(1, 0)], v[2]);
        assert_eq!(skew[(1, 2)], -v[0]);
        assert_eq!(skew[(2, 0)], -v[1]);
        assert_eq!(skew[(2, 1)], v[0]);
    }
    #[test]
    fn skew_symmetric_to_vector() {
        let v: Vector3<f64> = Vector3::new(1.0, 2.0, 3.0);
        let skew: Matrix3<f64> = super::vector_to_skew_symmetric(&v);
        let v2: Vector3<f64> = super::skew_symmetric_to_vector(&skew);
        assert_eq!(v, v2);
    }
    #[test]
    fn gravity() {
        // test polar gravity
        let latitude: f64 = 90.0;
        let grav = super::gravity(&latitude, &0.0);
        assert_approx_eq!(grav, GP);
        // test equatorial gravity
        let latitude: f64 = 0.0;
        let grav = super::gravity(&latitude, &0.0);
        assert_approx_eq!(grav, GE);
    }
    #[test]
    fn gravitation() {
        // test equatorial gravity
        let latitude: f64 = 0.0;
        let altitude: f64 = 0.0;
        let grav: Vector3<f64> = super::gravitation(&latitude, &0.0, &altitude);
        assert_approx_eq!(grav[0], 0.0);
        assert_approx_eq!(grav[1], 0.0);
        assert_approx_eq!(grav[2], (GE + 0.0339), 1e-4);
        // test polar gravity
        let latitude: f64 = 90.0;
        let grav: Vector3<f64> = super::gravitation(&latitude, &0.0, &altitude);
        assert_approx_eq!(grav[0], 0.0);
        assert_approx_eq!(grav[1], 0.0);
        assert_approx_eq!(grav[2], GP, 1e-2);
    }
    #[test]
    fn magnetic_radial_field() {
        // Using magnetic co-latitude [0, 180]
        let lat: f64 = 0.0;
        let b_r: f64 = calculate_radial_magnetic_field(lat.to_radians(), MEAN_RADIUS);
        assert_approx_eq!(b_r, -2.0 * MAGNETIC_FIELD_STRENGTH, 1e-7);
        let lat: f64 = 180.0;
        let b_r: f64 = calculate_radial_magnetic_field(lat.to_radians(), MEAN_RADIUS);
        assert_approx_eq!(b_r, 2.0 * MAGNETIC_FIELD_STRENGTH, 1e-7);
        let lat: f64 = 90.0;
        let b_r: f64 = calculate_radial_magnetic_field(lat.to_radians(), MEAN_RADIUS);
        assert_approx_eq!(b_r, 0.0, 1e-7);
    }
    #[test]
    fn wgs84_to_magnetic() {
        let lat: f64 = 80.8;
        let lon: f64 = -72.8;
        let (mag_lat, mag_lon) = super::wgs84_to_magnetic(&lat, &lon);
        assert_approx_eq!(mag_lat, 0.0, 1e-7);
        assert_approx_eq!(mag_lon, 0.0, 1e-7);
    }
    #[test]
    fn eci_to_ecef() {
        let time: f64 = 30.0;
        let rot: Matrix3<f64> = super::eci_to_ecef(time);
        assert_approx_eq!(rot[(0, 0)], (RATE * time).cos(), 1e-7);
        assert_approx_eq!(rot[(0, 1)], (RATE * time).sin(), 1e-7);
        assert_approx_eq!(rot[(1, 0)], -(RATE * time).sin(), 1e-7);
        assert_approx_eq!(rot[(1, 1)], (RATE * time).cos(), 1e-7);
        let rot_t = ecef_to_eci(time);
        assert_approx_eq!(rot_t[(0, 0)], (RATE * time).cos(), 1e-7);
        assert_approx_eq!(rot_t[(0, 1)], -(RATE * time).sin(), 1e-7);
        assert_approx_eq!(rot_t[(1, 0)], (RATE * time).sin(), 1e-7);
        assert_approx_eq!(rot_t[(1, 1)], (RATE * time).cos(), 1e-7);
        assert_approx_eq!(rot_t[(2, 2)], 1.0, 1e-7);
    }
    #[test]
    fn ecef_to_lla() {
        let latitude: f64 = 45.0;
        let longitude: f64 = 90.0;
        let rot: Matrix3<f64> = super::ecef_to_lla(&latitude, &longitude);
        assert_approx_eq!(
            rot[(0, 0)],
            -longitude.to_radians().sin() * latitude.to_radians().cos(),
            1e-7
        );
        assert_approx_eq!(
            rot[(0, 1)],
            -longitude.to_radians().sin() * latitude.to_radians().sin(),
            1e-7
        );
        assert_approx_eq!(rot[(0, 2)], latitude.to_radians().cos(), 1e-7);
        assert_approx_eq!(rot[(1, 0)], -longitude.to_radians().sin(), 1e-7);
        assert_approx_eq!(rot[(1, 1)], longitude.to_radians().cos(), 1e-7);
        assert_approx_eq!(
            rot[(2, 0)],
            -latitude.to_radians().cos() * longitude.to_radians().cos(),
            1e-7
        );
        assert_approx_eq!(
            rot[(2, 1)],
            -latitude.to_radians().cos() * longitude.to_radians().sin(),
            1e-7
        );
        assert_approx_eq!(rot[(2, 2)], -latitude.to_radians().sin(), 1e-7);
    }
    #[test]
    fn lla_to_ecef() {
        let latitude: f64 = 45.0;
        let longitude: f64 = 90.0;
        let rot1: Matrix3<f64> = super::lla_to_ecef(&latitude, &longitude);
        let rot2: Matrix3<f64> = super::ecef_to_lla(&latitude, &longitude).transpose();
        assert_approx_eq!(rot1[(0, 0)], rot2[(0, 0)], 1e-7);
        assert_approx_eq!(rot1[(0, 1)], rot2[(0, 1)], 1e-7);
        assert_approx_eq!(rot1[(0, 2)], rot2[(0, 2)], 1e-7);
        assert_approx_eq!(rot1[(1, 0)], rot2[(1, 0)], 1e-7);
        assert_approx_eq!(rot1[(1, 1)], rot2[(1, 1)], 1e-7);
        assert_approx_eq!(rot1[(1, 2)], rot2[(1, 2)], 1e-7);
        assert_approx_eq!(rot1[(2, 0)], rot2[(2, 0)], 1e-7);
        assert_approx_eq!(rot1[(2, 1)], rot2[(2, 1)], 1e-7);
        assert_approx_eq!(rot1[(2, 2)], rot2[(2, 2)], 1e-7);
    }
}
