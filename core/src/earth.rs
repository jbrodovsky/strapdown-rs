//! Earth-related constants and functions
//!
//! This module contains constants and functions related to the Earth's shape and gravity.
//! The Earth is modeled as an ellipsoid (WGS84) with a semi-major axis and a semi-minor
//! axis. The Earth's gravity is modeled as a function of the latitude and altitude using
//! the Somigliana method. The Earth's rotation rate is also included in this module.
//! This module relies on the `nav-types` crate for the coordinate types and conversions,
//! but provides additional functionality for calculating rotations for the strapdown
//! navigation filters. This permits the transformation of additional quantities (velocity,
//! acceleration, etc.) between the Earth-centered Earth-fixed (ECEF) frame and the
//! local-level frame.
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
//! For basic positional conversions, the `nav-types` crate is used. This crate provides
//! the `WGS84` and `ECEF` types for representing the Earth's position in geodetic and
//! Cartesian coordinates, respectively. The `nav-types` crate also provides the necessary
//! conversions between the two coordinate systems.
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
use ::nalgebra::{Matrix3, Vector3};
use ::nav_types::{ECEF, WGS84};

// Earth constants (WGS84)
pub const RATE: f64 = 7.2921159e-5; // rad/s (omega_ie)
pub const RATE_VECTOR: Vector3<f64> = Vector3::new(0.0, 0.0, RATE);
pub const EQUATORIAL_RADIUS: f64 = 6378137.0; // meters
pub const POLAR_RADIUS: f64 = 6356752.31425; // meters
pub const ECCENTRICITY: f64 = 0.0818191908425; // unit-less
pub const ECCENTRICITY_SQUARED: f64 = ECCENTRICITY * ECCENTRICITY;
pub const GE: f64 = 9.7803253359; // m/s^2, equatorial radius
pub const GP: f64 = 9.8321849378; // m/s^2, polar radius
pub const F: f64 = 1.0 / 298.257223563; // Flattening factor
pub const K: f64 = (POLAR_RADIUS * GP - EQUATORIAL_RADIUS * GE) / (EQUATORIAL_RADIUS * GE); // Somigliana's constant

/// Convert a three-element vector to a skew-symmetric matrix
/// Groves' notation uses a lot of skew-symmetric matrices to represent cross products
/// and to perform more concise matrix operations (particularly involving rotations).
/// This function converts a three-element vector to a skew-symmetric matrix.
///
/// # Skew-symmetric matrix conversion
///
/// Give a nalgebra vector `v` = [v1, v2, v3], the skew-symmetric matrix `skew` is defined as:
///
/// ```text
/// skew = |  0  -v3   v2 |
///        | v3   0   -v1 |
///        |-v2   v1   0  |
/// ```
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
    return skew;
}
/// Convert a skew-symmetric matrix to a three-element vector
/// This function converts a skew-symmetric matrix to a three-element vector. This is the
/// inverse operation of the `vector_to_skew_symmetric` function.
///
/// # Skew-symmetric matrix conversion
///
/// Give a nalgebra Matrix3 `skew` where
/// ```text
/// skew = |  0  -v3   v2 |
///        | v3   0   -v1 |
///        |-v2   v1   0  |
/// ```
/// the vector `v` is defined as v = [v1, v2, v3]
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
    return v;
}
/// Coordinate conversion from the Earth-centered Inertial (ECI) frame to the Earth-centered
/// Earth-fixed (ECEF) frame. The ECI frame is a right-handed Cartesian coordinate system with
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
    return rot;
}
/// Coordinate conversion from the Earth-centered Earth-fixed (ECEF) frame to the Earth-centered
/// Inertial (ECI) frame. The ECI frame is a right-handed Cartesian coordinate system with
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
    return eci_to_ecef(time).transpose();
}
/// Coordinate conversion from the Earth-centered Earth-fixed (ECEF) frame to the local-level frame.
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
    return rot;
}

/// Coordinate conversion from the local-level frame to the Earth-centered Earth-fixed (ECEF) frame.
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
    return ecef_to_lla(latitude, longitude).transpose();
}
/// Calculate principal radii of curvature 
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
    return (r_n, r_e, r_p);
}
/// Calculate the WGS84 gravity scalar
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
    return g0 - 3.08e-6 * altitude;
}

/// Calculate the gravitational force vector in the local-level frame
/// The [gravity model](https://en.wikipedia.org/wiki/Gravity_of_Earth) is based on the [Somigliana
/// method](https://en.wikipedia.org/wiki/Theoretical_gravity#Somigliana_equation), which models
/// the Earth's gravity as a function of the latitude and altitude. The gravity model is used to
/// calculate the gravitational force vector in the local-level frame. This is then combined
/// with the rotational effects of the Earth to calculate the effective gravity vector. This
/// differs from the gravity scalar in that it includes the centrifugal effects of the Earth's
/// rotation.
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
    let wgs84: WGS84<f64> = WGS84::from_degrees_and_meters(*latitude,*longitude, *altitude);
    let ecef: ECEF<f64> = ECEF::from(wgs84);
    // Get centrifugal terms in ECEF
    let ecef_vec: Vector3<f64> = Vector3::new(ecef.x(), ecef.y(), ecef.z());
    let omega_ie: Matrix3<f64> = vector_to_skew_symmetric(&RATE_VECTOR);
    // Get rotation and gravity in LLA    
    let rot: Matrix3<f64> = ecef_to_lla(latitude, longitude);
    let gravity: Vector3<f64> = Vector3::new(0.0, 0.0, gravity(latitude, altitude));
    // Calculate the effective gravity vector combining gravity and centrifugal terms    
    return gravity + rot * omega_ie * omega_ie * ecef_vec;
}
/// Calculate the Earth rotation rate vector in the local-level frame
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
    return omega_ie;
}
/// Calculate the transport rate vector in the local-level frame
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
    return omega_en_n;
}

// TODO: #82 Implement magnetic field calculations

/// Calculate the magnetic field intensity in the local-level frame
/// 
pub fn calculate_magnetic_field() -> f64 {
    return 0.0;
}

// === Unit tests ===
#[cfg(test)]
mod tests {
    use super::*;
    use assert_approx_eq::assert_approx_eq;
    #[test]
    fn test_vector_to_skew_symmetric() {
        let v: Vector3<f64> = Vector3::new(1.0, 2.0, 3.0);
        let skew: Matrix3<f64> = vector_to_skew_symmetric(&v);
        assert_eq!(skew[(0, 1)], -v[2]);
        assert_eq!(skew[(0, 2)], v[1]);
        assert_eq!(skew[(1, 0)], v[2]);
        assert_eq!(skew[(1, 2)], -v[0]);
        assert_eq!(skew[(2, 0)], -v[1]);
        assert_eq!(skew[(2, 1)], v[0]);
    }
    #[test]
    fn test_skew_symmetric_to_vector() {
        let v: Vector3<f64> = Vector3::new(1.0, 2.0, 3.0);
        let skew: Matrix3<f64> = vector_to_skew_symmetric(&v);
        let v2: Vector3<f64> = skew_symmetric_to_vector(&skew);
        assert_eq!(v, v2);
    }
    #[test]
    fn test_gravity() {
        // test polar gravity
        let latitude: f64 = 90.0;
        let grav = gravity(&latitude, &0.0);
        assert_approx_eq!(grav, GP);
        // test equatorial gravity
        let latitude: f64 = 0.0;
        let grav = gravity(&latitude, &0.0);
        assert_approx_eq!(grav, GE);
    }
    #[test]
    fn test_gravitation() {
        // test equatorial gravity
        let latitude: f64 = 0.0;
        let altitude: f64 = 0.0;
        let grav: Vector3<f64> = gravitation(&latitude, &0.0, &altitude);
        assert_approx_eq!(grav[0], 0.0);
        assert_approx_eq!(grav[1], 0.0);
        assert_approx_eq!(grav[2], GE + 0.0339, 1e-4);
        // test polar gravity
        let latitude: f64 = 90.0;
        let grav: Vector3<f64> = gravitation(&latitude, &0.0, &altitude);
        assert_approx_eq!(grav[0], 0.0);
        assert_approx_eq!(grav[1], 0.0);
        assert_approx_eq!(grav[2], GP);
    }
}
