//! Rust toolbox for geophysical navigation and map matching
//!
//! This module works alongside the `strapdown-core` crate to provide complimentary geophysical navigation aiding
//! and map matching functionality. It also provides a set of tools for working with geophysical maps, such as
//! relief, gravity, and magnetic maps. These maps are downloaded from the GMT database and can be used for
//! navigation and map matching purposes.
use std::any::Any;
use std::fmt::Debug;
use std::io;
use std::path::PathBuf;

use nalgebra;
use nalgebra::{DMatrix, DVector, Vector3};
use strapdown::IMUData;
use strapdown::earth::METERS_TO_DEGREES;
use strapdown::filter::{InitialState,
    GPSPositionAndVelocityMeasurement, MeasurementModel, RelativeAltitudeMeasurement, UnscentedKalmanFilter,
};
use strapdown::sim::{NavigationResult, TestDataRecord};

//================= Map Information ========================================================================
/// Resolution values for bathymetric or terrain relief maps
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReliefResolution {
    OneDegree,
    ThirtyMinutes,
    TwentyMinutes,
    FifteenMinutes,
    TenMinutes,
    SixMinutes,
    FiveMinutes,
    FourMinutes,
    ThreeMinutes,
    TwoMinutes,
    OneMinute,
    ThirtySeconds,
    FifteenSeconds,
    ThreeSeconds,
    OneSecond,
}
// impl ReliefResolution {
//     /// Convert the resolution to a string. This can be used for calling the GMT library
//     fn as_str(&self) -> &'static str {
//         match self {
//             ReliefResolution::OneDegree => "01d",
//             ReliefResolution::ThirtyMinutes => "30m",
//             ReliefResolution::TwentyMinutes => "20m",
//             ReliefResolution::FifteenMinutes => "15m",
//             ReliefResolution::TenMinutes => "10m",
//             ReliefResolution::SixMinutes => "06m",
//             ReliefResolution::FiveMinutes => "05m",
//             ReliefResolution::FourMinutes => "04m",
//             ReliefResolution::ThreeMinutes => "03m",
//             ReliefResolution::TwoMinutes => "02m",
//             ReliefResolution::OneMinute => "01m",
//             ReliefResolution::ThirtySeconds => "30s",
//             ReliefResolution::FifteenSeconds => "15s",
//             ReliefResolution::ThreeSeconds => "03s",
//             ReliefResolution::OneSecond => "01s",
//         }
//     }
// }
/// Resolution values for gravity maps
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GravityResolution {
    OneDegree,
    ThirtyMinutes,
    TwentyMinutes,
    FifteenMinutes,
    TenMinutes,
    SixMinutes,
    FiveMinutes,
    FourMinutes,
    ThreeMinutes,
    TwoMinutes,
    OneMinute,
}
// impl GravityResolution {
//     /// Convert the resolution to a string. This can be used for calling the GMT library
//     fn as_str(&self) -> &'static str {
//         match self {
//             GravityResolution::OneDegree => "01d",
//             GravityResolution::ThirtyMinutes => "30m",
//             GravityResolution::TwentyMinutes => "20m",
//             GravityResolution::FifteenMinutes => "15m",
//             GravityResolution::TenMinutes => "10m",
//             GravityResolution::SixMinutes => "06m",
//             GravityResolution::FiveMinutes => "05m",
//             GravityResolution::FourMinutes => "04m",
//             GravityResolution::ThreeMinutes => "03m",
//             GravityResolution::TwoMinutes => "02m",
//             GravityResolution::OneMinute => "01m",
//         }
//     }
// }
/// Resolution values for magnetic maps
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MagneticResolution {
    OneDegree,
    ThirtyMinutes,
    TwentyMinutes,
    FifteenMinutes,
    TenMinutes,
    SixMinutes,
    FiveMinutes,
    FourMinutes,
    ThreeMinutes,
    TwoMinutes,
}
// impl MagneticResolution {
//     /// Convert the resolution to a string. This can be used for calling the GMT library
//     fn as_str(&self) -> &'static str {
//         match self {
//             MagneticResolution::OneDegree => "01d",
//             MagneticResolution::ThirtyMinutes => "30m",
//             MagneticResolution::TwentyMinutes => "20m",
//             MagneticResolution::FifteenMinutes => "15m",
//             MagneticResolution::TenMinutes => "10m",
//             MagneticResolution::SixMinutes => "06m",
//             MagneticResolution::FiveMinutes => "05m",
//             MagneticResolution::FourMinutes => "04m",
//             MagneticResolution::ThreeMinutes => "03m",
//             MagneticResolution::TwoMinutes => "02m",
//         }
//     }
// }
/// Enum for the different types of maps. A GeoMap is defined by its measurement type and resolution.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GeophysicalMeasurementType {
    Relief(ReliefResolution),
    Gravity(GravityResolution),
    Magnetic(MagneticResolution),
}
/// Struct for the GeoMap object
/// This struct contains the latitude and longitude vectors, the data matrix, and the type of map
/// The data matrix is a 2D matrix of data values, where the rows are the latitudes and the columns
/// are the longitudes. The data values are the values at the corresponding lat/lon points.
/// The map type is an enum that indicates the type of map (Relief, Gravity, Magnetic)
/// The lat/lon vectors are used to determine the bounds of the map and to interpolate the data values
#[derive(Clone, PartialEq)]
pub struct GeoMap {
    lats: DVector<f64>,
    lons: DVector<f64>,
    data: DMatrix<f64>,
    map_type: GeophysicalMeasurementType,
}
impl Debug for GeoMap {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "GeoMap {{ lats: {:?}, lons: {:?}, map_type: {:?} }}",
            self.lats, self.lons, self.map_type
        )
    }
}
impl GeoMap {
    /// Create a new GeoMap object from the supplied latitudes, longitudes, data matrix, and map type
    ///
    /// # Arguments
    /// - `lats` - A vector of latitudes
    /// - `lons` - A vector of longitudes
    /// - `data` - A matrix of data values
    /// - `map_type` - The type of map (Relief, Gravity, Magnetic)
    ///
    /// # Returns
    /// - A new GeoMap object
    ///
    /// # Example
    /// ```rust
    /// use nalgebra::{DVector, DMatrix};
    /// use navtoolbox::gmt_toolbox::{GeoMap, GeoMeasurement, ReliefResolution};
    /// let lats = DVector::from_vec(vec![1.0, 2.0, 3.0]);
    /// let lons = DVector::from_vec(vec![1.0, 2.0, 3.0]);
    /// let data = DMatrix::from_vec(3, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
    /// let map_type = GeoMeasurement::Relief(ReliefResolution::OneDegree);
    /// let map = GeoMap::new(lats, lons, data, map_type);
    /// ```
    pub fn new(
        lats: DVector<f64>,
        lons: DVector<f64>,
        data: DMatrix<f64>,
        map_type: GeophysicalMeasurementType,
    ) -> Self {
        GeoMap {
            lats,
            lons,
            data,
            map_type,
        }
    }
    /// Load a GeoMap from a netcdf file. GMT processing does not encode the map type in the file, so this
    /// function requires the user to specify the type of map along with the filename.
    ///
    /// # Arguments
    /// - `filename` - The PathBuf of the netcdf file
    /// - `map_type` - The type of map (Relief, Gravity, Magnetic)
    ///
    /// # Returns
    /// - A Result containing a reference to the GeoMap object or an error message
    ///
    /// # Example
    /// ```rust
    /// use navtoolbox::gmt_toolbox::{GeoMap, GeoMeasurement, ReliefResolution};
    /// use std::path::PathBuf;
    /// let map = GeoMap::load_geomap(PathBuf::from("path/to/file.nc"), GeophysicalMeasurementType::Relief(ReliefResolution::OneDegree));
    /// ```
    pub fn load_geomap(
        filename: PathBuf,
        map_type: GeophysicalMeasurementType,
    ) -> Result<GeoMap, String> {
        // Open the netcdf file
        let file = match netcdf::open(filename) {
            Ok(file) => file,
            Err(e) => panic!("Error opening file: {e:?}"),
        };
        // Get the lat/lon variables
        let lats: &netcdf::Variable<'_> =
            &file.variable("lat").expect("Could not find variable 'lat'");
        let lons: &netcdf::Variable<'_> =
            &file.variable("lon").expect("Could not find variable 'lon'");
        // Get the data variable
        let data: netcdf::Variable<'_> = file.variable("z").expect("Could not find variable 'z'");
        // Conversion to basic types
        let lats: Vec<f64> = lats.get_values(..).unwrap();
        let lons: Vec<f64> = lons.get_values(..).unwrap();
        let data: Vec<f64> = data.get_values(..).unwrap();
        // Convert the data to DVector
        let lats = DVector::from_vec(lats.to_vec());
        let lons = DVector::from_vec(lons.to_vec());
        // Convert the data to DMatrix
        let data = DMatrix::from_row_slice(lats.len(), lons.len(), &data);
        // Create the GeoMap object
        Ok(GeoMap::new(lats, lons, data, map_type))
    }
    /// Get the latitude vector
    /// # Returns
    /// - A reference to the latitude vector
    pub fn get_lats(&self) -> &DVector<f64> {
        &self.lats
    }
    /// Get the longitude vector
    /// # Returns
    /// - A reference to the longitude vector
    pub fn get_lons(&self) -> &DVector<f64> {
        &self.lons
    }
    /// Get the data matrix
    /// # Returns
    /// - A reference to the data matrix
    pub fn get_data(&self) -> &DMatrix<f64> {
        &self.data
    }
    /// Get the map type
    /// # Returns
    /// - A reference to the map type
    pub fn get_map_type(&self) -> &GeophysicalMeasurementType {
        &self.map_type
    }
    /// Get the map type as a string
    /// # Returns
    /// - A string representation of the map type
    pub fn get_map_type_str(&self) -> String {
        match self.map_type {
            GeophysicalMeasurementType::Relief(_) => "Relief".to_string(),
            GeophysicalMeasurementType::Gravity(_) => "Gravity".to_string(),
            GeophysicalMeasurementType::Magnetic(_) => "Magnetic".to_string(),
        }
    }
    /// Get the map data contained at a specific point.
    ///
    /// Queries the map for the data at a specific point given by the latitude and longitude coordinates
    /// and interpolates using a bilinear interpolation method.
    ///
    /// # Arguments
    /// - `lat` - The latitude of the point in degrees
    /// - `lon` - The longitude of the point in degrees
    ///
    /// # Returns
    /// - An Option containing the data value at the point, or None if the point is not in the map
    ///
    /// # Example
    /// ```rust
    /// use navtoolbox::gmt_toolbox::{GeoMap, GeoMeasurement, ReliefResolution};
    /// let map = GeoMap::load_geomap(PathBuf::from("path/to/file.nc"), GeoMeasurement::Relief(ReliefResolution::OneDegree));
    /// let value = map.get_point(&1.5, &1.5);
    /// ```
    /// # Panics
    /// - Panics if the lat/lon are out of bounds
    /// - Panics if the lat/lon are not in the map
    pub fn get_point(&self, lat: &f64, lon: &f64) -> Option<f64> {
        // Check if the lat/lon are within the bounds of the map
        if lat < &self.lats[0] || lat > &self.lats[self.lats.len() - 1] {
            panic!(
                "Latitude out of bounds: {} not in [{}, {}]",
                lat,
                self.lats[0],
                self.lats[self.lats.len() - 1]
            );
        }
        if lon < &self.lons[0] || lon > &self.lons[self.lons.len() - 1] {
            panic!("Longitude out of bounds");
        }
        // Check if the lat/lon are at the origin or the end of the map
        if lat == &self.lats[0] && lon == &self.lons[0] {
            // If the lat/lon are at the origin, return the first data point
            return Some(self.data[(0, 0)]);
        }
        if lat == &self.lats[self.lats.len() - 1] && lon == &self.lons[self.lons.len() - 1] {
            // If the lat/lon are at the end, return the last data point
            return Some(self.data[(self.lats.len() - 1, self.lons.len() - 1)]);
        }
        // Structure the interpolation in a few different ways. If the lat/lon are on the edge of the map,
        // only interpolate using the coordinate that is not on the edge.
        let lat_index = self.lats.iter().position(|&x| x >= *lat).unwrap();
        let lon_index = self.lons.iter().position(|&x| x >= *lon).unwrap();
        if lat == &self.lats[0] || lat == &self.lats[self.lats.len() - 1] {
            // If the latitude is on the edge, only interpolate using longitude
            let lon_index = self.lons.iter().position(|&x| x >= *lon).unwrap();
            // Special case for the edges of the map
            if lon_index == 0 {
                return Some(self.data[(lat_index, lon_index)]);
            }
            let lon1_index = lon_index - 1;
            let a = self.data[(lat_index, lon_index)];
            let b = self.data[(lat_index, lon1_index)];
            println!("a: {}, b: {}", a, b);
            let lon_diff = self.lons[lon_index] - self.lons[lon1_index];
            let result = ((a - b) / lon_diff) * (lon - &self.lons[lon1_index]) + b;
            return Some(result);
        }
        if lon == &self.lons[0] || lon == &self.lons[self.lons.len() - 1] {
            // If the longitude is on the edge, only interpolate using latitude
            let lat_index = self.lats.iter().position(|&x| x >= *lat).unwrap();
            if lat_index == 0 {
                return Some(self.data[(lat_index, lon_index)]);
            }
            let lat1_index = lat_index - 1;
            let a = self.data[(lat_index, lon_index)];
            let b = self.data[(lat1_index, lon_index)];
            let lat_diff = self.lats[lat_index] - self.lats[lat1_index];
            return Some(((a - b) / lat_diff) * (lat - &self.lats[lat1_index]) + b);
        }
        // If the lat/lon are not on the edge, use normal bilinear interpolation
        self.bilinear_interpolation(lat, lon)
    }
    /// Bilinear interpolation helper method for get_point
    fn bilinear_interpolation(&self, lat: &f64, lon: &f64) -> Option<f64> {
        // Find the indices that surround the point
        let lat2_index: usize = self.lats.iter().position(|&x| x >= *lat).unwrap();
        let lon2_index: usize = self.lons.iter().position(|&x| x >= *lon).unwrap();
        let lat1_index: usize = lat2_index - 1;
        let lon1_index: usize = lon2_index - 1;
        // Get the four surrounding points
        let q11: f64 = self.data[(lat1_index, lon1_index)];
        let q12: f64 = self.data[(lat2_index, lon1_index)];
        let q21: f64 = self.data[(lat1_index, lon2_index)];
        let q22: f64 = self.data[(lat2_index, lon2_index)];
        // Get the coordinates of the four surrounding points
        let lon1: f64 = self.lons[lon1_index];
        let lat1: f64 = self.lats[lat1_index];
        let lon2: f64 = self.lons[lon2_index];
        let lat2: f64 = self.lats[lat2_index];
        // Perform bilinear interpolation via weighted mean
        let w11: f64 = ((lon2 - lon) * (lat2 - lat)) / ((lon2 - lon1) * (lat2 - lat1));
        let w12: f64 = ((lon2 - lon) * (lat - lat1)) / ((lon2 - lon1) * (lat2 - lat1));
        let w21: f64 = ((lon - lon1) * (lat2 - lat)) / ((lon2 - lon1) * (lat2 - lat1));
        let w22: f64 = ((lon - lon1) * (lat - lat1)) / ((lon2 - lon1) * (lat2 - lat1));
        Some(w11 * q11 + w12 * q12 + w21 * q21 + w22 * q22)
    }
    // TODO: #95 Implement direct GMT interface using system shell calls
}
//================= Map Information ========================================================================

/// Generic Geophysical Map Measurement trait
#[derive(Clone, Debug)]
pub struct GeophysicalMeasurement {
    /// Source map
    pub map: GeoMap, //&'a should probably a references with lifetimes
    /// Measurement Noise
    pub noise_std: f64,
    /// Measured value
    pub measurement: f64,
}
/// Geophysical Measurement Model implementation
///
/// The geophysical feedback model is different from traditional measurement models in that it needs to relate
/// a scalar (or vector) measurement to the position states. We can somewhat do this via a map, but the scalar
/// values on the map are not necessarily unique. This means that we cannot directly use the measurement to
/// provide feedback and instead must use it to weight sigma points that do contain position information.
///
/// Working hypothesis: we can treat the UKF as a hybrid KF/PF where the sigma point / particles can be weighted
/// in the get_sigma_points function by a generic normally distributed anomaly measurement model.
impl MeasurementModel for GeophysicalMeasurement {
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn get_dimension(&self) -> usize {
        1 // Single measurement: map value at current position
    }
    // nalgebra::base::matrix::Matrix<f64, nalgebra::base::dimension::Dyn, nalgebra::base::dimension::Const<1>, nalgebra::base::vec_storage::VecStorage<f64, nalgebra::base::dimension::Dyn, nalgebra::base::dimension::Const<1>>>
    fn get_vector(&self) -> DVector<f64> {
        //let map_value = self.map.get_point(&0.0, &0.0);
        //DVector::from_vec(vec![map_value.unwrap_or(0.0)])
        DVector::<f64>::from_vec(vec![self.measurement]) // Placeholder until state is integrated
    }
    fn get_noise(&self) -> DMatrix<f64> {
        DMatrix::from_diagonal(&DVector::from_element(
            self.get_dimension(),
            self.noise_std.powi(2),
        ))
    }
    fn get_sigma_points(&self, state_sigma_points: &DMatrix<f64>) -> DMatrix<f64> {
        let mut measurement_sigma_points =
            DMatrix::<f64>::zeros(self.get_dimension(), state_sigma_points.ncols());
        for (i, sigma_point) in state_sigma_points.column_iter().enumerate() {
            measurement_sigma_points[i] = self
                .map
                .get_point(&sigma_point[0].to_degrees(), &sigma_point[1].to_degrees())
                .unwrap();
        }
        measurement_sigma_points
    }
}
/// Helper function to initialize a UKF for closed-loop geophysical navigation
///
/// This function sets up the Unscented Kalman Filter (UKF) with initial pose, attitude covariance, and IMU biases based on
/// the provided `TestDataRecord`. It initializes the UKF with position, velocity, attitude, and covariance matrices.
/// Optional parameters for attitude covariance and IMU biases can be provided to customize the filter's initial state.
/// Additionally you need to proved the measurement types as well as the geophysical measurement model's standard deviation.
/// Optionally, you can specify the GPS interval for the navigation updates to simulate operation in a GPS-denied environment.
///
/// This assumes a UKF state vector of the following form:
///
/// $$
/// \begin{bmatrix} x & y & z & \phi & \theta & \psi & \textbf{b}_{acc} & \textbf{b}_{gyro} & \beta \end{bmatrix}
/// $$
///
/// where $\beta$ is a generic placeholder for the geophysical measurement bias.
///
/// # Arguments
/// * `initial_pose` - A `TestDataRecord` containing the initial pose information.
/// * `attitude_covariance` - Optional vector of f64 representing the initial attitude covariance (default is a small value).
/// * `imu_biases` - Optional vector of f64 representing the initial IMU biases (default is a small value).
///
/// # Returns
///
/// * `UKF` - An instance of the Unscented Kalman Filter initialized with the provided parameters.
pub fn initialize_geo_ukf(
    initial_pose: &TestDataRecord,
    attitude_covariance: Option<Vec<f64>>,
    imu_biases: Option<Vec<f64>>,
    measurement_bias: f64,
    measurement_standard_deviation: f64,
) -> UnscentedKalmanFilter {
    //let mut rng = rand::thread_rng();
    let ukf_params = InitialState {
        latitude: initial_pose.latitude,
        longitude: initial_pose.longitude,
        altitude: initial_pose.altitude,
        northward_velocity: initial_pose.speed * initial_pose.bearing.cos(),
        eastward_velocity: initial_pose.speed * initial_pose.bearing.sin(),
        downward_velocity: 0.0, // Assuming no vertical velocity for simplicity
        roll: initial_pose.roll,
        pitch: initial_pose.pitch,
        yaw: initial_pose.yaw,
        in_degrees: true,
    };
    // Covariance parameters
    let position_accuracy = initial_pose.horizontal_accuracy; //.sqrt();
    let mut covariance_diagonal = vec![
        (position_accuracy * METERS_TO_DEGREES).powf(2.0),
        (position_accuracy * METERS_TO_DEGREES).powf(2.0),
        initial_pose.vertical_accuracy.powf(2.0),
        initial_pose.speed_accuracy.powf(2.0),
        initial_pose.speed_accuracy.powf(2.0),
        initial_pose.speed_accuracy.powf(2.0),
    ];
    // extend the covariance diagonal if attitude covariance is provided
    match attitude_covariance {
        Some(att_cov) => covariance_diagonal.extend(att_cov),
        None => covariance_diagonal.extend(vec![1e-9; 3]), // Default values if not provided
    }
    // extend the covariance diagonal if imu biases are provided
    let imu_bias = match imu_biases {
        Some(imu_biases) => {
            covariance_diagonal.extend(&imu_biases);
            imu_biases
        }
        None => {
            covariance_diagonal.extend(vec![1e-3; 6]);
            vec![1e-3; 6] // Default values if not provided
        }
    };
    covariance_diagonal.extend(vec![measurement_standard_deviation]);
    let process_noise_diagonal = DVector::from_vec(vec![
        1e-6, // position noise
        1e-6, // position noise
        1e-6, // altitude noise
        1e-3, // velocity north noise
        1e-3, // velocity east noise
        1e-3, // velocity down noise
        1e-5, // roll noise
        1e-5, // pitch noise
        1e-5, // yaw noise
        1e-6, // acc bias x noise
        1e-6, // acc bias y noise
        1e-6, // acc bias z noise
        1e-8, // gyro bias x noise
        1e-8, // gyro bias y noise
        1e-8, // gyro bias z noise
        1e-8, // measurement bias noise
    ]);
    // Create the filter
    UnscentedKalmanFilter::new(
        ukf_params,
        imu_bias,
        Some(vec![measurement_bias]),
        covariance_diagonal,
        DMatrix::from_diagonal(&process_noise_diagonal),
        1e-3,
        2.0,
        0.0,
    )
}

/// Main simulation function to run a geophysical navigation simulation
pub fn run_geophysical_navigation(
    records: &[TestDataRecord],
    measurement_type: GeophysicalMeasurementType,
    measurement_bias: f64,
    measurement_standard_deviation: f64,
    geo_map: GeoMap,
    gps_interval: Option<f64>,
    gps_degradation: Option<f64>,
) -> Result<Vec<NavigationResult>, io::Error> {
    println!(
        "Running geophysical navigation with measurement type: {:?}",
        measurement_type
    );
    println!("Number of records: {}", records.len());
    // Setup and initialization

    let gps_interval = gps_interval.unwrap_or(0.0);
    let gps_degradation = gps_degradation.unwrap_or(1.0);
    println!("GPS interval: {} seconds", gps_interval);
    println!("GPS degradation factor: {}", gps_degradation);
    let reference_altitude = records[0].altitude; // Use the first record's pressure as reference
    let start_time = records[0].time;
    let records_with_elapsed: Vec<(f64, &TestDataRecord)> = records
        .iter()
        .map(|r| ((r.time - start_time).num_milliseconds() as f64 / 1000.0, r))
        .collect();
    let mut results: Vec<NavigationResult> = Vec::with_capacity(records.len());
    let mut ukf = initialize_geo_ukf(
        &records[0],
        None,
        None,
        measurement_bias,
        measurement_standard_deviation,
    );
    // Set the initial result to the UKF initial state
    results.push(NavigationResult::from((&records[0].time, &ukf)));
    // Begin processing
    let mut previous_timestamp = records[0].time;
    // Iterate through the records, updating the UKF with each IMU measurement
    let total: usize = records.len();
    let mut i: usize = 1;
    let mut last_gps_update_time = 0.0;
    for (elapsed, record) in records_with_elapsed.iter().skip(1) {
        // Calculate time difference from the previous record
        let current_timestamp = record.time;
        let dt = (current_timestamp - previous_timestamp).as_seconds_f64();
        // Print progress every 10 iterations
        if i % 10 == 0 || i == total - 1 {
            print!(
                "\rElapsed time since last measurement: {} seconds | Processing data {:.2}%...",
                dt,
                (i as f64 / total as f64) * 100.0
            );
            //print_ukf(&ukf, record);
            use std::io::Write;
            std::io::stdout().flush().ok();
        }
        // Create IMU data from the record subtracting out biases
        let mean = ukf.get_mean();
        let imu_data = IMUData {
            accel: Vector3::new(
                record.acc_x - mean[9], // subtract accel bias
                record.acc_y - mean[10],
                record.acc_z - mean[11],
            ),
            gyro: Vector3::new(
                record.gyro_x - mean[12], // subtract gyro bias
                record.gyro_y - mean[13],
                record.gyro_z - mean[14],
            ),
        };
        // Update the UKF with the IMU data
        ukf.predict(imu_data, dt);
        ukf.get_covariance();
        // ---- Perform various measurement updates based on the available data ----
        // If GPS data is available, update the UKF with the GPS position and speed measurement
        if !record.latitude.is_nan()
            && !record.longitude.is_nan()
            && !record.altitude.is_nan()
            && !record.bearing.is_nan()
            && !record.speed.is_nan()
            && (elapsed - last_gps_update_time) >= gps_interval
        {
            let measurement = GPSPositionAndVelocityMeasurement {
                latitude: record.latitude,
                longitude: record.longitude,
                altitude: record.altitude,
                northward_velocity: record.speed * record.bearing.cos(),
                eastward_velocity: record.speed * record.bearing.sin(),
                horizontal_noise_std: record.horizontal_accuracy.sqrt(),
                vertical_noise_std: record.vertical_accuracy * gps_degradation,
                velocity_noise_std: record.speed_accuracy * gps_degradation,
            };
            ukf.update(&measurement);
            last_gps_update_time = *elapsed;
        }
        // If barometric altimeter data is available, update the UKF with the altitude measurement
        if !record.pressure.is_nan() {
            let altitude = RelativeAltitudeMeasurement {
                relative_altitude: record.relative_altitude,
                reference_altitude: reference_altitude,
            };
            ukf.update(&altitude);
        }
        // If anomaly data is available, update the UKF with the geophysical anomaly
        let state = ukf.get_mean();

        match measurement_type {
            GeophysicalMeasurementType::Relief(_) => {
                // Relief data isn't used in this example, but you could implement it similarly
                println!("Relief measurement not implemented in this example.");
            }
            GeophysicalMeasurementType::Gravity(_) => {
                if !record.grav_x.is_nan() & !record.grav_y.is_nan() & !record.grav_z.is_nan() {
                    let freeair = (record.grav_x.powf(2.0)
                        + record.grav_y.powf(2.0)
                        + record.grav_z.powf(2.0))
                    .sqrt()
                        - state[15];
                    let _geo_measurement = GeophysicalMeasurement {
                        map: geo_map.clone(),
                        noise_std: measurement_standard_deviation,
                        measurement: freeair,
                    };
                    // ukf.update(geo_measurement);
                }
            }
            GeophysicalMeasurementType::Magnetic(_) => {
                if !record.mag_x.is_nan() & !record.mag_y.is_nan() & !record.mag_z.is_nan() {
                    let mag_anom =
                        (record.mag_x.powf(2.0) + record.mag_y.powf(2.0) + record.mag_z.powf(2.0))
                            .sqrt()
                            - state[15];
                    let _geo_measurement = GeophysicalMeasurement {
                        map: geo_map.clone(),
                        noise_std: measurement_standard_deviation,
                        measurement: mag_anom,
                    };
                    // ukf.update(geo_measurement);
                }
            }
        }
        // Store the current state and covariance in results
        results.push(NavigationResult::from((&current_timestamp, &ukf)));
        i += 1;
        previous_timestamp = current_timestamp;
    }
    println!("Done!");
    Ok(results)
}

// ========================================================================================================================
#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{DMatrix, DVector};
    fn make_simple_geomap() -> GeoMap {
        // 2x2 grid: lats=[0,1], lons=[0,1], data = [[10,20],[30,40]]
        let lats = DVector::from_vec(vec![0.0, 1.0]);
        let lons = DVector::from_vec(vec![0.0, 1.0]);
        let data = DMatrix::from_row_slice(2, 2, &[10.0, 20.0, 30.0, 40.0]);
        GeoMap::new(
            lats,
            lons,
            data,
            GeophysicalMeasurementType::Relief(ReliefResolution::OneDegree),
        )
    }
}
