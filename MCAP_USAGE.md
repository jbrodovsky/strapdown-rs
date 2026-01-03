# MCAP Support in Strapdown-rs

This document describes how to use MCAP (Modular Container for Analysis and Playback) file format support in the strapdown-rs library.

## Overview

MCAP is a container file format designed for storing timestamped data streams, commonly used in robotics applications. It's the recommended storage format for ROS 2 bags.

## Supported Types

### NavigationResult (Full Support ✓)

`NavigationResult` has full MCAP support with MessagePack encoding for efficient binary serialization.

#### Writing NavigationResult to MCAP

```rust
use strapdown::sim::NavigationResult;
use chrono::Utc;

// Create some navigation results
let mut result1 = NavigationResult::default();
result1.timestamp = Utc::now();
result1.latitude = 37.7749;
result1.longitude = -122.4194;
result1.altitude = 100.0;
result1.velocity_north = 10.0;
result1.velocity_east = 5.0;

let results = vec![result1];

// Write to MCAP file
NavigationResult::to_mcap(&results, "navigation_output.mcap")
    .expect("Failed to write navigation results");
```

#### Reading NavigationResult from MCAP

```rust
use strapdown::sim::NavigationResult;

// Read navigation results from MCAP file
let results = NavigationResult::from_mcap("navigation_output.mcap")
    .expect("Failed to read navigation results");

println!("Read {} navigation results", results.len());
for (i, result) in results.iter().enumerate() {
    println!("Result {}: lat={}, lon={}, alt={}", 
        i, result.latitude, result.longitude, result.altitude);
}
```

### TestDataRecord (Partial Support ⚠️)

`TestDataRecord` has **write-only** MCAP support. Reading back is currently not supported due to CSV-specific field deserializers.

#### Writing TestDataRecord to MCAP (Write-Only)

```rust
use strapdown::sim::TestDataRecord;

// Read from CSV
let records = TestDataRecord::from_csv("sensor_data.csv")
    .expect("Failed to read CSV");

// Write to MCAP (writing works fine)
TestDataRecord::to_mcap(&records, "sensor_data.mcap")
    .expect("Failed to write to MCAP");
```

**Note**: For production use with MCAP, we recommend converting `TestDataRecord` to `NavigationResult` after processing, as `NavigationResult` has full MCAP support.

## Technical Details

### Encoding Format

- **Message Encoding**: MessagePack (msgpack)
- **Channels**: 
  - `navigation_results` for NavigationResult messages
  - `sensor_data` for TestDataRecord messages
- **File Reading**: Uses memory-mapped files for efficient large file access

### Why MessagePack?

MessagePack was chosen for MCAP serialization because:
1. Efficient binary encoding (smaller file sizes than JSON)
2. Good compatibility with serde's derive macros
3. Self-describing format with schema information
4. Wide language support for interoperability

### Limitations

#### TestDataRecord CSV Deserializers

`TestDataRecord` uses custom deserializers (`de_f64_nan`) optimized for CSV parsing that expect string inputs. These conflict with binary serialization formats that provide native floating-point values. 

**Workaround**: Convert TestDataRecord to NavigationResult for MCAP storage:

```rust
use strapdown::sim::{TestDataRecord, NavigationResult};

// Read sensor data
let records = TestDataRecord::from_csv("sensor_data.csv")?;

// Process with navigation filter...
// (your navigation processing code here)

// Convert to NavigationResult and save as MCAP
NavigationResult::to_mcap(&nav_results, "output.mcap")?;
```

## MCAP File Structure

MCAP files created by strapdown-rs have the following structure:

```
MCAP File
├── Schema (msgpack encoding)
│   └── NavigationResult or TestDataRecord struct definition
├── Channel (topic: "navigation_results" or "sensor_data")
│   ├── encoding: "msgpack"
│   └── references schema above
└── Messages
    ├── Message 1 (sequence 0, timestamp, data)
    ├── Message 2 (sequence 1, timestamp, data)
    └── ...
```

## Viewing MCAP Files

You can inspect MCAP files using the [MCAP CLI tool](https://github.com/foxglove/mcap):

```bash
# Install MCAP CLI
pip install mcap-cli

# View file info
mcap info navigation_output.mcap

# List messages
mcap list navigation_output.mcap

# Export to JSON
mcap cat navigation_output.mcap --json
```

## Integration with ROS 2

MCAP files are natively supported by ROS 2 Iron and later. While strapdown-rs uses MessagePack encoding (not standard ROS 2 message types), the MCAP files can still be read by ROS 2 tools and visualized in [Foxglove Studio](https://foxglove.dev/).

For ROS 2 message compatibility, you would need to:
1. Define ROS 2 message types for your data structures
2. Use the ROS 2 serialization format (CDR)
3. Add appropriate ROS 2 metadata to the MCAP file

This is left as a future enhancement.

## Performance Considerations

### Memory-Mapped Reading

The `from_mcap()` methods use memory-mapped file I/O for efficient reading of large files:
- OS handles paging of file contents
- Minimal memory overhead
- Fast random access if needed
- Automatic cleanup when done

### File Sizes

MessagePack encoding provides good compression compared to text formats:
- ~50-70% smaller than JSON
- ~30-40% smaller than CSV for typical navigation data
- Comparable to other binary formats like Protobuf

## Example: Complete Workflow

```rust
use strapdown::sim::{TestDataRecord, NavigationResult};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Load sensor data from CSV
    let sensor_data = TestDataRecord::from_csv("raw_sensor_data.csv")?;
    
    // 2. Process with your navigation filter
    // (your navigation processing code here)
    let nav_results: Vec<NavigationResult> = process_navigation(&sensor_data);
    
    // 3. Save results to MCAP
    NavigationResult::to_mcap(&nav_results, "navigation_solution.mcap")?;
    
    // 4. Later, load and analyze
    let loaded_results = NavigationResult::from_mcap("navigation_solution.mcap")?;
    
    println!("Processed {} navigation solutions", loaded_results.len());
    
    Ok(())
}
```

## Dependencies

MCAP support requires the following crates:
- `mcap = "0.23"` - MCAP file format support
- `memmap2 = "0.9"` - Memory-mapped file I/O
- `rmp-serde = "1.3"` - MessagePack serialization

These are included in the strapdown-core dependencies.

## Further Reading

- [MCAP Format Specification](https://mcap.dev/spec)
- [MCAP Rust Library Documentation](https://docs.rs/mcap/)
- [MessagePack Format](https://msgpack.org/)
- [Foxglove Studio](https://foxglove.dev/) - Visualization tool for MCAP files
