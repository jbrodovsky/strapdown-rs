#!/bin/bash
# Test script for the geophysical configuration wizard

echo "Testing strapdown-sim config wizard with geophysical parameters..."
echo ""

# Simulated user input:
# - Config name: test_geo_config.toml
# - Save path: /tmp
# - Input: data/input/test_data.csv
# - Output: /tmp/test_output.csv
# - Mode: cl (closed-loop)
# - Filter: 1 (UKF)
# - Seed: 42
# - Parallel: n
# - Log level: info
# - Log file: (empty, use stderr)
# - Scheduler: 1 (Always)
# - Fault: 1 (None)
# - Enable geophysical: y
# - Gravity enabled: y
# - Gravity resolution: 11 (OneMinute)
# - Gravity bias: 5.0
# - Gravity noise std: 100
# - Gravity map file: (empty, auto-detect)
# - Magnetic enabled: y
# - Magnetic resolution: 11 (OneMinute)
# - Magnetic bias: 10.0
# - Magnetic noise std: 150
# - Magnetic map file: (empty, auto-detect)
# - Geo frequency: 1.0

cd /home/james/Code/strapdown-rs

cat << EOF | cargo run --package strapdown-sim -- config
test_geo_config.toml
/tmp
data/input/test_data.csv
/tmp/test_output.csv
cl
1
42
n
info

1
1
y
y
11
5.0
100
auto
y
11
10.0
150
auto
1.0
EOF

echo ""
echo "Configuration file created. Checking contents..."
echo ""

if [ -f /tmp/test_geo_config.toml ]; then
    cat /tmp/test_geo_config.toml
    echo ""
    echo "✓ Test completed successfully!"
    rm /tmp/test_geo_config.toml
else
    echo "✗ Configuration file was not created"
    exit 1
fi
