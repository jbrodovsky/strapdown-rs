# Strapdown Analysis Tools

Python package for analyzing strapdown INS simulations, creating datasets, and working with geophysical navigation data.

## Installation

### For Development (Editable Install)

From the workspace root:

```bash
pip install -e analysis/
```

Or using Pixi (recommended):

```bash
pixi install
```

### For Production

```bash
pip install analysis/
```

## Usage

The package provides a unified CLI tool `strapdown-analysis` with multiple subcommands:

### Preprocess Sensor Logger Data

Clean and merge raw sensor data from the Sensor Logger app:

```bash
strapdown-analysis preprocess --input data/raw/ --output data/input/
```

### Create Simulation Datasets

Run predefined simulation configurations against input CSV files:

```bash
strapdown-analysis create-dataset \
    --bin target/release/strapdown-sim \
    --input data/input/ \
    --output-root data/ \
    --jobs 4
```

### Download Geophysical Maps

Download relief, gravity, and magnetic anomaly maps for trajectories:

```bash
strapdown-analysis get-maps \
    --input data/input/ \
    --output data/input/ \
    --buffer 0.1
```

### Plot Routes

Generate route visualizations from CSV data:

```bash
strapdown-analysis plot \
    --input data/input/2023-08-04_214758.csv \
    --output plots/ \
    --title "Example Route"
```

### Generate Truth Mechanization

Run truth mechanization and create performance plots:

```bash
strapdown-analysis truth \
    --input-dir data/input/ \
    --output-dir data/baseline/
```

## Python API

You can also use the package programmatically:

```python
from strapdown_analysis.utils import find_strapdown_binary, inflate_bounds
from strapdown_analysis.utils.plotting import plot_street_map
from strapdown_analysis.configs import SIMULATION_CONFIGS, FILTER_TYPES

# Find the strapdown-sim binary
binary = find_strapdown_binary()

# Use geometry utilities
bounds = inflate_bounds(lon_min, lon_max, lat_min, lat_max, buffer=0.1)

# Create visualizations
fig = plot_street_map(latitudes, longitudes)
```

## Package Structure

```
analysis/
├── src/
│   └── strapdown_analysis/
│       ├── __init__.py           # Package entry point
│       ├── cli.py                # Unified CLI
│       ├── configs.py            # Simulation configurations
│       ├── commands/             # CLI command implementations
│       │   ├── create_dataset.py
│       │   ├── get_maps.py
│       │   ├── plot.py
│       │   ├── preprocess.py
│       │   └── truth.py
│       └── utils/                # Shared utilities
│           ├── binary.py
│           ├── geometry.py
│           ├── metrics.py
│           ├── plotting.py
│           └── simulation.py
├── pyproject.toml               # Package configuration
└── README.md                    # This file
```

## Development

### Running Tests

```bash
pytest
```

### Code Formatting

```bash
ruff format .
```

### Linting

```bash
ruff check --fix .
```

## License

MIT License - see LICENSE file in the repository root.
