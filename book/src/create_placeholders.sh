#!/bin/bash

# Installation pages
echo "# Installing from Crates.io" > installation/crates-io.md
echo -e "\nDetailed instructions for installing from Crates.io.\n\n*Content coming soon.*" >> installation/crates-io.md

echo "# Building from Source" > installation/building.md
echo -e "\nDetailed instructions for building from source.\n\n*Content coming soon.*" >> installation/building.md

# User Guide pages
echo "# User Guide Overview" > user-guide/overview.md
echo -e "\nComprehensive overview of using Strapdown-rs.\n\n*Content coming soon.*" >> user-guide/overview.md

echo "# Core Concepts" > user-guide/concepts.md
echo "# Strapdown Mechanization" > user-guide/strapdown-mechanization.md
echo "# Coordinate Frames" > user-guide/coordinate-frames.md
echo "# State Representation" > user-guide/state-representation.md
echo "# Running Simulations" > user-guide/simulations.md
echo "# Open-Loop Mode" > user-guide/open-loop.md
echo "# Closed-Loop Mode" > user-guide/closed-loop.md
echo "# Particle Filter Mode" > user-guide/particle-filter.md
echo "# Input Data Format" > user-guide/data-format.md
echo "# Configuration Files" > user-guide/configuration.md

# Copy existing logging doc
cp ../../docs/LOGGING.md user-guide/logging.md 2>/dev/null || echo "# Logging\n\n*Content coming soon.*" > user-guide/logging.md

# Filters
echo "# Kalman Filters" > filters/kalman.md
echo "# Extended Kalman Filter (EKF)" > filters/ekf.md
echo "# Unscented Kalman Filter (UKF)" > filters/ukf.md
echo "# EKF vs UKF Comparison" > filters/comparison.md
echo "# Particle Filters" > filters/particle-filter.md
echo "# Rao-Blackwellized Particle Filter" > filters/rbpf.md
echo "# Measurement Models" > filters/measurements.md

# Geonav
echo "# Geophysical Navigation Overview" > geonav/overview.md
echo "# Gravity Anomaly Navigation" > geonav/gravity.md
echo "# Magnetic Anomaly Navigation" > geonav/magnetic.md
echo "# Data Sources and Maps" > geonav/data-sources.md

# GNSS
echo "# Fault Simulation" > gnss/fault-simulation.md
echo "# Dropout Scenarios" > gnss/dropouts.md
echo "# Reduced Update Rates" > gnss/reduced-rates.md
echo "# Measurement Corruption" > gnss/corruption.md

# API
echo "# API Reference" > api/core.md
echo "# earth Module" > api/earth.md
echo "# kalman Module" > api/kalman.md
echo "# measurements Module" > api/measurements.md
echo "# particles Module" > api/particles.md
echo "# sim Module" > api/sim.md
echo "# strapdown-sim Binary" > api/sim-binary.md
echo "# strapdown-geonav" > api/geonav.md

# Examples
echo "# Example Configurations" > examples/configurations.md
echo "# Tutorial: Basic INS Simulation" > examples/tutorial-basic.md
echo "# Tutorial: GPS Degradation" > examples/tutorial-gps-degradation.md
echo "# Tutorial: Particle Filter" > examples/tutorial-particle-filter.md

# Development
echo "# Contributing" > development/contributing.md
echo "# Building and Testing" > development/building.md
echo "# Architecture" > development/architecture.md
echo "# Project Structure" > development/structure.md

# Resources
echo "# Publications" > resources/publications.md
echo "# External Links" > resources/links.md
echo "# Glossary" > resources/glossary.md

# FAQ
echo "# Frequently Asked Questions" > faq.md

echo "Placeholder files created!"
