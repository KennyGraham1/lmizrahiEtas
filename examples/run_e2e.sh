#!/bin/bash
# End-to-End Pipeline for NZ-Wide ETAS Calibration & Evaluation
# This script runs the entire pipeline from data ingestion to final reporting.

set -e

BACKGROUND_RATE_FILE="../input_data/hftlongtermmodel005.txt"
BACKGROUND_RATE_MAG="5.0"

echo "================================================================"
echo "    NZ-Wide ETAS Pipeline: End-to-End Execution                 "
echo "================================================================"

# Step 1: Download / Update Catalog
echo -e "\n[1/2] Downloading / Updating GeoNet Catalog..."
echo "----------------------------------------------------------------"
python3 download_geonet_catalog.py
echo "Catalog download complete."

# Step 2: Run Calibration Sweep (Inversion -> Forecast -> pyCSEP -> Plots)
echo -e "\n[2/2] Executing Calibration Sweep & Forecast Evaluation..."
echo "----------------------------------------------------------------"
# By default this will:
# - Use the polygon filter (from input_data/nz_polygon.npy)
# - Run parameter estimation (inversion) up to 2021-01-01
# - Run yearly forecast evaluations from 2021 to 2026
# - Generate PyCSEP diagnostic dashboards
# - Aggregate the scorecard and markdown report
python3 run_nz_wide_calibration_sweep.py \
  --background-rate-file "$BACKGROUND_RATE_FILE" \
  --background-rate-mag "$BACKGROUND_RATE_MAG"

echo -e "\n================================================================"
echo "    Pipeline Execution Complete!                                "
echo "================================================================"
echo "Check 'examples/output_nz_wide_calibration/' for the final scorecard and report."
