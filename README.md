# Wildfire Occurrence and Burned Area Analysis

This repository contains Python scripts used in the study:

Kim (2026), Integration of Meteorological Accumulation and Satellite-Derived Vegetation Indices to Assess Wildfire Occurrence and Burned Area.

## Description

The code implements:

- Forest Fire Danger Index (FFDI) calculation
- Temperature and relative humidity accumulation windows (1h, 24h, 72h)
- Event-time percentile analysis
- Burned-area prediction analysis
- Rank-based ensemble comparison

## Data Sources

- Korea Meteorological Administration (KMA) 500 m gridded data
- GK2A satellite NDVI/EVI products

Note: Raw data are available from official portals (see manuscript Open Research section).

## How to Run

1. Install dependencies:
   pip install -r requirements.txt

2. Run main script:
   python main_analysis.py

## Author

Sang Yeob Kim
Department of Fire and Disaster Prevention
Konkuk University
