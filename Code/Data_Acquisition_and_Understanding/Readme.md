# Data Acquisition and Understanding

This directory contains scripts and notebooks for preprocessing raw HVAC data, aligning asynchronous sources, and conducting exploratory data analysis (EDA).

### Files:

* **`dataPrep.py`**
  The main preprocessing script. It cleans, merges, and interpolates sporadic indoor thermostat logs with 15-minute interval outdoor weather data to create machine-learning-ready datasets.

* **`datapipeline.json`**
  A placeholder configuration file (nothing done here yet)

* **`temperature_analysis.ipynb`**
  An EDA notebook used to evaluate physical hardware behavior, thermal dynamics, and temperature error against outdoor environmental variables.

* **`TimestampAnalysis.ipynb`**
  An EDA notebook that analyzes the temporal gaps in the raw IoT logs to identify the longest continuous stretches of usable data for modeling.

* **`outsideDataPrep.py`**
  This a preprocessing script for the outdoor weather data. 
