# Data and Feature Definitions

This document provides a central hub for the raw data sources, the processed/transformed data, and feature sets. More details of each dataset is provided in the data summary report. 

For each data, an individual report describing the data schema, the meaning of each data field, and other information that is helpful for understanding the data is provided. If the dataset is the output of processing/transforming/feature engineering existing data set(s), the names of the input data sets, and the links to scripts that are used to conduct the operation are also provided. 

For each dataset, the links to the sample datasets in the _**Data**_ directory are also provided. 


## Raw Data Sources

| Raw Dataset Name | Link to the Full Dataset   | Full Dataset Size (MB)  | Link to Report |
| ---:| ---: | ---: | ---: |
| outdoorweather.csv | [link](outdoorweather.csv) | 0.903 |[Report Link](https://github.com/Temple-University-CIS-4496/cis-4496-project-hvac/blob/9da935a5ab17d1edf88714bb62209fd4659eabb0/Docs/Data_Report/DataSummaryReport.md)|
| timeseries_table_timeseries_table.csv(1)-(100) | https://www.kaggle.com/datasets/lsobieski/raw-thermostat-data | Each ~5 MB | [Report Link](https://github.com/Temple-University-CIS-4496/cis-4496-project-hvac/blob/9da935a5ab17d1edf88714bb62209fd4659eabb0/Docs/Data_Report/DataSummaryReport.md)|

#### Raw Data summary Set1 : This data set is the raw outdoor weather data with data reproting frequency of about every 15 mintutes. The dataset includes the measurments of the environment outside such as outdoor temperature, minimum and maximum temperature, humidity, and timestamps indicating when each observation was recorded. During preprocessig this dataset is merged with the thermostat dataset. 
#### Raw Data summary Set2 : The raw thermostat files include variables such as timestamp, indoor temperature, setpoint, system mode, occupancy state, fan state, and running mode. However, these raw files do not initially contain calculated fields such as cumulative HVAC runtime or cumulative operating cost. These data sets also include a large number of null values wich is adressed during the preprocessing stage. 


## Processed Data Sources

| Processed Dataset Name | Link to the Full Processed Dataset   | Full Processed Dataset Size (MB)  | Link to Report |
| ---:| ---: | ---: | ---: |
| processed_timeseries_table_timeseries_table.csv(1)-(100) | [https://www.kaggle.com/datasets/lsobieski/raw-thermostat-data](https://www.kaggle.com/datasets/lsobieski/processed-thermostat-data) | Each ~15 MB | [Report Link](https://github.com/Temple-University-CIS-4496/cis-4496-project-hvac/blob/9da935a5ab17d1edf88714bb62209fd4659eabb0/Docs/Data_Report/DataSummaryReport.md)|

#### Processed Data summary: The data used for this project is time series thermostat data combined with outdoor weather data. What is included in the outdoor weather data is the timestamp when the data was reported, the outside temperature, the outside temperature minimum, the outside temperature maximum, and outside humidity. This data set was combined with the time series of thermostat data.The information provided from the thermostats are; time stamp, temperature inside, set point, mode, occupied state, fan state output state, running mode, and rental status. Currently there are 100 csv files from 100 different thermostats that have been combined with outdoor weather data. Within those files the cummulative runtime, and cummulative cost were calulcated.
<!--
## Feature Sets

| Feature Set Name | Input Dataset(s)   | Feature Engineering Tools/Scripts | Link to Report |
| ---:| ---: | ---: | ---: | 
| Feature Set 1 | [Dataset1](link/to/dataset1/report), [Processed Dataset2](link/to/dataset2/report) | [R_Script2.R](link/to/R/script/file/in/Code) | [Feature Set1 Report](link/to/report1)|
| Feature Set 2 | [Processed Dataset2](link/to/dataset2/report) |[SQL_Script2.sql](link/to/sql/script/file/in/Code) | [Feature Set2 Report](link/to/report2)|

* Feature Set1 summary. <Provide detailed description of the feature set, such as the meaning of each feature. More detailed information about the feature set should be in the Feature Set1 Report.>
* Feature Set2 summary. <Provide detailed description of the feature set, such as the meaning of each feature. More detailed information about the feature set should be in the Feature Set2 Report.> 
-->
