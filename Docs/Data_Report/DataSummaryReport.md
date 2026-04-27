# Data Report 

## General summary of the data 

There are multiple data sets that are used in this project, the first of which is time series thermostat data, and the second data set is the outdoor weather data.  The outdoor weather data includes a timestamp when the data was reported, the outside temperature, the outside temperature minimum, the outside temperature maximum, and outside humidity.  This data set was combined with the time series of thermostat data. Which includes the following features; time stamp, temperature inside, set point, mode, occupied state, fan state output state, running mode, and rental status. There are 100 csv files from 100 different thermostats that have been combined with outdoor weather data.  

#### Private Kaggle Data Set: https://www.kaggle.com/datasets/lsobieski/raw-thermostat-data 

#### Figure 1: Features from the Raw Dataset
<div style="display: flex; gap: 10px; align-items: center;">
 <img style="height: 150px; width: 56%; alt="Screenshot 2026-04-27 at 4 20 31 PM" src="https://github.com/user-attachments/assets/7d4e5ab9-fc2f-43dd-bff2-67f2e03af579" />  
 <img style="height: 150px; width: 42%;" alt="Screenshot 2026-04-27 at 4 22 23 PM" src="https://github.com/user-attachments/assets/750c0a24-4079-47ec-b15a-a7e633a694d6" />
</div>

#### Figure 2: Additional Features from the Raw Dataset 
 <img width="600" height="150" alt="Screenshot 2026-04-27 at 4 19 52 PM" src="https://github.com/user-attachments/assets/8a80e2aa-1b6c-4f81-aacf-e043242a90fd" />

Figure 1 and 2 show an example of the data distribution from one thermostat before preprocessing. It shows that majority of the data besides set point and timestamp consist of null values and that the set point is between 18.3 degrees Celsius and 25.6 degrees Celsius. 

The time period used in the combined data is from October 1st, 2024 to January 27th, 2026. The sampling frequency for the outdoor weather data is every 15 minutes, while the sampling frequency of the thermostat data is recorded at irregluar, event-driven intervals. This means that there is no fixed sampling frequency for the thermostat measurements. 

To support predictive modeling, the raw time series data was aggregated into a daily-level dataset with engineered features capturing environmental conditions, system behavior, and temporal structure. This transformation is necessary because HVAC systems operate continuously but respond to both short-term fluctuations and longer-term seasonal patterns. 

## Data Quality Summary 

Before the processing of the raw data, several data quality issues were identified. Those issues include a high percentage of null data points in the thermostat time series data, occasional gaps in the indoor timestamps, the synchronization of ourdoor and indoor data sets, and the lack of a cumulative runtime column in the dataset.  

First, to clean up the data, the thermostat time series data column names were standardized to follow a similar naming convention. This allowed all additional engineered variables to follow a similar naming pattern.  

#### Figure 3: From the Raw Kaggle Dataset  
<img width="631" height="252" alt="Screenshot 2026-04-27 at 4 22 57 PM" src="https://github.com/user-attachments/assets/38e76084-50a0-40e1-b39b-745334954df3" />
<img width="631" height="190" alt="Screenshot 2026-04-27 at 4 22 35 PM" src="https://github.com/user-attachments/assets/6ab2d4b2-458e-4ae1-8288-a43ab258d1a6" />

We then addressed the missing values in our dataset. As seen in Figure 3, the mode, occupied, and fan state variables primarily have null values. This is the case because if the thermostat records a data point and the fan state does not change modes, it will record a null value until the mode changes again. We chose to  address this using forward filling, which allowed for the imputation of missing observations in a way that preserved temporal continuity. This approach ensured that key variables such as fan state, running mode, setpoint temperature, and indoor temperature maintained complete and continuous time series without introducing artificial variability. Forward filling was applied to preserve temporal continuity in state variables such as fan status, running mode, and setpoint temperature, under the assumption that HVAC states persist between recorded changes. 

-----

After this, the running mode was normalized, then virtual rows were injected. Since the thermostat timestamps are not heartbeat data with consistent time intervals, if there is a time gap that is more than 30 minutes, there is an unknown row that is inserted. Examples of these time gaps can be seen in figures 4 and figures 5. Figure 4 shows a thermostat that has time gabs up most of them are concentrated between the 0-hour mark and the 5-hour mark. However, there are about 5 outliers that have larger gaps with more than 500-hour gaps. This is different to figure 5 where the highest gap is 24 hours. Showing that figure 4 has significant gaps in their thermostat data which could cause issues with the model's accuracy.   

#### Figure 4: Time gaps from thermostat 1 

 <img width="631" height="252" alt="Screenshot 2026-04-27 at 4 23 08 PM" src="https://github.com/user-attachments/assets/969e046d-1b69-48e7-be52-31b483d6725c" />

#### Figure 5: Time Gaps from thermostat 100  
<img width="631" height="252" alt="Screenshot 2026-04-27 at 4 23 19 PM" src="https://github.com/user-attachments/assets/c0613dc2-495a-46c7-9abd-ed5bdcee6b9a" />


To address the fact that outdoor weather data was stored separately from the thermostat datasets, a multi-step merging process was implemented. First, an outer join was used to preserve all available timestamps from both datasets, ensuring that no potential observations were lost during integration. Finally, a left join was used to map the enriched weather data back to each individual thermostat dataset, ensuring that every thermostat observation had a corresponding set of aligned environmental variables. Virtual rows were inserted to handle irregular timestamp gaps greater than 30 minutes, ensuring consistent temporal spacing across observations. This logic includes a specific numeric reset for the setpoint, acting as a crucial "kill-switch" to terminate forward-fill persistence and prevent stale targets from influencing post-blackout predictions. Additionally, the midnight-slicing ensures that runtime is accounted for in the correct calendar day, even for intervals spanning across 00:00:00. 

Comprehensive feature engineering further enhances this pipeline by computing Bessel-corrected weighted variance and weighted skewness/kurtosis, providing the model visibility into thermal environment volatility rather than just simple central tendencies. 

To address the absence of key engineered variables such as cumulative runtime and energy cost, additional feature construction was performed. Cumulative HVAC runtime was calculated by identifying all time intervals in which the system was actively running (either heating or cooling) and summing up these active intervals over time. This allowed for a continuous measure of system usage intensity across each thermostat. Once cumulative runtime was derived, it was converted into a standardized energy usage metric by transforming runtime from seconds into hours through division by 3600. This normalization step was necessary to align the runtime units with standard energy consumption calculations. 

A key challenge throughout this entire process was ensuring proper temporal alignment between datasets that were not originally synchronized. Because thermostat data is recorded at irregular intervals and often depends on device-specific reporting behavior, while weather data is structured at fixed 15-minute intervals, careful interpolation and alignment strategies were required to avoid introducing bias. Without proper alignment, there is a risk that weather conditions could be incorrectly associated with HVAC states, leading to misleading patterns during model training and evaluation. 

Additionally, multiple data consistency checks were performed after the merging process to ensure the integrity of the final dataset. These checks included verifying that no duplicate timestamps existed within individual thermostat time series, ensuring that all time series remained strictly ordered, and confirming that no missing intervals were introduced during the join operations. Each thermostat dataset was also validated to ensure continuity over time, meaning that the resulting structure represented a coherent and logically consistent sequence of observations suitable for downstream modeling. 

Overall, these preprocessing steps transformed the raw, heterogeneous time-series data into a clean, structured, and analysis-ready dataset capable of supporting reliable feature engineering and predictive modeling of HVAC runtime behavior. 

#### Private Processed Data Sets: https://www.kaggle.com/datasets/lsobieski/raw-thermostat-data  

## Target variable 

The target variable in this analysis is HVAC runtime, defined as the total time an HVAC system operates (heating or cooling) during a given time interval. The target variable is defined as daily runtime hours, representing the total number of hours per day that the HVAC system is actively running. This variable is critical for estimating energy consumption and understanding how environmental conditions and user behavior influence HVAC usage. This is because the only time the HVAC unit uses energy is when it is running. When it is off, it will not use any energy.  

From a modeling perspective, this target is continuous and time-dependent, meaning it is influenced not only by current-day conditions but also by prior system behavior and environmental inertia. Figures 6 and 7 show the daily run time from two thermostats. Which shows that it is a continuous value, and also shows that the runtime does vary per house.  

#### Figure 6: Daily Runtime from thermostat 1 

 <img width="595" height="239" alt="Screenshot 2026-04-27 at 4 24 16 PM" src="https://github.com/user-attachments/assets/5e06067b-0ba1-4bfe-acdb-c30a347edf44" />

#### Figure 7: Daily runtime from thermostat 100 

 <img width="595" height="239" alt="Screenshot 2026-04-27 at 4 24 25 PM" src="https://github.com/user-attachments/assets/913399c9-3764-47fb-90db-ecfc7459d2f6" />


## Individual variables 

The variables in this dataset can be grouped into four categories: weather variables, thermostat variables, time variables, and engineered features. 

### Weather variables: 
Outdoor temperature and humidity measurements that influence heating and cooling demand. Weather variables are particularly important because HVAC systems are fundamentally reactive systems designed to maintain indoor comfort relative to external environmental conditions. Sudden changes in temperature or humidity often led to increased system activation, especially when indoor setpoints deviate significantly from outdoor conditions.  

### Thermostat variables: 
Indoor temperature, setpoint temperature, occupancy state, and system mode. 

These variables represent user-driven and system-driven control mechanisms. The setpoint temperature reflects user preference, while the occupancy state introduces behavioral context, indicating whether energy-saving modes are likely to be active. System mode (heating, cooling, fan, off) directly encodes operational state transitions. 

### Time variables: 
Timestamp-derived features such as hour of day, day of week, and seasonality (e.g., month). 

Time-based variables capture behavioral cycles that are not directly observable from temperature alone. For example, HVAC usage tends to increase during morning and evening hours due to occupancy patterns and decreases during mid-day or late-night hours when homes are unoccupied or thermally stable. 

### Engineered features (NEW): 
These were created to better capture patterns in HVAC usage and improve predictive performance. 

#### Lagged Features (based on previous LAG_DAYS):  

daily_off_hours, daily_heating_hours, daily_cooling_hours  

temp_gradient_mean (rate of indoor temperature change over time)  

setpoint_gap_mean (difference between indoor temperature and setpoint)  

outdoor temperature statistics (min, max, mean, quartiles, median)  

outdoor_temp_time_weighted_mean (accounts for duration at each temperature)  

outdoor_temp_raw_moment_2 and 3 (capture variance and skewness)  

setpoint statistical features (mean, time-weighted mean, variance, skewness)  

daily_unknown_hours  

daily_runtime_hours (target variable, also used in lagged form for temporal dependence)  

Lagged features are crutial for capturing temporal dependence in HVAC behavior because if one day has an increased runtime due to weather, there is a high likleyhood that the next day has a similar weather pattern. 

#### Unlagged Features:  

true_outside_mean, true_outside_min, true_outside_max  

true_humidity_mean  

month (captures seasonal effects)  

outdoor_temp_trend_gradient (daily temperature trend)  

Unlagged features capture same-day environmental summaries that are not dependent on previous days' data. These variables are particularly useful for real-time prediction scenarios where only current-day information is available. 

## Variable ranking 

The ranking of variables was determined using mutual information scores which is denoted in figure 8 as a score. However current day variables like indoor temperature were excluded to prevent temporal leakage. This shows that Daily Hours Off and Daily Heating hours are the top two highest ranked variables.  

#### Figure 8: Top 25 Base Features (By Mutual Information) 

 <img width="501" height="314" alt="Screenshot 2026-04-27 at 4 25 11 PM" src="https://github.com/user-attachments/assets/a4b6a0ae-7a59-4e17-8801-4fff13e1e949" />

Another key finding for the variable accuracy is that the ACF in figure 9 shows the past runtime remains predictive of current behavior for up to about two weeks. 

#### Figure 9: ACF and CCF  

 
<img width="536" height="493" alt="Screenshot 2026-04-27 at 4 25 30 PM" src="https://github.com/user-attachments/assets/0d7ec2a1-a520-41ae-a58f-c53f451a07f2" />


 

Finally, distribution-based temperature features such as quartiles, skewness, and time-weighted averages show that variability in weather conditions matters just as much as average conditions. Days with unstable or highly fluctuating temperatures tend to produce higher HVAC activity compared to stable temperature days, even when the average temperature is similar. 

 

 

 

 
 

## Relationship between explanatory variables and target Variables 

### Analysis of the relationship between explanatory variables and HVAC runtime revealed clear patterns. 

There are clear seasonal differences between the heating season (winter) and the cooling season (summer). During the heating season, there are large peaks in the early morning, while in the cooling season, there are peaks in the afternoon that trigger AC activation. This shows that when outdoor temperatures increase during the cooling season, the HVAC system must work harder to maintain indoor comfort. 

#### Figure 10

 <img width="647" height="511" alt="Screenshot 2026-04-27 at 4 26 27 PM" src="https://github.com/user-attachments/assets/d9f196b4-78b8-4bac-9d0c-524657cc5968" />

### CCF by Season 

Some notable trends in the data is that the impact of temperature on runtime depends on the season (figure 11): in winter, colder temperatures are associated with increased runtime, while in summer, higher temperatures drive more runtime. When all data is pooled together, these opposing seasonal effects cancel out, making the relationship appear weak or nonexistent. Additionally, differences across individual pieces of equipment indicate that behavior is not uniform, so analyzing each unit separately is important. Overall, this means that accurate modeling should incorporate lagged effects, seasonal context, and account for unit-level variation.  

#### Figure 11: CCF by SeasonRuntime Vs Outdoor Temperature by Season 
<img width="647" height="431" alt="Screenshot 2026-04-27 at 4 26 44 PM" src="https://github.com/user-attachments/assets/9555edd3-bb77-445d-8c3e-1637dfcde4e5" />


Figure 12 illustrates the relationship between HVAC runtime and outdoor temperature across different seasons, highlighting clear seasonal separation in system behavior. The plot shows that runtime responds differently to temperature depending on the time of year, with winter conditions generally exhibiting increased heating-related activity at lower temperatures, while summer conditions show elevated cooling-related runtime as temperatures rise. This seasonal stratification reinforces that temperature effects are not uniform year-round but instead depend strongly on operational context and seasonal demand patterns. 

#### Figure 12: Runtime vs Outdoor Temperature 
<img width="599" height="408" alt="Screenshot 2026-04-27 at 4 26 58 PM" src="https://github.com/user-attachments/assets/9cf99ce8-3265-49ba-aab7-43ccbd277d27" />

#### Figure 13: Average HVAC Active Rate by Month and Season 

 
<img width="630" height="408" alt="Screenshot 2026-04-27 at 4 27 23 PM" src="https://github.com/user-attachments/assets/06cae367-4ee5-4cf5-b95b-b84f5649c6dd" />


 

### Daily Peak Analysis 

Another trend within the data is the peak heating and cooling done by a thermostat. This figure shows that for the heating the peak at 7 am during the morning and for cooking at 6pn at night. These peak heating and cooling times make sense when it is the hottest outside at around 2-3pm. This additionally shows a trend of runtime within the heating and cooling.  

#### Figure 14: Daily Peak Analysis 
<img width="599" height="408" alt="Screenshot 2026-04-27 at 4 27 07 PM" src="https://github.com/user-attachments/assets/e73db3a0-4926-46e1-a361-32168773668c" />

### Proportion of Time spent in each mode 

Another observation is that the HVAC system spends most of its time idle or off, and heating and cooling modes are only activated when necessary. Additionally, heating and cooling systems appear to have similar active workloads, meaning that both modes operate for comparable amounts of time during their respective seasons. 

#### Figure 15: Proportion of Time spent in each mode  

<img width="388" height="363" alt="Screenshot 2026-04-27 at 4 27 42 PM" src="https://github.com/user-attachments/assets/9abed467-c7dc-4bbd-a1de-a5e9c69dc6c0" />


This idle-dominant behavior suggests that HVAC systems operate in a threshold-based control regime rather than continuous operation. This is consistent with real-world thermostat logic, where systems activate only when indoor temperature deviates beyond a set threshold from the desired setpoint. 

### How the Observed Trends Relate to the Project Problem Statement 

The main goal of this project is to predict HVAC runtime based on weather conditions, thermostat behavior, and time-based patterns. The trends observed in the data strongly support this objective and help explain how runtime is actually driven in real systems.  

The observed trends directly support the project’s goal of predicting HVAC runtime from weather conditions, thermostat behavior, and time-based patterns by showing that runtime is strongly structured and explainable rather than random. Across all analyses, outdoor temperature and seasonal context emerge as key drivers of HVAC usage, which confirms that weather-based features are essential predictors for the model. The clear seasonal separation in behavior—where winter is dominated by heating demand and summer by cooling demand—shows that the relationship between temperature and runtime is not constant throughout the year, but changes depending on operational context. This is critical for the problem statement because it demonstrates that accurate prediction requires capturing these seasonal shifts rather than relying on a single global relationship. 

The CCF by season (Figure 11) further strengthens this by showing that temperature impacts runtime in opposite ways depending on the season: colder temperatures increase runtime in winter, while hotter temperatures increase runtime in summer. When all data is combined, these effects cancel out, which would mislead a predictive model into underestimating the importance of temperature. This directly informs the modeling approach by showing why seasonal stratification is necessary for accurate prediction. 

The Runtime vs Outdoor Temperature by Season plot (Figure 12) reinforces the same idea by showing distinct temperature-response patterns across seasons, confirming that HVAC runtime is highly sensitive to environmental conditions in a structured way. The average HVAC active rate (Figure 13) further relates to the problem statement by showing that HVAC operation is not continuous but occurs in demand-driven bursts, meaning runtime prediction must account for periods of activation versus inactivity rather than assuming steady usage. 

Daily peak analysis (Figure 14) connects directly to time-based prediction features in the problem statement by showing predictable daily cycles in HVAC usage, with heating peaks occurring in the early morning and cooling peaks occurring later in the day. This demonstrates that time-of-day is a strong predictor of runtime and should be included in the model alongside weather variables. 

Finally, the proportion of time spent in each mode (Figure 15) shows that HVAC systems spend most of their time idle and only activate when necessary, confirming that runtime is driven by threshold-based control behavior tied to setpoints and environmental triggers. Overall, these trends validate the problem statement by showing that HVAC runtime can be effectively predicted using a combination of weather conditions, seasonal context, and time-based patterns, all of which are clearly reflected in the observed system behavior. 

## Does the Data Foretell Any Issues That May Arise in Later Stages of the Project Lifecycle? 

The data reveals several potential issues that could impact later stages such as modeling, evaluation, and deployment. 

One major issue is high variability across thermostats. Since performance differs significantly between devices, a single global model may not generalize well to all homes. Some thermostats may have unique usage patterns, missing data, or inconsistent reporting behavior, which can reduce model accuracy and lead to negative R² values for certain units. This suggests that a one-size-fits-all model may not be sufficient, and that future work may need to explore per-device modeling or clustering-based approaches.  

Another issue is temporal drift over time. The model performance decreases as the dataset moves further away from the training period, which indicates that HVAC behavior changes over time, especially across seasons. This could become a problem in deployment, where the model may need frequent retraining to remain accurate under changing seasonal conditions. 

The third issue is data imbalance and idle dominance. Since HVAC systems spend most of their time in an off or idle state, the dataset is heavily skewed toward low-activity periods. This can make it harder for the model to accurately learn patterns during active heating or cooling periods, which are actually the most important for energy prediction. 

There is also a potential issue with feature dependency and leakage risk, especially with lagged runtime variables. Because lag features are extremely strong predictors, there is a risk that the model becomes overly reliant on past runtime rather than learning true environmental relationships. While this improves accuracy, it may reduce interpretability and limit generalization to new or unseen systems. 

Finally, there is a risk related to data quality inconsistencies across thermostats, especially for devices with irregular sampling or short histories. These devices may introduce noise into the model and reduce overall stability unless handled separately or filtered during preprocessing. 

Overall, these issues suggest that while the dataset is strong for modeling, careful attention will be needed in future stages to ensure generalization, prevent overfitting to individual devices, and maintain performance across different time periods and operating conditions. 

 

