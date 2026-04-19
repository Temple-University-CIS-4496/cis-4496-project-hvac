# Baseline Model Report



## Analytic Approach

The baseline model for this project is a rolling window linear regression. While it is not the most effective time series model, it does offer an advantage over random prediction. For this model, we are predicting the daily runtime of the HVAC system using 26 variables chosen by mutual information score with our target variable. Of those variables, 20 are selected as eligible for becoming lagged features, while 6 are mutually exclusive to the day and are not lagged. We then determine an appropriate window size and amount of days to lag features for. 

## Feature Set

Lagged Features: These features are lagged for the number of LAG_DAYS
<ul style="padding-left: 40px;" id="n5x2zq">
<li>daily_off_hours - The number of hours during the day that the HVAC system is not running</li>

<li>daily_heating_hours - The number of hours during the day that the HVAC system is actively heating</li>

<li>daily_cooling_hours - The number of hours during the day that the HVAC system is actively cooling</li>

<li>temp_gradient_mean - The average rate of change of indoor temperature over the past 3 days</li>

<li>setpoint_gap_mean - The average difference between the setpoint temperature and the indoor temperature</li>

<li>outdoor_temp_min - The minimum outdoor temperature recorded during the day</li>

<li>outdoor_temp_raw_moment_3 - The third raw statistical moment of outdoor temperature (captures skewness relative to zero)</li>

<li>outdoor_temp_mean - The average outdoor temperature over the day</li>

<li>outdoor_temp_time_weighted_mean - The time-weighted average outdoor temperature, accounting for duration at each value</li>

<li>outdoor_temp_max - The maximum outdoor temperature recorded during the day</li>

<li>outdoor_temp_q25 - The 25th percentile (lower quartile) of outdoor temperature</li>

<li>outdoor_temp_median - The median (50th percentile) outdoor temperature</li>

<li>outdoor_temp_q75 - The 75th percentile (upper quartile) of outdoor temperature</li>

<li>outdoor_temp_raw_moment_2 - The second raw statistical moment of outdoor temperature (related to variance, but not centered)</li>

<li>setpoint_raw_moment_2 - The second raw statistical moment of the setpoint temperature</li>

<li>setpoint_mean - The average thermostat setpoint temperature over the day</li>

<li>setpoint_raw_moment_3 - The third raw statistical moment of the setpoint temperature (captures skewness)</li>

<li>setpoint_time_weighted_mean - The time-weighted average setpoint temperature</li>

<li>daily_unknown_hours - The number of hours where the HVAC system state is unknown or unclassified</li>

<li>daily_runtime_hours - The total number of hours the HVAC system is actively running (target variable)</li>
</ul>

Unlagged Features: These features are not suitable to be lagged due to their nature. 

<ul style="padding-left: 40px;" id="n5x2zq">
<li>true_outside_mean - The average outdoor temperature over the day based on the true recorded values</li>

<li>true_outside_min - The minimum outdoor temperature recorded during the day</li>

<li>true_outside_max - The maximum outdoor temperature recorded during the day</li>

<li>month - The numerical month of the year (1–12), used to capture seasonal effects</li>

<li>outdoor_temp_trend_gradient - The overall trend (slope) of outdoor temperature change throughout the day</li>

<li>true_humidity_mean - The average outdoor humidity level over the day</li>
</ul>

Target Variable: Daily Runtime 

All features are selected based on their mutual information score with the target variable

## Model Building

While building this model, different parameters for rolling window sizing and lag_days were tested (window sizing between 3 - 21, lag_days between 1 - 10). We determined that a window size of 7 days, with 7 days of lagged information was the best model based on multiple evaluation criteria, including MAE and R^2.

### In Progress Still

## Methodology
While building the model, the window sizes listed above were tested, as well as others. The other window sizes included 3, 11, and 17 days, which represent midpoints in the above windows. They exhibited consistent linear trends with the non-naive windows. 

In order to represent the available features from the dataset, minimal feature engineering was done. The engineered features were chosen to prevent data leakage and represent the irregular timeseries data in a regular fashion. In order to obtain accurate Y values, the runtime is derived from the running_mode feature in the raw data. The difference in run time at the first and last observation is computed, as well as whether the thermostat was running from the final observation until midnight.


## Results (Model Performance)
* The naive model performs the best, with an R^2 of .59, and a RMSE of 1.19e4, which is about 3 hours. However, looking at the MAE and Median AE, which lower the impact of outliers, we see that these values are 7.22e3 and 4.46e3 seconds respectively (2 hours and 1.2 hours, respectively).
* For non-naive models, the R^2 increases until stabilizing around a window of 21 days, with the R^2 for window sizes 21, 28, and 35 ranging between .37 and .38.
* The model filters households with an R^2 value below -5. Typically, these dropped households have outlier R^2 values at the magnitude of -10^25. The naive model dropped 0 houses, while the window sizes of 21, 28, and 35 days dropped 6, 4, and 2 houses respectively. Further investigation revealed that the data for these households was missing significant chunks of runtime, where the HVAC unit claimed to not be turned on, or the inverse case where large jumps in runtime were seen due to the HVAC unit being on during the data gaps.
* The graphs for the R^2 values for larger windows imply a median R^2 higher than the mean, with some left tail lowering the mean. It appears the median R^2 is around .5, which increases the feasibility of this task.
* Graph of R^2 values of 94 houses for a rolling window size of 21 days
<img width="400" height="300" alt="image" src="https://github.com/user-attachments/assets/dfcffbe5-7e87-4ea5-9d10-6950093df7e9" />



## Model Understanding

From individual model performance on the csv processed_timeseries_data (10), we can tell that the outdoor temperature is the most important variable to how much the HVAC runs, having a high positive correllation. We also see a meedium sized positive correllation for the outdoor humidity. The setpoint and indoor temperatures have a small negative correllation, which makes sense because the majority of this file's data is in the summer. In addition, we expect the variables to have less impact because they have much less variance than outdoor temperatures. In order to perform a better analysis of coefficient trends across all the files, a feature to differentiate indoor and outdoor temperature could be created to explore the impact on seasonality that investigating files which may have peak runtimes in different seasons would have.

## Conclusion and Discussions for Next Steps

Given the results, it seems feasible to model HVAC runtimes with our data. However, more precise modeling is needed. It does not appear that the model is overfitting based on individual csv exploration, though there is a chance that the model is overfitting which is causing the R^2 spikes. In order to create better models, we may need more creative feature generation. One which is interesting could be the creation of an absolute temperature differential from the setpoint, which may assist in determining how hard the HVAC system needs to work. Another interesting feature could be lagged outdoor temperatures, because HVAC systems tend to work harder during continuous heatwaves due to a lack of nighttime cooling. 
