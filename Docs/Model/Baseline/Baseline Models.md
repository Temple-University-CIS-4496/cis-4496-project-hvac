# Baseline Model Report



## Analytic Approach

The baseline model for this project is a rolling window linear regression. While it is not the most effective time series model, it does offer an advantage over random prediction. Our target variable is the time that the HVAC system runs in a day, and the model factors in the average setpoint of the previous day, the temperature at midnight (indoors and outdoors), as well as the outdoor humidity. These features were chosen from the dataset due to their lack of null values as well as the likelihood that they could assist in predicting the HVAC runtime. A window size of 21 days was chosen for the baseline model, due to the hypothesis that this is a small enough window to train the model on quickly, while hopefully filtering some noise from factors like week-long vaccations, which are unknown to us, but likely have an effect on HVAC runtime due to the house being empty.

## Model Description

The model chosen was a rolling window linear regression using scikit-learn's LinearRegression model. 

Features Utilized for X: 
- 1 day lagged average setpoint
- First indoor temperature reading of the day
- First outdoor temperature reading of the day
- First outdoor humidity reading of the day

Parameters for Y: The differential in cumulative runtime for each day

Window Size Parameters: 1 (naive case), 7, 14, 21, 28, and 35 days

Hyperparameters: None 

## Methodology
While building the model, the window sizes listed above were tested, as well as others. The other window sizes included 3, 11, and 17 days, which represent midpoints in the above windows. They exhibited consistent linear trends with the non-naive windows. 

In order to represent the available features from the dataset, minimal feature engineering was done. The engineered features were chosen to prevent data leakage and represent the irregular timeseries data in a regular fashion. In order to obtain accurate Y values, the runtime is derived from the running_mode feature in the raw data. The difference in run time at the first and last observation is computed, as well as whether the thermostat was running from the final observation until midnight.


## Results (Model Performance)
* The naive model performs the best, with an R^2 of .59, and a RMSE of 1.19e4, which is about 3 hours. However, looking at the MAE and Median AE, which lower the impact of outliers, we see that these values are 7.22e3 and 4.46e3 seconds respectively (2 hours and 1.2 hours, respectively).
* For non-naive models, the R^2 increases until stabilizing around a window of 21 days, with the R^2 for window sizes 21, 28, and 35 ranging between .37 and .38.
* The model filters households with an R^2 value below -5. Typically, these dropped households have outlier R^2 values at the magnitude of -10^25. The naive model dropped 0 houses, while the window sizes of 21, 28, and 35 days dropped 6, 4, and 2 houses respectively. Further investigation revealed that the data for these households was missing significant chunks of runtime, where the HVAC unit claimed to not be turned on, or the inverse case where large jumps in runtime were seen due to the HVAC unit being on during the data gaps.
* The graphs for the R^2 values for larger windows imply a median R^2 higher than the mean, with some left tail lowering the mean. It appears the median R^2 is around .5, which increases the feasibility of this task.
* Graph of R^2 values of 94 houses for a rolling window size of 21 days
<img width="985" height="728" alt="image" src="https://github.com/user-attachments/assets/dfcffbe5-7e87-4ea5-9d10-6950093df7e9" />



## Model Understanding

From individual model performance on the csv processed_timeseries_data (10), we can tell that the outdoor temperature is the most important variable to how much the HVAC runs, having a high positive correllation. We also see a meedium sized positive correllation for the outdoor humidity. The setpoint and indoor temperatures have a small negative correllation, which makes sense because the majority of this file's data is in the summer. In addition, we expect the variables to have less impact because they have much less variance than outdoor temperatures. In order to perform a better analysis of coefficient trends across all the files, a feature to differentiate indoor and outdoor temperature could be created to explore the impact on seasonality that investigating files which may have peak runtimes in different seasons would have.

## Conclusion and Discussions for Next Steps

Given the results, it seems feasible to model HVAC runtimes with our data. However, more precise modeling is needed. It does not appear that the model is overfitting based on individual csv exploration, though there is a chance that the model is overfitting which is causing the R^2 spikes. In order to create better models, we may need more creative feature generation. One which is interesting could be the creation of an absolute temperature differential from the setpoint, which may assist in determining how hard the HVAC system needs to work. Another interesting feature could be lagged outdoor temperatures, because HVAC systems tend to work harder during continuous heatwaves due to a lack of nighttime cooling. 
