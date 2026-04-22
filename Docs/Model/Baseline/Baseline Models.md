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

## Model Building and Methodology

Prior to starting this model, we looked into multi-day gaps with null values in the data. Due to the lack of data in the temp_gradient_mean and setpoint_gap_mean variables, we chose to fill null values for these categories with 0. After this, and remaining multi-day gaps were dropped, and the largest window remaining was selected for the rolling window linear regression analysis.

To build the model, we select a WINDOW_SIZE and LAG_DAYS parameter. More information on the selection of these is below. We begin by creating our lagged features, which is done by shifting our lag worthy features back for each lag day, creating LAG_DAYS * 20 lagged features. Then we combine the new dataframe of lagged features with our unlagged features, plus the date for reporting purposes only.

Next, we build a set of windows to train the model on. Each window has X_train and X_test sets (len = WINDOW_SIZE) and Y_train and Y_test sets (len = 1) to develop the rolling window model. Once we have built our windows, each window is trained using sklearn's LinearRegression model. The predictions from the model are bounded between 0 and 24 hours, since predicting outside that window does not make sense. 

When training, we add y_true and y_pred to dataframes in order to be able to compute metrics for our models. We compute MSE, RMSE, MAE, MedianAE, MaxAE, MAPE, sMAPE, R^2, Explained Variance, and Pooled R^2. In order to make sure that we don't have outsize influences on our average metrics across the dataset, models which produce an R^2 less than -1 are dropped. However, in the final iterations of our baseline model, these outliers were minimal.

While building this model, different parameters for rolling window sizing and lag_days were tested (window sizing between 3 - 21, lag_days between 1 - 10). We determined that a window size of 7 days, with 7 days of lagged information was the best model based on multiple evaluation criteria, including MAE, R^2, and the number of outliers (R^2 < -1) produced by the model.

After looking at the results for the baseline model, we tested a few more models. The first, referred to as the naive model going forwards, was a simple prediction based on predicting the runtime as the previous day's runtime. Then, we tested two LinearRegression variants from scikit-learn: Ridge, and Lasso. Due to the results of Lasso regression discussed below, we also tested the hyperparameters for the model to determine which was the most effective.

## Results (Model Performance)
Of all baseline models, the naive model performs the best, with an average R^2 of .562 and a pooled R^2 of .704. This model has an average median error of 1.22 hours per house, so it predicts most cases fairly well. There are a few outliers which have an R^2 below 0, but overall we see a normal distribution of R^2 across the houses.

<img width="600" height="400" alt="image" src="https://github.com/user-attachments/assets/2d1f1e04-efb4-40f0-830b-c47039f8b2c9" />

The baseline linear regression model performs worse than the naive model, with an average R^2 of .343 and a pooled R^2 of .539. Looking at the average median error, we see it is 1.47 hours. However, the difference appears when examining the mean absolute error. For the baseline model, it is 2.44 hours compared to the naive model's 1.86 hours. This makes sense, as from a graph of a single house (R^2 = .765), we can see an overreaction to a small spike around the start of May. In the data, this corresponds with a setpoint drop coinciding with a runtime spike, which shows that the model is currently sensitive to rapid changes. The R^2 chart for the baseline model shows a similar curve to the naive model, albeit with more variance and more negative R^2 plots.

<div style="display: flex; gap: 10px; align-items: center;">
  <img src="https://github.com/user-attachments/assets/3776015e-30c1-44d2-a29c-bf7368ad9ba2" style="height: 300px; width: auto;" />
  <img src="https://github.com/user-attachments/assets/74aa3aec-0c4e-4da7-95a3-ef3754b9da48" style="height: 300px; width: auto;" />
</div>

When investigating a Ridge model, we saw similar performace to the baseline in terms of R^2 and pooled R^2. However, the Lasso regression model shows a clear improvement over the baseline linear model, achieving an average R² of .515 and a pooled R² of .652. While the median absolute error remains the same at 1.47 hours, the mean absolute error is reduced to 2.12 hours, indicating better handling of larger errors compared to the baseline model. This suggests that the Lasso model is more robust to extreme runtime spikes, likely due to its ability to eliminate less informative or noisy features. By enforcing sparsity, the model avoids overreacting to short-term fluctuationss, resulting in more stable predictions. The overall distribution of R² scores also shows fewer negative values, reflecting improved generalization and reduced sensitivity to outlier behavior.

<img width="600" height="400" alt="image" src="https://github.com/user-attachments/assets/eafea528-2e78-4f54-a1da-4df414f5e804" />


## Model Understanding



## Conclusion and Discussions for Next Steps

At this stage, the Lasso model provides the strongest performance among the linear approaches, improving over both the baseline and Ridge models with a pooled R² of 0.652 and an average R² of 0.515, alongside a reduced mean absolute error of 2.12 hours. However, despite these gains, it still remains approximately 0.05 behind the naive baseline in both pooled and average R², highlighting a gap in capturing the system’s short-term dynamics. While Lasso improves robustness to extreme runtime spikes through feature selection, the median absolute error remains unchanged at 1.47 hours. Overall, this suggests that further improvements are unlikely to come from additional linear regularization alone, and instead may require more feature engineering to capture short-term trends. As a result, the focus has shifted toward tree-based models, which may better capture interactions and changes in the data and potentially close the remaining gap to the naive benchmark.
