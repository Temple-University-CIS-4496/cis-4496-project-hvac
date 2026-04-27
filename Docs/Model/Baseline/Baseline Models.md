# Baseline Model Report

## Analytic Approach

The baseline model for this project is a rolling window linear regression. While it is not the most effective time series model, it does offer an advantage over random prediction. For this model, we are predicting the daily runtime of the HVAC system using 26 variables chosen by mutual information score with our target variable. Of those variables, 20 are selected as eligible for becoming lagged features, while 6 are mutually exclusive to the day and are not lagged. We then determine an appropriate window size and amount of days to lag features for. 

## Feature Set

Lagged Features: These features are lagged for the number of LAG_DAYS
<ul style="padding-left: 40px;" id="n5x2zq">
<li>daily_heating_hours - The number of hours during the day that the HVAC system is actively heating</li>

<li>daily_cooling_hours - The number of hours during the day that the HVAC system is actively cooling</li>

<li>setpoint_time_weighted_mean - The time-weighted average setpoint temperature</li>

<li>daily_runtime_hours - The total number of hours the HVAC system is actively running (target variable)</li>

<li>true_outside_mean - The average outdoor temperature over the day based on the true recorded values</li>

<li>true_humidity_mean - The average outdoor humidity level over the day</li>

</ul>

Unlagged Features: These features are not suitable to be lagged due to their nature. 

<ul style="padding-left: 40px;" id="n5x2zq">

<li>month - The numerical month of the year (1–12), used to capture seasonal effects</li>

<li>outdoor_temp_trend_gradient - The overall trend (slope) of outdoor temperature change over the last 3 days</li>

</ul>

Target Variable: Daily Runtime 

Features are selected based on their mutual information score with the target variable (see preprocessing notebook). Then, we choose the highest mutual information features for each category (temperatures, runtimes, etc.) which perform some sort of averaging function.

## Model Building and Methodology
### Rolling Window Methodology

Rolling window models are commonly used in time series problems where temporal structure is important. In many real-world settings, older observations may become less relevant for predicting near-future values. This is particularly true in domains such as weather forecasting, stock markets, and HVAC system behavior.

To address this, the rolling window approach trains a model on a fixed-size window of historical data and then iteratively shifts forward through time to generate predictions.

<img width="600" height="400" alt="Rolling Window Diagram" src="https://github.com/user-attachments/assets/3de1b08c-6d49-421d-a1f6-07784e9d5ada" />

**Figure 1: Schematic of Rolling Window Evaluation.**  
*Source: Macedo et al. (2022), A Machine Learning Approach for Spare Parts Lifetime Estimation, Conference Paper, ResearchGate*

---

### Model Specific Preprocessing

Before modeling, we analyzed the dataset for missing values and multi-day gaps. Due to sparsity in the `temp_gradient_mean` and `setpoint_gap_mean` variables, missing values in these features were imputed with 0.

After imputation, any remaining multi-day gaps in the time series were removed. We then retained the largest continuous time segment to ensure consistency in the rolling window analysis.

---

### Feature Engineering and Lag Construction

The model uses two key hyperparameters:

- **WINDOW_SIZE**: number of past days used for training
- **LAG_DAYS**: number of lagged feature steps included

Lagged features are constructed by shifting selected variables backward in time. For each lag step, this produces a new set of features, resulting in:

- (number of lagged features) = LAG_DAYS × (number of base features)


These lagged features are then combined with the original (non-lagged) features and the date column (used only for tracking predictions).

---

### Rolling Window Training Procedure

The dataset is then transformed into overlapping training windows:

- **X_train**: previous `WINDOW_SIZE` days of features  
- **y_train**: corresponding target values  
- **X_test**: the next single day  
- **y_test**: actual target for that day  

Each window is trained independently using a linear regression model (`sklearn.linear_model.LinearRegression`).

Predictions are constrained to the range **[0, 24] hours**, as values outside this range are not physically meaningful for daily runtime.

---

### Evaluation Metrics

For each prediction window, we store both true and predicted values and compute standard regression metrics:

- Mean Squared Error (MSE)  
- Root Mean Squared Error (RMSE)  
- Mean Absolute Error (MAE)  
- Median Absolute Error (MedianAE)  
- Maximum Absolute Error (MaxAE)  
- Mean Absolute Percentage Error (MAPE)  
- Symmetric MAPE (sMAPE)  
- R² Score  
- Explained Variance Score  
- Pooled R² (computed across all predictions)

To reduce the influence of unstable models, we exclude cases where the model produces an R² score below -1. These cases were rare in the final baseline configuration.

---

### Hyperparameter Selection

We evaluated multiple configurations of:

- WINDOW_SIZE: 3 to 28 days  
- LAG_DAYS: 1 to 10 days  

Based on MAE, R², and model stability (including the frequency of extreme outliers), the best-performing configuration was:

- **WINDOW_SIZE = 28**
- **LAG_DAYS = 1**

---

### Additional Models

After establishing the baseline model, we evaluated additional approaches for comparison:

- **Naive model**: predicts the current day’s runtime as the previous day’s runtime  
- **Ridge regression** (linear model with L2 regularization)  
- **Lasso regression** (linear model with L1 regularization)

Due to the observed performance of Lasso regression, additional hyperparameter tuning was conducted to evaluate its sensitivity and feature selection behavior. The optimal regularization strength was found to be α = 0.45.



## Results (Model Performance)

Across all models, performance was evaluated using both per-house averages and pooled metrics. All models use a 28-day window size and a 1-day lag structure. The naive model excludes the first 28 days of data to ensure a consistent evaluation period across methods.

---

### Naive Model

The naive model performs strongly, achieving an average R² of **0.552** and a pooled R² of **0.696**. This indicates that a large portion of HVAC runtime behavior can be explained by simple temporal persistence (i.e., yesterday’s value is a strong predictor of today’s value).

Error metrics are relatively low:
- MAE: **1.90 hours**
- Median AE: **1.27 hours**
- Max AE: **11.68 hours**

While most predictions are accurate, occasional large errors indicate that the model struggles during abrupt system changes.

<img width="600" height="400" alt="image" src="https://github.com/user-attachments/assets/8936721b-1a83-4633-a283-5153d5aa2191" />

---

### Baseline Linear Regression Model

The baseline linear regression model performs worse than the naive model, with:
- Average R²: **0.476**
- Pooled R²: **0.633**

Error metrics also increase:
- MAE: **2.09 hours**
- Median AE: **1.42 hours**
- Max AE: **14.59 hours**

This suggests that the linear model is not able to consistently improve upon simple persistence-based prediction. It appears sensitive to short-term fluctuations in input features, which can lead to overreaction during abrupt changes such as setpoint shifts or weather variability.

<div style="display: flex; gap: 10px; align-items: center;">
  <img src="https://github.com/user-attachments/assets/44b6b44f-c2c4-4a2f-8756-e6d0eb2a6ce0" style="height: 300px; width: auto;" />
  <img src="https://github.com/user-attachments/assets/b884d782-1848-4429-ace5-233ae28398dd" style="height: 300px; width: auto;" />
</div>


---

### Lasso Regression Model

The Lasso regression model performs best among the linear approaches, achieving:
- Average R²: **0.580**
- Pooled R²: **0.706**

Error metrics:
- MAE: **1.92 hours**
- Median AE: **1.35 hours**
- Max AE: **12.32 hours**

While median error remains similar across linear models, Lasso reduces larger errors and improves stability. This suggests better robustness to noisy or redundant features.

The improvement is likely driven by Lasso’s regularization, which enforces sparsity and reduces sensitivity to irrelevant inputs.

<div style="display: flex; gap: 10px; align-items: center;">
  <img src="https://github.com/user-attachments/assets/4004619e-363f-4d62-ad52-b9621363293c" style="height: 300px; width: auto;"/>
  <img src="https://github.com/user-attachments/assets/20d854cc-2367-494c-a5e7-05d246dbb76d" style="height: 300px; width: auto;"/>
</div>


---

### Summary Comparison

Overall:
- The **naive model** is very strong due to high temporal autocorrelation in HVAC runtime.
- The **baseline linear model** underperforms relative to naive persistence.
- The **Lasso model** provides the best linear-model performance, slightly surpassing naive in pooled R² while improving robustness to extreme errors.


## Model Understanding

### Lasso Model Interpretation

The Lasso regression model provides insight into which factors are most consistently useful for predicting HVAC runtime, while also reducing the influence of redundant or noisy features through regularization.

Across the selected feature set, the strongest predictors are dominated by a combination of prior runtime behavior and outdoor environmental conditions.

<div style="display: flex; gap: 10px; align-items: center;">
  <img src="https://github.com/user-attachments/assets/d9cfbd45-206d-4e7e-b4b8-78181332d3ef" style="height: 300px; width: 45%;" />
  <img src="https://github.com/user-attachments/assets/fec58123-75a8-4bcc-b6de-271581d922ac" style="height: 300px; width: 45%;" />
</div>

### Dominant Predictive Signals

The most influential feature is the lagged outdoor temperature signal (`true_outside_mean_lag_1`), which has both a high coefficient magnitude and the highest selection frequency. This indicates that outdoor conditions from the previous day are one of the most stable drivers of HVAC runtime.

Similarly, `daily_cooling_hours_lag_1` and `daily_heating_hours_lag_1` are among the most important predictors. This suggests strong persistence in system behavior: cooling and heating demands tend to carry over from one day to the next rather than changing abruptly. The inclusion of `daily_runtime_hours_lag_1` confirms that HVAC runtime is highly autocorrelated, meaning yesterday’s runtime is one of the strongest predictors of today’s runtime—even in a multivariate model.

### Temporal and Seasonal Effects

The `month` feature also appears consistently, indicating that there is a seasonal structure in HVAC usage patterns that the model captures even after accounting for temperature-related variables.

This aligns with expected behavior, as HVAC demand is strongly influenced by seasonal weather cycles rather than purely short-term fluctuations.

### Setpoint and Control-Driven Behavior

Setpoint-related variables (e.g., `setpoint_time_weighted_mean_lag_1`) contribute meaningfully but less consistently. Their moderate selection frequency suggests that control system settings do influence runtime, but their effect is secondary compared to environmental conditions and prior system usage.

### Humidity and Secondary Weather Effects

Humidity (`true_humidity_mean_lag_1`) and temperature trend features (`outdoor_temp_trend_gradient`) show moderate importance. These variables likely capture comfort-related adjustments in HVAC behavior but are less dominant than temperature and runtime history.

### Overall Interpretation

The Lasso model effectively performs feature selection that aligns with physical intuition:

- **Primary drivers:** prior runtime + outdoor temperature  
- **Secondary drivers:** seasonal effects (month), humidity, and control settings  
- **Behavioral pattern:** strong temporal persistence in HVAC usage

This explains why Lasso improves performance over standard linear regression. Lasso regression removes unstable or redundant features while retaining the core temporal and environmental structure of the system.

## Conclusion and Discussions for Next Steps

At this stage, the Lasso model provides the strongest performance among the linear approaches, improving over both baseline model with a pooled R² of 0.706 and an average R² of 0.580, alongside a reduced mean absolute error of 1.92 hours. However, despite these gains, it now performs roughly on par with the naive baseline in both pooled and average R², highlighting a gap in capturing the system’s short-term dynamics. While Lasso improves robustness to extreme runtime spikes through feature selection, the median absolute error remains largely unchanged at 1.35 hours. Overall, this suggests that further improvements are unlikely to come from additional linear regularization alone, and instead may require more feature engineering to capture short-term trends. As a result, the focus has shifted toward tree-based models, which may better capture interactions and changes in the data and potentially close the remaining gap to the naive benchmark.
