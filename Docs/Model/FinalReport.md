# Final Model Report
_Report describing the final model to be delivered - typically comprised of one or more of the models built during the life of the project_

## Analytic Approach
* What is target definition
* What are inputs (description)
* What kind of model was built?

## Solution Description
* Simple solution architecture (Data sources, solution components, data flow)
* What is output?

## Data
* Source
* Data Schema
* Sampling
* Selection (dates, segments)
* Stats (counts)

## Features
* List of raw and derived features 
* Importance ranking.

## Algorithm
* Description or images of data flow graph
  * if AzureML, link to:
    * Training experiment
    * Scoring workflow
* What learner(s) were used?
* Learner hyper-parameters

## Results
* ROC/Lift charts, AUC, R^2, MAPE as appropriate
* Performance graphs for parameters sweeps if applicable










# Final Model Report

## Analytical Method

The method used for prediction of the HVAC compressor runtime in our project went through several steps in its development. Initially, the target metric in the project was set as the daily difference in cumulative runtime. Then, in the next step, this target was changed in an intermediate stage to a continuous regression target which corresponded to the active compressor minutes per hour, varying from zero to sixty minutes. Finally, in the final model, the target was set as the total daily runtime. As for the input features, they became more sophisticated in correlation with the evolution of the target metric, starting from simple single daily observations, then going through hourly rolling metrics, and ending with complex time-weighted statistic distributions, continuous calendar grids, and historical lags.

Regarding the modeling frameworks, their evolution included different training paradigms. Initially, we applied rolling window linear regression, locally trained and tested models on individual houses using chronological splits. In the intermediate model, we shifted towards using global XGBoost regressors, not trained locally but globally, using all available thermostats, thus maximizing the amount of training data. Here, we use strict 80/20 chronological splits combined with a time series five-fold cross-validation technique. In our final model, we use an advanced global framework with both LightGBM and CatBoost regressors and evaluate it on a stratified blocked time series split.

## Solution Description

Our solution is aimed at ingesting raw manufacturer indoor thermostat CSVs and outdoor weather data, performing extensive cleaning and temporal alignment procedures, and feeding data to the machine learning pipeline for the purposes of obtaining a continuous numeric prediction of daily HVAC runtime. In the original solution architecture, our data processing was built heavily on top of unbounded forward filling and simple outer joins to artificially prolong the periods of data state persistence, in case of device blackouts. In the redesigned solution architecture, we fixed a same timestamp bursts issue, enforced a strict thirty-minute validity threshold in interpolation, added an automatic injection of virtual expiration rows to break state persistence in case of prolonged periods of missing data. Additionally, we have truncated event intervals by midnight, thus getting a correct daily aggregation. The next step is mapping obtained aggregated features to continuous calendars in order to avoid any temporal leakages before training and scoring of our pipeline. At the output, our solution generates predictions on how many hours HVAC unit will operate on a given day.

## Data

Our initial data source for analysis is a Kaggle dataset consisting of 100 raw indoor thermostat time series files and one outdoor weather data file. Data is stored in event level records format with the description of device states and surrounding environment condition at the time. During this project, various selection and sampling processes have been applied. Firstly, our intermediate models excluded devices which had less than six months worth of data to have enough training samples. Secondarily, during the final models selection, we have mapped data to continuous daily grid and have divided the timeline into fourteen days blocks. Blocks were assigned to meteorological seasons, and we used stratified blocked time series split with 20% blocks of the season being used for testing our model. This way, we managed to prevent temporal leakage through the outdoor weather features as well as having samples from all seasons. Overall, we selected 26,459 rows for training and 15,333 rows for testing.

Before selection of base features, our entire generated dataset included following features:

| Category | Features |
| :--- | :--- |
| **Identifiers and Time** | `Equipment_ID`, `Date`, `day_of_week`, `is_weekend`, `month`, `month_sin`, `month_cos`. |
| **Daily Runtime and States** | `daily_heating_hours`, `daily_cooling_hours`, `daily_off_hours`, `daily_unknown_hours`, `daily_runtime_hours`, `daily_fan_on_hours`, `fan_runtime_ratio`. |
| **Thermostat Behavior** | `setpoint_change_count`, `occupied_ping_count`, `unoccupied_ping_count`. |
| **Time-Weighted Means and Gaps** | `indoor_temp_time_weighted_mean`, `setpoint_time_weighted_mean`, `outdoor_temp_time_weighted_mean`, `outdoor_humidity_time_weighted_mean`, `setpoint_gap_mean`, `temp_gradient_mean`. |
| **Statistical Distributions** (min, max, median, IQR, skewness, variance, moments, etc.) | individually calculated for indoor temperature, setpoint, outdoor temperature and outdoor humidity. |
| **True External Weather** | `true_outside_min`, `true_outside_max`, `true_outside_mean`, `true_humidity_mean`, `outdoor_temp_trend_gradient`. |

## Features

In terms of feature engineering, our solution evolved through simple observations and advanced generation of statistical and time-weighted metrics. Our baseline and intermediate models included simple calculations like calculating difference between indoor temperature and setpoint, along with three-hour rolling metrics of runtime. However, in our final model implementation, we took it up a notch and started to compute time-weighted distribution statistics, including mean, variance, skewness, and kurtosis. In order to eliminate temporal leakage of endogenous state variables, we decided to exclude them completely from current daily features, since these features represent concurrent HVAC operation and cannot be accessed on a prediction date. Mutual information scores helped to select the most informative base features and added historical lags to give us a good behavioral context. Automated evaluation of models using lag windows from one to twenty-one days showed that four-day lag windows maximized the adjusted R-square.

The list of top 25 base features based on mutual information scores are as follows:

| Rank | Feature | Score |
| :--- | :--- | :--- |
| 1 | `daily_off_hours` | 3.1026 |
| 2 | `daily_heating_hours` | 3.0796 |
| 3 | `daily_cooling_hours` | 2.2163 |
| 4 | `temp_gradient_mean` | 0.2483 |
| 5 | `setpoint_gap_mean` | 0.2404 |
| 6 | `true_outside_mean` | 0.2198 |
| 7 | `outdoor_temp_min` | 0.2148 |
| 8 | `true_outside_min` | 0.2131 |
| 9 | `outdoor_temp_raw_moment_3` | 0.2073 |
| 10 | `outdoor_temp_mean` | 0.2049 |
| 11 | `outdoor_temp_time_weighted_mean` | 0.2049 |
| 12 | `true_outside_max` | 0.2014 |
| 13 | `outdoor_temp_max` | 0.2012 |
| 14 | `outdoor_temp_q25` | 0.1986 |
| 15 | `outdoor_temp_median` | 0.1910 |
| 16 | `outdoor_temp_q75` | 0.1866 |
| 17 | `outdoor_temp_raw_moment_2` | 0.1745 |
| 18 | `month` | 0.1178 |
| 19 | `outdoor_temp_trend_gradient` | 0.1150 |
| 20 | `true_humidity_mean` | 0.1069 |
| 21 | `setpoint_raw_moment_2` | 0.1029 |
| 22 | `setpoint_mean` | 0.1025 |
| 23 | `setpoint_raw_moment_3` | 0.1008 |
| 24 | `setpoint_time_weighted_mean` | 0.0989 |
| 25 | `daily_unknown_hours` | 0.0955 |

Top five features used in our LightGBM optimization process with respect to information gain are:

| Rank | Feature | Gain |
| :--- | :--- | :--- |
| 1 | `daily_runtime_hours_lag_1` | 21,551,452.9 |
| 2 | `daily_runtime_hours_lag_2` | 3,361,982.5 |
| 3 | `daily_off_hours_lag_1` | 986,498.3 |
| 4 | `true_outside_mean` | 760,938.6 |
| 5 | `daily_runtime_hours_lag_3` | 465,033.9 |

## Algorithm

Data flow and scoring procedure evolved through three major stages in the algorithm development process. Baseline model used locally trained model for each individual house based on scikit-learn Linear Regression algorithm with rolling windows to obtain a baseline for our models. Intermediate model implemented XGBoost regressor model which predicted hourly active minutes with learning rate of 0.02, maximum depth of 8 and 1000 estimators, limited by early stop mechanism in order to avoid overfitting on chronological hold-out sets. Final implementation used combination of LightGBM and CatBoost regressors on stratified blocked splits of our datasets. Final scoring workflow implemented global LightGBM model and performed thorough lag window sweep, training models from one to twenty-one day lag windows. Optimal lag window was found to be four days with 800 estimators, learning rate of 0.01, 63 leaves, max depth of 8 and 50 early stopping rounds.

Other comparisons included naive persistence models (predict yesterday's runtime) and training-set mean performance. Such a comparison is necessary in order to understand the degree of improvements brought about by our models in the project.

## Results

During the progress of our models, their effectiveness in predicting the HVAC runtime has improved dramatically. Initial models were able to prove the feasibility of predicting HVAC runtime but suffered greatly from outlier impact. Intermediate pooled model confirmed that historical rolling metrics work significantly better than instant weather data and gave results slightly below ASHRAE benchmark criteria. Finally, we have established that properly structured time-weighted global models with optimal lag windows can reach extremely high prediction accuracy. Final LightGBM model proved to be very stable throughout all season blocks and outperform naive persistence baselines by a substantial margin.

**Baseline Model Test Results (Locally Trained, Chronological Split):**

| Metric | Value |
| :--- | :--- |
| Mean R-squared (21-Day Window) | around 0.37–0.38 |
| Root Mean Square Error (1-Day Naive Window) | around 3.0 hours |
| Median Absolute Error (1-Day Naive Window) | around 1.2 hours |

**Intermediate XGBoost Model Test Results (Globally Pooled, Chronological Split):**

| Metric | Value |
| :--- | :--- |
| Mean R-squared | 0.440 |
| Root Mean Square Error | 12.42 minutes per hour |
| Mean Absolute Error | 9.12 minutes per hour |
| Coefficient of Variation | 49.6 percent |

**Final Optimized LightGBM Test Results (Lag 1 to 4 Window, 86 Features):**

| Metric | Value |
| :--- | :--- |
| R-squared | 0.7822 (Adjusted: 0.7810) |
| Root Mean Square Error | 2.6666 hours |
| Mean Absolute Error | 1.8394 hours |

**Additional Evaluations of Final Model:**

| Evaluation | Result |
| :--- | :--- |
| Naive Persistence Baseline (predict yesterday) | RMSE = 3.0567 hours |
| Naive Train Mean Baseline | RMSE = 5.9038 hours |
| Winter Performance | R-squared = 0.7313, RMSE = 3.1284 hours |
| Summer Performance | R-squared = 0.7480, RMSE = 2.6704 hours |
