# Baseline Model Report

_Baseline model is the the model a data scientist would train and evaluate quickly after he/she has the first (preliminary) feature set ready for the machine learning modeling. Through building the baseline model, the data scientist can have a quick assessment of the feasibility of the machine learning task._

## Analytic Approach

The baseline model for this project is a rolling window linear regression. While it is not the most effective time series model, it does offer an advantage over random prediction. Our target variable is the time that the HVAC system runs in a day, and the model factors in the average setpoint of the previous day, the temperature at midnight (indoors and outdoors), as well as the outdoor humidity. These features were chosen from the dataset due to their lack of null values as well as the likelihood that they could assist in predicting the HVAC runtime. A window size of 21 days was chosen for the baseline model, due to the hypothesis that this is a small enough window to train the model on quickly, while hopefully filtering some noise from factors like week-long vaccations, which are unknown to us, but likely have an effect on HVAC runtime due to the house being empty.

## Model Description

The model chosen was a rolling window linear regression using scikit-learn's LinearRegression model. 

Parameters for X: 1 day lagged average setpoint, indoor temperature, outdoor temperature, humidity, all measured at midnight to avoid leakage.

Parameters for Y: The differential in cumulative runtime for each day

Window Size Parameters: 21 days

Hyperparameters: None 

## Results (Model Performance)
* Across 94 households, this model achieves an average R^2 of .36 and a median R^2 of about .5. In addition, the mean absolute error is about 2.3 hours and the median absolute error is about 1.5 hours.
* 6 households were dropped due to outlier R^2 values at the magnitude of -10^25. Further investigation revealed that the data for these households was missing significant chunks of runtime, where the HVAC unit claimed to not be turned on. 
* Graph of R^2 values of 94 houses
<img width="1171" height="830" alt="image" src="https://github.com/user-attachments/assets/a8618a98-b843-4771-b6ba-b04e7e6c7dfb" />


## Model Understanding

* Variable Importance (significance)

* Insight Derived from the Model


## Conclusion and Discussions for Next Steps

Given the results, it seems feasible to model HVAC runtimes with our data. However, more precise modeling is needed. It does not appear that the model is overfitting based on individual csv exploration, though there is a chance that the model is overfitting which is causing the R^2 spikes. In order to create better models, we may need more creative feature generation. One which is interesting could be the creation of an absolute temperature differential from the setpoint, which may assist in determining how hard the HVAC system needs to work. Another interesting feature could be lagged outdoor temperatures, because HVAC systems tend to work harder during continuous heatwaves due to a lack of nighttime cooling. 
