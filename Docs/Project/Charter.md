# Project Charter

## Project Charter: HVAC Runtime Prediction & Energy Optimization 

## Business Background 

Telkonet Inc. A provider of energy management systems; however, many rental homes still operate HVAC systems inefficacy due to static thermostat settings and limited responsiveness to changing environmental conditions. Most issues come from thermostats that stick to fixed settings. These systems do not account for how long a building retains heat or cooling resulting in significant energy waste and operating costs. The U.S. Department of Energy found that heating and cooling make up 45% of all residential energy use (Center for Sustainable Systems, 2024). Traditional thermostat control strategies do not account for dynamic environmental conditions or building thermal behavior. However, smarter systems like Model Predictive Control have shown that it is possible to save about 40% of that energy (Serale et al., 2018). This reveals a significant gap between current thermostat strategies and achievable efficiency through predictive control methods.  

The proposed solution is a counterfactual energy analysis framework that evaluates how HVAC runtime would change under different thermostat settings. This is a system that helps property managers and residents shift from reactive energy management towards predictive energy optimization. We want to show people the actual financial value of energy-efficient habits. The best way to do that is by illustrating the cost difference between actual runtimes and simulated runtimes under better settings. This makes it much easier to make data-driven choices without making the people inside the building uncomfortable. 

## Scope & Data Science Methodology 

The proposed solution relies on a structured data processing pipeline capable of handling telemetry collected from around 1,000 thermostats over the last two years. This data contains sparse time-series observations, as the thermostat only records new values when the temperature changes, or the system's state changes. As a result, the data does not align naturally with external data sets, such as hourly weather observations. 

To address this limitation, the data must be preprocessed to clean up inconsistencies and to synchronize timestamps between thermostat logs and outdoor weather data. Since thermostat measurements occur at irregular intervals, alignment methods are required to create a consistent time series. Exploratory data analysis (EDA) is used to evaluate interpolation strategies, for aligning data sets and handling missing values. The step ensures that the outside weather and thermostat data can be analyzed on the same timeline.   

Moreover, HVAC runtime is calculated by analyzing output status signals that indicate when heating or cooling is actively running.  Once that data is ready, it feeds into a predictive modeling framework. A smaller subset of data is used to evaluate different modeling approaches. Decision tree-based models are evaluated because they perform well with structured tabular data and provide interpretable relationships between weather variables and HVAC runtime. A naive model is implemented, which will predict the runtime behavior based on the runtime of the previous day and serve as a baseline comparison with more complex machine learning models. We also evaluate deep learning methods such as LSTMs or Transformers to determine whether sequence-based models would improve prediction of temporal HVAC usage patterns. These models must remain robust enough to handle sudden weather shifts, especially sharp spikes in usage that happen when the temperature outside drops fast. 

The primary output of this project is a comparison engine for What-If simulations for evaluating HVAC runtime under alternative thermostat settings. The model is fed two separate inputs: the real weather with real settings, and then that same weather with optimized settings. After that, the system calculates the difference in predicted runtime. Property managers can use this as a monthly auditing tool to monitor performance and identify inefficiencies. It enables them to quickly pinpoint high-consumption units that may require attention and implement corrective measures. 

## Personnel 

The project is handled by a team of data scientists who work closely with the project lead and client contacts. The project team is organized to hit every technical milestone, so the team is organized to handle everything from the first data pull to the final dashboard. The core team consists of Zack Aidarov, Ayush Gupta, Lauren Sobieski, Max Suc, Andrew Coffman, and Dylan Heathcote. 

The work is still divided into three main areas to keep things organized: 

Data Engineering: This group focuses on the backend infrastructure; they are the ones building the pipeline that cleans the logs and compresses the data to run more efficiently. The team will ensure that the large CSV files are valid as well as keeping the interpolation logic working. 

Modeling & Simulation: This part of the project is all about the core data science problems. This team develops and tests the algorithms to predict runtime and figure out the savings. The team will conduct most of the testing on both tree-based and deep learning models to find the most accurate one. 

Application Development: This group is in charge of the part the user actually sees. They are building the web dashboard and the interface. Their job is to take the complex math from the backend and transform it into something a property manager could better understand and use. 

This setup is flexible enough that we can move people around as the project shifts from analysis to development. 

## Metrics & Quantifiable Objectives 

The primary measure for success for this project is the accuracy of the HVAC runtime model. The model predicts hourly HVAC runtime based on weather conditions and thermostat settings as input variables. Model performance is evaluated by Mean Absolute Error (MAE) and R^2. MAE measures the average absolute difference between the predicted runtime and the observed runtime, while the R^2 will evaluate how well the model explains the variability in HVAC runtime.   

In addition to accuracy of the model, the project evaluates estimated savings by comparing predicted runtime under the observed thermostat settings with the predicted runtime under optimized thermostat settings in the same weather. The difference between these two runtimes will represent the potential runtime inefficiency in the systems operation. The project objective is to identify at least 5% of excess HVAC runtime across analyzed portfolio of thermostats.  

Qualitative Objective: The goal is to keep a comfortable environment for residents and proof they should change their settings to decrease their HVAC runtime and save money.  

Technical Metric: The model needs to be accurate; the average error, or RMSE, should be less than 5 minutes every hour. A consistency score of less than 30% should be met for the project to be viable. This follows the ASHRAE (2014) standard for models that are considered properly calibrated to real building data. 

To visualize our end-to-end data flow, we have created a one-page flowchart. This flow chart starts with raw data processing and ends with the output.  

<img width="463" height="345" alt="Screenshot 2026-02-09 at 6 21 15 PM" src="https://github.com/user-attachments/assets/aeb64147-c592-4728-b6e2-915ce93b5a7f" />


## Plan 

The timeline is broken down into five phases so that we can move logically from raw data to a finished product. 

Phase 1: Data Preprocessing + Data Analysis. We will clean the data to remove any weird inconsistencies. Then we align the weather and HVAC data using interpolation to create a unified timeline. We also need to interpret the on/off signals to get consistent runtime data and establish a financial baseline. This is necessary because the thermostat records when the HVAC system turns on and off, while the timestamps between these state changes often contain null values. This interpretation will allow the system to accurately reconstruct how long the HVAC unit was running.  

Phase 2: Develop Models. This phase involves using a small subset of the data to explore different time series model approaches. Models that will be tested are neural network solutions like LSTMs or Transformers, as well as baseline approaches. Naive model, linear rolling window. Additional models may also be researched to determine if there is another model that will provide accurate runtime predictions. and research into additional models to test.  

Phase 3: Evaluate Models. We will use a separate validation set, so the model doesn't just memorize the training data. We will also use cross-validation to make sure the model stays stable over different periods of time. 

Phase 4: Dashboard + Scaling. We scale the models to a full set of 1,000 thermostats. We also build the dashboard features so users can upload new data and see analytics or charts. 

Phase 5: Predictive Model with Weather Forecasts. This is the stretch goal where we pull in weather forecasts from an API. This would allow the system to suggest temperature changes for efficiency before the weather actually hits. 

## Architecture & Communication 

The technical setup is built to handle a lot of time-series data without slowing down. The data comes from 1,000 homes over two years; each file is about 4MB, roughly 4GB total. Due to the amount of data, compression of the file is needed. This was done by changing the running mode values from off, heat, or cool to 0, 1, or 2 respectively. 

For tools, we are using Kaggle for initial storage to make it easier for the team to collaborate. We are using Python for all modeling and evaluation because it has all the standard libraries we need. The dashboard itself will be built with HTML, CSS, and JS, and we might use Django for the backend if the functionality needs to be more robust. 

The goal is to produce results that clearly demonstrate the benefits of optimized HVAC control. The model should not affect the customers' comfortability but save them money on their heating and cooling bills and create a more efficient home without changing any infrastructure. 

Effective communication is the only way this cross-functional team succeeds. We will use a group chat for quick questions and GitHub to share files. We also plan to meet outside of class every week to keep things moving. Finally, we will use a GitHub project board to keep track of tasks, making it easier to assign tasks to people, keeping everything transparent. 

 

## References 
 
Bamdad, K., Mohammadzadeh, N., Cholette, M., & Perera, S. (2023). Model predictive control for energy optimization of HVAC systems using EnergyPlus and ACO algorithm. Buildings, 13(12), 3084. https://doi.org/10.3390/buildings13123084 

Casimirri, M. (2025). Leveraging ASHRAE 14 Guidelines for Robust Building Energy Modeling: Computer Simulation and Decarbonization Strategies. International Journal of Energy Management (IJEM, 7(1). 

Center for Sustainable Systems. (2024). Residential buildings factsheet. University of Michigan https://css.umich.edu/publications/factsheets/built-environment/residential-buildings-factsheet  

Serale, G., Fiorentini, M., Capozzoli, A., Bernardini, D., & Bemporad, A. (2018). Model predictive control (MPC) for enhancing building and HVAC system energy efficiency: Problem formulation, applications and opportunities. Energies, 11(3), 631 https://doi.org/10.3390/en11030631  
