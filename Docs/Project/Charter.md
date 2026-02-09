# Project Charter

## Project Charter: HVAC Runtime Prediction & Energy Optimization 

## Business Background 

Telkonet Inc. is currently a major player in the energy management field, yet they are running into a serious roadblock regarding how heating and cooling systems waste energy. Most of the trouble comes down to thermostats that stick to fixed settings. These devices essentially ignore how long a building holds onto heat, and they don't really pay attention to the shifting weather outside either. Truth be told, this is a massive financial drain. To put it in perspective, the U.S. Department of Energy found that heating and cooling make up 45% of all residential energy use (Center for Sustainable Systems, 2024). Traditional controls just let that waste happen. However, smarter systems like Model Predictive Control have shown that it is possible to save about 40% of that energy (Serale et al., 2018). There is a huge gap between how things work now and how efficient they could actually be. 

The real opportunity here is to build a Counterfactual Energy Analysis tool. This is a system that helps property managers and residents move away from just reacting to problems such that they can start predicting them instead. We want to show people the actual financial value of energy-efficient habits. The best way to do that is by illustrating the cost difference between actual runtimes and simulated runtimes under better settings. This makes it much easier to make data-driven choices without making the people inside the building uncomfortable. 

## Scope & Data Science Methodology 

The system we are looking at depends on a solid data processing pipeline. This pipeline has to handle raw information from about 1,000 different thermostats over a two-year span. This data is often very sparse. To put it simply, the thermostat only records a new number whenever something actually changes. To manage that, the team is going to stick closely to the Phase 1 protocols, which involve cleaning the data and using linear interpolation. This is a necessary step so that we can line up the thermostat logs with outdoor weather data on the exact same timeline. 

Moreover, we are going to calculate runtime by looking at specific signals from the output status, such as when the heat is actively running, rather than just looking at the general system mode. This is important because it helps us tell the difference between the system actually using energy and the system just sitting on standby. Once that data is ready, it will feed into a predictive engine. In addition to that, the team will use a smaller training set during Phase 2 to try out different models. We want to explore decision trees as well as deep learning methods like LSTMs or Transformers. These models have to be tough enough to handle sudden weather shifts, especially those sharp spikes in usage that happen when the temperature outside drops fast. 

The main result of all this work will be a comparison engine for What-If simulations. We will feed the model two separate inputs: the real weather with real settings, and then that same weather with optimized settings. After that, the system calculates the difference in how long the unit ran. Property managers can then use this as a monthly auditing tool. It helps them spot high-waste units so they can adjust their policies for the next month before bills get too high. 

## Personnel 

The project is going to be handled by a team of data scientists who will work closely with the project lead and the client contacts. We want to make sure we hit every technical milestone, so the team is organized to handle everything from the first data pull to the final dashboard. The core team consists of Zack Aidarov, Ayush Gupta, Lauren Sobieski, Max Suc, Andrew Coffman, and Dylan Heathcote. 

We are avoiding a setup where everyone has one rigid task because that usually creates a lot of bottlenecks. Instead, everyone shares ownership of the code. All that being said, the work is still divided into three main areas to keep things organized: 

Data Engineering: This group focuses on the backend infrastructure. They are the ones building the pipeline to clean up the logs and compress the data so it runs fast. Not only that, but they have to make sure the large CSV files stay valid and that the interpolation logic actually works. 

Modeling & Simulation: This part of the project is all about the core data science problems. This team develops and tests the algorithms to predict runtime and figure out the savings. They will be doing a lot of testing on both tree-based and deep learning models to find the most accurate one. 

Application Development: This group is in charge of the part the user actually sees. They are building the web dashboard and the interface. Their job is to take all that complex math from the backend and turn it into something a property manager can actually understand and use. 

This setup is flexible enough that we can move people around as the project shifts from analysis to development. 

## Metrics & Quantifiable Objectives 

The main way we are measuring success is through Estimated Cost Savings. While we are definitely tracking technical accuracy, the project is only a win if we can identify at least 5% in energy waste that can be recovered across the whole portfolio. 

We define this metric by taking the difference between the baseline runtime and the optimized runtime, then multiplying that by the power the hardware uses, which is about 8.79 kilowatts for a standard unit. We also have to multiply by the cost of electricity. Since prices change depending on where you are, the calculator will check savings at different rates, such as 0.15 USD or 0.20 USD per kilowatt-hour. 

Qualitative Objective: We want to make the people in the homes more comfortable while giving property managers the proof they need to change settings. 

Technical Metric: The model needs to be accurate. To be specific, the average error, or RMSE, should be less than 5 minutes every hour. On top of that, we are aiming for a consistency score of less than 30%. This follows the ASHRAE (2014) standard for models that are considered properly calibrated to real building data. 

To visualize our end-to-end data flow, we have created a one-page flowchart. This flow chart starts with raw data processing and ends with the output. 

<img width="463" height="345" alt="Screenshot 2026-02-09 at 6 21 15 PM" src="https://github.com/user-attachments/assets/aeb64147-c592-4728-b6e2-915ce93b5a7f" />


## Plan 

The timeline is broken down into five phases so that we can move logically from raw data to a finished product. 

Phase 1: Data Preprocessing + Data Analysis. We will clean the data to get rid of any weird inconsistencies. Then we align the weather and HVAC data using interpolation to create a unified timeline. We also need to interpret the on/off signals to get consistent runtime data and establish a financial baseline. 

Phase 2: Develop Models. This involves using a small subset of the data to explore different time series models. We want to test neural network solutions like LSTMs or Transformers to see how they handle complex relationships in the data. 

Phase 3: Evaluate Models. We will use a separate validation set so the model doesn't just memorize the training data. We will also use cross-validation to make sure the model stays stable over different periods of time. 

Phase 4: Dashboard + Scaling. At this point, we scale the models to the full set of 1,000 thermostats. We also build the dashboard features so users can upload new data and see analytics or charts. 

Phase 5: Predictive Model with Weather Forecasts. This is the stretch goal where we pull in weather forecasts from an API. This would allow the system to suggest temperature changes for efficiency before the weather actually hits. 

## Architecture & Communication 

The technical setup is built to handle a lot of time-series data without slowing down. The data comes from 1,000 homes over two years, and since each file is about 4MB, we are looking at roughly 4GB total. Efficiency is the name of the game here. Because there is so much data, we are going to compress the files by changing the running mode values from off, heat, or cool to 0, 1, or 2 respectively. 

For tools, we are using Kaggle for the initial storage to make it easier for the team to collaborate. We are using Python for all the modeling and evaluation because it has all the standard libraries we need. The dashboard itself will be built with HTML, CSS, and JS, and we might use Django for the backend if the functionality needs to be more robust. 

At the end of the day, the customer is going to use these results to decide where to set their thermostats. People usually aren't motivated to change their habits unless you can show them exactly how much money they are going to save. 

Effective communication is the only way this cross-functional team succeeds. We will use a group chat for quick questions and Outlook for sharing files or code. We also plan to meet outside of class every week to keep things moving. Finally, we will use a Github project board to keep track of who is doing what so everything stays transparent. 

 

## References 
 
 

Bamdad, K., Mohammadzadeh, N., Cholette, M., & Perera, S. (2023). Model predictive control for energy optimization of HVAC systems using EnergyPlus and ACO algorithm. Buildings, 13(12), 3084. https://doi.org/10.3390/buildings13123084 

Casimirri, M. (2025). Leveraging ASHRAE 14 Guidelines for Robust Building Energy Modeling: Computer Simulation and Decarbonization Strategies. International Journal of Energy Management (IJEM, 7(1). 

Residential buildings factsheet | center for sustainable systems. Residential Buildings Factsheet. (2024). https://css.umich.edu/publications/factsheets/built-environment/residential-buildings-factsheet  
