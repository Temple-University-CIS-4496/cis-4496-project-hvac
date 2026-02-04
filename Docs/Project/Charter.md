# Project Charter

## Business background

- Who is the client, what business domain the client is in.

  Telkonet Inc. Energy Management/Heating and Cooling domain.

- What business problems are we trying to address?

  Making HVAC machines more energy and cost efficient

  Are we able to illustrate to the customer more details about potential savings based on restricted set points, or encourage changing set points based on our findings?

## Scope

- What data science solutions are we trying to build?

  We are trying to predict HVAC runtime based on the outdoor weather, then using this to understand how much money could be saved by making changes to the thermostat set points. How much does weather variablilty change HVAC runtimes?

- What will we do?

  We are going to analyze data from 1000 thermostats across 2 years and create a model that will use HVAC runtime to predict how long the HVAC will run based on the temperature outside and the set point. 

  **TODO** ADD more details

- How is it going to be consumed by the customer?

  The customer will be able to save money. Possibly make a GUI that would allow the customer to get the live data. Possibly also have predictive modeling using the meteorology data.

## Personnel

- Who are on this project:
    - Project lead
    - PM
    - Data scientist(s): Zack Aidarov, Ayush Gupta, Lauren Sobieski, Max Suc, Andrew Coffman, Dylan Heathcote
  - Client:
    - Data administrator
    - Business contact

## Metrics

- What are the qualitative objectives? (e.g. reduce user churn)

  Improving comfort of the occupant inside of their home. Business objective: If I ask a resident to lower their set point in the winter one degree I need data to show that it will save them energy and money. 

- What is a quantifiable metric (e.g. reduce the fraction of users with 4-week inactivity)

  Reduce the cost of HVAC-related bills. 

- Quantify what improvement in the values of the metrics are useful for the customer scenario (e.g. reduce the fraction of users with 4-week inactivity by 20%)

  We can give concrete energy/money data that will help them make informed decisions to try and improve the cost by 3%.
  Business questions: can we give customers an efficiency score and how they compare to everyone else? To help push energy efficiency 

- What is the baseline (current) value of the metric? (e.g. current fraction of users with 4-week inactivity = 60%)

  During our project we will determine the current HVAC baseline each house pays based off runtime and 17 cents per KWH. 

- How will we measure the metric? (e.g. A/B test on a specified subset for a specified period; or comparison of performance after implementation to baseline)

  2.5 ton AC unit is 8,790 watts. We will measure it by

## Plan
- Phases (milestones), timeline, and short description of what we'll do in each phase.

Phase 1: Data Preprocessing + Data Analysis
 - Clean data
 - Align time values for weather and HVAC information through interpolation
 - Create consistent runtime data through interpreting on/off signals
 - Figure out bill amounts through runtime and kWh pricing
 - Chart data to explore pre-existing seasonal trends, etc.

Phase 2: Develop Models
 - Use a small subset of the data as a training set for initial data exploration and model development
 - Explore different time series models for the data
 - Test neural network time series solutions (LSTM, RNN, Transformer, etc.)

Phase 3: Evaluate Models
 - Utilize a validation set separate from training set during training to prevent overfitting
 -  use cross validation depending on model runtime

Phase 4: Dashboard + Scaling
 - Scale models to full dataset
 - Dashboard Feature: Upload new data – send through a pipeline
 - Dashboard Feature: View analytics / graphs / charts

Phase 5: Predictive Model with Weather Forecasts (Stretch Goal)
 - Pull weather forecasts from API call to change temperature for efficiency

## Architecture

- Data
  - The data we expect for this project is raw data from customer-owned thermostats, which include CSV files containing time-series data. This data includes the fields: Timestamp, Temperature, display_setpoint, system_mode, system_occ, fan_mode, fan_state, humidity, output_state, running_mode. We expect approximately 1,000 CSV files, each of which has a continuous record of thermostat behavior over the last two years.

  - Initially, data on Kaggle using multiple folders for training data, validation, and testing data to make models run faster while exploring different models. TBD: Use a tool such as the kagglehub API to fetch the data for the project to run, or choose a cloud storage (AWS S3, Azure, etc) once the data is all in one place.
- Tools and Data Storage
  - Kaggle for initial data storage
  - Python for modeling and model evaluation
  - HTML/CSS/JS for dashboard creation; Django / Flask (?) for backend if necessary
- How will the score or operationalized web service(s) (RRS and/or BES) be consumed in the business workflow of the customer? If applicable, write down pseudo code for the APIs of the web service calls.
- How will the customer use the model results to make decisions?
  - The customer will use the model results to make decisions on what set point to set their thermostat. A consumer will not be motivated to change the set point of their house unless they are provided set statistics to back up doing so will save them money. 
- Data movement pipeline in production: Make a 1 -slide diagram showing the end -to-end data flow and decision architecture
 <img width="770" height="352" alt="image" src="https://github.com/user-attachments/assets/33520373-83e5-4293-b28d-6ba96c5453bb" />


  - If there is a substantial change in the customer's business workflow, make a before/after diagram showing the data flow.

## Communication
Communicate logistics using text message group chat. Use Outlook for file sharing / code sharing as necessary. Meet outside of class during the week and utilize Github project board for organizing who completes which tasks.
