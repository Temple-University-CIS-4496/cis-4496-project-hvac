# Project Charter

## Business background

- Who is the client, what business domain the client is in.

  Telkonet Inc. Energy Management/Heating and Cooling domain.

- What business problems are we trying to address?

  Are we able to illustrate to the customer more detials about potential savings based on restricted set points, or encouraging changing set points. 

## Scope

- What data science solutions are we trying to build?

  Trying to predict HVAC runtime based on the outdoor weather. Using this to understand how much money could be saved by making changes to the termostat set points, and does it vary by wheater? 

- What will we do?

We are going to analyze data from 1000 thermostats across 2 years and create a model that will use hvac runtime to predict how long the hvac will run based on the temperature outside and the set point. 
**TODO** ADD more detials

- How is it going to be consumed by the customer?

  The customer will be able to save money. Possibly make a GUI that would allow the customer to get the live data. Possibly also have predictive modeling using the meteorology data.

## Personnel

- Who are on this project:
  - Microsoft:
    - Project lead
    - PM
    - Data scientist(s): Zack Aidarov, Ayush Gupta, Lauren Sobieski, Max Suc, Andrew Coffman, Dylan Heathcote
  - Client:
    - Data administrator
    - Business contact

## Metrics

- What are the qualitative objectives? (e.g. reduce user churn)

  Improving comfort of the occupant inside of their home. Buisness objective: If I ask a resident to lower their set point in the winter one degree I need data to show that it will save them energy and money. 

- What is a quantifiable metric (e.g. reduce the fraction of users with 4-week inactivity)

  Reduce the cost of HVAC-related bills. 

- Quantify what improvement in the values of the metrics are useful for the customer scenario (e.g. reduce the fraction of users with 4-week inactivity by 20%)

  We can give concret energy/money data that will help them make infomred decisions to try and improve the cost by 3%.
  Buisness questions: can we give customers an effiencincy score and how they compare to everyone else? To help push energy effiencicy 

- What is the baseline (current) value of the metric? (e.g. current fraction of users with 4-week inactivity = 60%)

During our project we will determine the current HVAC baseline each house pays based off runtime and 17cents per KWH. 

- How will we measure the metric? (e.g. A/B test on a specified subset for a specified period; or comparison of performance after implementation to baseline)

2.5 ton AC unit is 8,790 watts. 
We will measure it by

## Plan

- Phases (milestones), timeline, short description of what we'll do in each phase.

## Architecture

- Data
  - What data do we expect? Raw data in the customer data sources (e.g. on-prem files, SQL, on-prem Hadoop etc.)
We expect CSV files 
  
- Data movement from on-prem to Azure using ADF or other data movement tools (Azcopy, EventHub etc.) to move either
  - all the data,
  - after some pre-aggregation on-prem,
  - Sampled data enough for modeling

- What tools and data storage/analytics resources will be used in the solution e.g.,
  - ASA for stream aggregation
  - HDI/Hive/R/Python for feature construction, aggregation and sampling
  - AzureML for modeling and web service operationalization
- How will the score or operationalized web service(s) (RRS and/or BES) be consumed in the business workflow of the customer? If applicable, write down pseudo code for the APIs of the web service calls.
  - How will the customer use the model results to make decisions
  - Data movement pipeline in production
  - Make a 1 slide diagram showing the end to end data flow and decision architecture
    - If there is a substantial change in the customer's business workflow, make a before/after diagram showing the data flow.

## Communication

- How will we keep in touch? Weekly meetings?
- Who are the contact persons on both sides?
