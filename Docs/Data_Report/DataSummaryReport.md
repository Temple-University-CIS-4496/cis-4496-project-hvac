# Data Report


## General summary of the data
The data used for this project is time series thermostat data combined with outdoor weather data. What is included in the outdoor weather data is the timestamp when the data was reported, the outside temperature, the outside temperature minimum, the outside temperature maximum, and outside humidity.  This data set was combined with the time series of thermostat data.  

The information provided from the thermostats are; time stamp, temperature inside, set point, mode, occupied state, fan state output state, running mode, and rental status. Currently there are 100 csv files from 100 different thermostats that have been combined with outdoor weather data. 

The time period of this data is from October 1st 2024 to January 27th 2026. The sampling frequency for the outdoor weather data is every 15 minutes. While the sampling frequency of the thermostat data is recorded at irregular intervals, which meanst that there is no fixed sampling frequency for the thermostat measurments.  

## Data quality summary
Before the processing of the raw data, several data quality issues were identified. There mostly null values in the thermostat time series data, occasional gapsin the timestamps, timestamps from the thermostat files did not match up with the timestamps on the outdoor weather data, and the dataset did not initially include a column for HVAC cumulative runtime or energy cost.  

Through data processing the null values were fixed through forward filling columns creating complete columns with no null values for key variables such as fan state, running mode, setpoint, and indoor temperature.. Then, to address theoutdoor temperature being on a separate csv file, an outer join, interpolation, then left join was used so each thermostat csv was able to get the outdoor temperature data added to their csv. To address the lack of cummulative runtime and energy cost columns, cummulative runtime was calulated by counting the number of time interavles when the HVAC was running, and then summing this over time. The cummulative runtime is was used to calulate operating costs. To convert seconds to hours, the runtime is divded by 3600, then it is mulitped by $0.17 per kWh and multiplied by 3.0 kW as the estimated power usage to appoximate the toal cost of operation. Overall after the data processing the data quality increased leaving us with a working working data set to use in modeling.  
## Target variable
The target variable is in this analysis is HVAC runtime, which is the total time an HVAC system heats or cools during a time interval. The HVAC runtime is used to determine how much energy is used for heating and cooling. By modleing runtime with outdor weather condituions and thermostat settings, we can estimate how environemntal facutes will influence energy usage.  
## Individual variables
This data set influences several variables that influence the HVAC runtime. All of the variables can be sorted into three categories: weather variables, thermostat variables, and time variables.  

#### Weather variables: Outdoor temperature, humidity, and other environmental conditions that influence heating or cooling demand. 

#### Thermostat variables: Indoor temperature setpoints, occupancy state, and system mode  

#### Time variables: Timestamp: such as hour of day, day of week or season 

## Variable ranking
The variables that most importnat for predicted run time are ranked as followed: 

#### Outdoor temperature 

This is the most important for predicting runtime as the temperature outside plays a big factor on how hard a HVAC must work and what temperature the occupant wants to set their set point to  

#### Cooling or heating set point  

This is the temperature the occupant sets their thermostat to. Meaning that the HVAC will run in order to maintain this set point  

#### Time of day  

HVAC usage changes at differnt times during the day. For example, during cooling seasons the HVAC tends to run more frequently during warmer daytime hours, while heating demand can increase during colder early morning periods during the heating season. 

#### Occupancy state 

This reports if there is an occupant in the house. When the home is occupied, the occupants are more likely to adjust the thermostat settings for comfort, which will change the HVAC runtime.  
## Relationship between explanatory variables and target variable
Analysis of the relationship between explanatory variables and the HVAC runtime revealed clear patterns.  

The first one is that there are clear seasonal differences between the heating season (winter) and the cooling season (summer). During the heating season there are large peaks in the early morning, while in the cooling season there are peaks in the afternoon that forces AC activation. Showing that when it gets hotter during the cooling season, the AC needs to work harder. 

Another observation is that the HVAC system spends most of its time sitting idle or off, and that the heating and cooling modes are only used when necessary. Additionally,heating and cooling systems appear to have similar active workloads, meaning that both modes operate for comparable amounts of time during their respective seasons. 

