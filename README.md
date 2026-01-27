## Flight Delay Predictor (Denver International Airport)
This model is a time-series machine learning system built to support airport operation decision-making by predicting two things: whether a flight will be delayed and by how many minutes at DIA.

The project uses two LSTM neural netwoks. One for binary classification and one for delay regression. Both are trained on a large scale dataset of historical aviation and weather data to model congestion, operational slack, and temporal delay propogation.

## Problem Statement
Flight delays are rarely isolated events. Aircraft turnarounds, congestion, and weather conditions cause delays across many tightly scehdule departure banks. This project attempts to model those dynamics by answering the following questions:
    Will this departure be delayed?
    If so, by how many minutes will the flight be delayed by?

## Modeling Approach
Two independent LSTM models were trained on sequential flight data.
A classifier was developed for binary classification (yes/no) delay. A regressor was developed to predict total delay minutes from scheduled departure time. Each model uses the last 12 flights to predict whether or not the current flight will be delayed.

## Data Sources
The data used largely revolved around the Burea of Transportation Statistics (BTS) On-time Performance reports for both departures and arrivals coming out of Denver International. The 5 biggest airlines'(Southwest, United, Frontier, Delta, American) reports were used for both 2023 and 2024. On top of such reports, BTS T-100 domestic data and the meteostat API were used to add supplementary data. 

## Feature Engineering
Through exploratory data analysis, it was discovered that, contrary to intital belief, most delays do not occur due to adverse weather conditions. To better understand what actually influences departure delays, the following features were created:

    Bank Density: The number of flights departing or arriving within 
    +/- 15 minutes of the scheduled departure. 

    Turnaround slack: The scheduled time between an aircraft's arrival and subsequent departure, linked via tail numbers present in BTS data

    Rolling Delay Context:
        Mean delay minutes (3h/6h/12h)
        Delay counts (3h/6h/12h)
        Arrival & departure counts (1h)

## Results
The classification model has a 0.80 f1 score. The regression model has a mean absolute error of 19.5 minutes (after being scaled back out). Data followed a 80/20 data split, but had to randomness to it due to the nature of time-series data. Both models had a 0.1 dropout rate and utilized the Adam optimizer.

## Running Locally
A docker image of the project was created so that containers can be created locally. To run said image, use the following commands:
    
    docker build -t flight_delay_api
    docker run --rm -p 5000:5000 flight_delay_api

## Note on Live Inference
The initial goal of the project was to have a live inference dashboard. Unfortunately, there are no free/"freemium" APIs that report aviation data based of tail numbers (in a consistent manner at least). Therefore, a makeshift live-demo API had to be created as an alternative. 

## Future Improvements
The biggest future improvement without a doubt is the creating of a live inference dashboard. The second biggest improvement would come in the shape of having a multi-airport model. Both of these improvements would create a much more complete and rounded model. 