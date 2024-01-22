I used Python code to analyze the flight delay in the United States. 

# Introduction
Since the pandemic has eased and the prevention policies have been released, more and more people are traveling. As a Taiwanese, I want to know if the arrival delay depressed our good mood on the trip. I assume we take United Airlines, the only one conducting direct flights between Taiwan and the USA(San Fransico), and make a trip to various cities in the USA. Also, I use the concept of codeshare flights, taking off from San Fransico, and build a linear regression to estimate the flight delay at airports.

# data information
* Dataset: from Kaggle, Flight Delay and Cancellation Dataset (2019-2023). (download time: 2023/12)
* Data source: Aviation System Performance Metrics, ASPM.

# Breif result
* On-time or arrive early: about 65%; Arrival delay: about 33% (about 44% of flights delay less than 15 mins)
* Most arrival delays happen during bad weather and national holidays.
* The choice of test size for linear regression is 6:4 considering MSE, R-squared.
* Please see the file(flight delay presentation_2024.01_online) for more visual result. 

