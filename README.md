# flight-delay
I used Python to analyze the flight delay in the United States. 
<Intrduction>\n
Since the pandemic has eased and the prevention policies have been released, more and more people are traveling. As a Taiwanese, I want to know if the arrival delay depressed our good mood on the trip. I assume we take United Airlines, the only one conducting direct flights between Taiwan and the USA(San Fransico), and make a trip to various cities in the USA. Also, I use the concept of codeshare flights, taking off from San Fransico, and build a linear regression to estimate the flight delay at airports.\n

<data information>\n
dataset: from Kaggle, Flight Delay and Cancellation Dataset (2019-2023). (download time: 2023/12)\n
data source: Aviation System Performance Metrics, ASPM.\n

<Breif result>\n
# On-time or arrive early: about 65%; Arrival delay: about 33% (about 44% of flights delay less than 15 mins)\n
# Most arrival delays happen during bad weather and national holidays.\n
# The choice of test size for linear regression is 6:4 considering MSE, R-squared.\n
*** Please see the file(flight delay presentation_2024.01_online) for more visual result. \n

