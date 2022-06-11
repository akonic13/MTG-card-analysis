# MTG-card-analysis 
## Project overview
-Goal: Peform data analysis and build a machine learning model to predict the prices of cards legal in the Commander format of Magic the Gathering
-Format Choice: I chose to analyze cards from the Commander format as it is my personal favorite format of the game and the currently fastes growing format in all of Magic the Gathering. The format also has a large card pool, over 22000 legal cards, leading to an extremely large and diverse data set

## Data source
scryfall.com

## Method Outline
1) build web scraper to pull pertinent card inofrmation from scryfall website
2) clean and organize data and perform exploratory data analysis
3) Use insights from data analysis to build a machine learning model to predict the price of cards based on given features

## Results Summary
Four machine learning model types were used to predict the price of the careds:
1) Linear Regression
2) Support Vector Regression
3) Random Forest Regression
4) Neural Network Regression
Out of all 4 models tested, the neural network regression model perofrmed the best overall with a mean absolute error(MAE) of $1.10 and a root mean squared error(RMSE) of $2.78. Further details on each model can be found in the project powerpoint.
