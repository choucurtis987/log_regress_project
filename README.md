# Logistic Regression Project
- [Data Cleaning](/Data_Cleaning.ipynb)
- [Logistic Regression Model](/Logistic_Regression_Model.ipynb) with a few of its features
## My Logistic Regression Algorithm Built From Scratch 
- A binary classification algorithm 
- Utilizes Numpy as main component to essentially perform all the matrix multiplication needed
- A cross entropy loss function is used to determine the cost, optimize the weights, and bias
- A gradient descent optimization approach is taken by updating weights and bias every iteration
- The probabilities of the prediction function are taken by matrix multiplying the weights with the features and adding a matrix of bias in which the result is then passed through the sigmoid activation function (equilvalent to the estimated regression equation) to give a probability
## Purpose of Logisitic Regression
- to model the probability of an event occuring based on input values
- classify new data to a class by estimating the probability it belongs in that class
## Other Links
- Devpost Link for further background: https://devpost.com/software/n-a-ljwa0f
- To run interactive demo, run [this script](/Interactive_Demo/Interactive.py) with an interactive environment but make sure all dependencies are there. All dependencies should be in [this folder](/Interactive_Demo)
- Dataset Link: https://www.kaggle.com/uciml/student-alcohol-consumption
