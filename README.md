# Logistic Regression Project
- [Data Cleaning](/Data_Cleaning.ipynb)
- [Logistic Regression Model](/Logistic_Regression_Model_Final.ipynb) with a few of its features
## My Logistic Regression Algorithm Built From Scratch 
- A binary classification algorithm 
- Utilizes Numpy as main component to essentially perform all the matrix multiplication needed
- A cross entropy loss function is used to determine the cost, optimize the weights, and bias
- A gradient descent optimization approach is taken by updating weights and bias every iteration
- The probabilities of the prediction function are taken by matrix multiplying the weights with the features and adding a bias in which the result is then passed through the sigmoid activation function (equilvalent to the estimated regression equation) to give a probability
## Purpose of Logisitic Regression
- to model the probability of an event occuring based on input values
- classify new data to a class by estimating the probability it belongs in that class
## This Algorithm's Use
- Determine is a student is going to pass a class or not given the student's lving circumstances as input
## Potential Furture Directions
- Ultilize a different loss function
- Use more data, different columns in the dataframe
- Use Stochastic Gradient Descent to update weights and bias
## Resources Used
- [ML Cheatsheet](https://ml-cheatsheet.readthedocs.io/en/latest/logistic_regression.html#binary-logistic-regression)
- [Statistics of Logistic Regression](https://www.youtube.com/playlist?list=PLIeGtxpvyG-JmBQ9XoFD4rs-b3hkcX7Uu)
- [Cross-Entropy Loss function](https://www.youtube.com/watch?v=MztgenIfGgM&list=PLXf4nYvqR6VMI_rsU0otM9KHmUBDhCGX_&index=4&t=0s)
- [Good Video on Overall Math of Logistic Regression](https://www.youtube.com/watch?v=QHm0UDG6IU4&list=PLXf4nYvqR6VMI_rsU0otM9KHmUBDhCGX_&index=5&t=1435s)
## Other Links
- Devpost Link for further background: https://devpost.com/software/n-a-ljwa0f
- Dataset Link: https://www.kaggle.com/uciml/student-alcohol-consumption
