# log_regress_project

## Inspiration
The project was inspired by my desire to learn more about machine learning and to place more emphasis on mental in the public and private high school school systems. In our high school systems today, the amount of workload continues to increase, and with the recent removal of the SAT requirement at some colleges is a step in the right direction. However, many schools still fail to aid students mentally because they don't have qualified counselors or have trouble identifying students going through a difficult time. This is where N/A software comes in. 

## What it does
N/A is binary classifier that can identify students who have a potential to fail a certain class. Students who have the potential of failing can be identified based on a survey they take where they answer questions about their living situation in which heavily contributes to whether students pass or fail in school. By identifying students who can potentially fail with this software, schools can focus more on counseling and providing resources to these students instead of leaving them behind with a "yearly" checkup. In doing this, we prioritize a student's mental health over their education making sure that resources are provided to them to combat whatever challenges they be facing. This software enriches a student's environment by allowing schools to give students a support system who may have never have even had one. Ultimately, this allows school to maximize their student's capabilities, confidence, and school performance. 

## How I built it
This code could have been done in a lot less lines by simply using a higher level Logistic Regression model but I decided to challenge myself and build the entire model from scratch learning it ins and outs. First, I cleaned the data from the original dataset converting strings to numeric values and setting up the data so it could be inputted into the model. The model was built many using Numpy, a lot of math, and a cross entropy loss function. The prediction part was easy and it was simply multiplying the weights with the inputs, adding in the bias, and passing them through a sigmoid function to output the predicted probabilities. From there we could classify the probabilities as 0 if the probability was less than less 60% (a failing grade) and 1 if the student could potentially pass the class. However, my challenges came with updating the weights, bias, and finding the right matrix shapes having never taken a linear algebra class. 

## Challenges I ran into
In trying to updating the weights and bias, I had the learn more about the cross entropy cost function. From there I was able to input its formula into code along with its partial derivative to update the weights and bias. Once I was able to understand cross entropy math, it became a lot simpler. However, the biggest challenge I ran into was just figuring out the right dimensions for the matrix multiplication and addition required in the program. 

## What I learned
I was able to further understand weights and bias (feed forward and backpropagation) by creating this program and ultimately the math behind these concepts which is a win for me. I was also able to pick up some basic linear algebra skills after this program. Both of which have helped me improve my understanding of machine learning.

## What's next for N/A

