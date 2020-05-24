from Logistic_Regression_Model import *

import numpy as np

arr = np.empty((0,23), int)
arr = np.append(arr, features, axis=0)

your_data = []

print("\nHi, welcome to our survey! Please answet the following questions the best you can.\n")

questions = [
    '\nDo you live in an urban or rural area? (1-yes, 0-no) ',
    '\nHow big is your family size? (1-greater than three, 0-three or less ) ',
    '\nAre your parents living together or apart? (1-together, 0-apart) ',
    '\nWhat is your mothers highest education? (0 - none, 1 - primary education (4th grade), 2 – 5th to 9th grade,\n3 – secondary education or 4 – higher education) ',

    '\nWhat is your fathers highest education? 0 - none, 1 - primary education (4th grade), 2 – 5th to 9th grade,\n3 – secondary education or 4 – higher education ',
    '\nWhat is your travel time to school? (enter a number 1-4 where 1 is less than 10min and 4 is greater than or equal to an hour) ',
    '\nHow much time do you study per week? (enter hours) ',
    '\nHow many classes have you failed? (enter a number 1-3, if 4 or more enter 4) ',
    '\nDo you have extra educational support? (1-yes, 0-no) ',
    '\nDo you have family educational support? (1-yes, 0-no) ',
    '\nDo you pay for additional support in classes? (1-yes, 0-no) ',
    '\nAny extra curricular-activities? (1-yes, 0-no) ',
    '\nAttended a nursery school? (1-yes, 0-no) ',
    '\nWant a higher education? (1-yes, 0-no) ',
    '\nInternet access at home? (1-yes, 0-no) ',
    '\nIn a romantic relationship? (1-yes, 0-no) ',
    '\nquality of family relationships? (enter a number from 1 - very bad to 5 - excellent) ',
    '\nDo you have freetime after school? (enter a number from 1 - very low to 5 - very high) ',
    '\nDo you go out with friends often? (enter a number from 1 - very low to 5 - very high) ',
    '\nWorkday alcohol consumption? (enter a number from 1 - very low to 5 - very high) ',
    '\nWeekend alcohol consumption? (enter a number from 1 - very low to 5 - very high) ',
    '\nCurrent health status (from 1 - very bad to 5 - very good) ',
    '\nNumber of absences (enter a number 0-93 if 93 or higher enter 93)'
]

for i in questions:
    answer = int( input(i) )
    your_data.append(answer)

your_data = np.array(your_data).reshape(1,23)

arr = np.append(arr, your_data, axis=0)

features = arr

predictions = predict(features[len(bias)+2:], weights, bias)

predicted = classify(predictions)
print("\nCONFIDENTIAL RESULT:")
print(predicted[-1])

print("\naccuracy of result:")
print( accuracy(predicted, targets[198:]) )
