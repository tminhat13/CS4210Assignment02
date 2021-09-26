#-------------------------------------------------------------------------
# AUTHOR: Nhat Tran
# FILENAME: title of the source file
# SPECIFICATION: description of the program
# FOR: CS 4210- Assignment #2
# TIME SPENT: 1
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard vectors and arrays

#importing some Python libraries
import csv

from sklearn.naive_bayes import GaussianNB

dbTraining = []
#reading the training data
with open('weather_training.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for i, row in enumerate(reader):
        if i > 0:  # skipping the header
            dbTraining.append(row)

#transform the original training features to numbers and add to the 4D array X. For instance Sunny = 1, Overcast = 2, Rain = 3, so X = [[3, 1, 1, 2], [1, 3, 2, 2], ...]]
#--> add your Python code here
num_attributes = 4
dim1, dim2 = (num_attributes, len(dbTraining))
X = [[0 for i in range(dim1)] for j in range(dim2)]
for k in range(len(dbTraining)):
    for j in range(len(dbTraining[k]) - 1):
        if (dbTraining[k][j] == "Sunny") | (dbTraining[k][j] == "Hot") | (dbTraining[k][j] == "High") | (dbTraining[k][j] == "Weak"):
            X[k][j] = 1
        elif (dbTraining[k][j] == "Overcast") | (dbTraining[k][j] == "Mild") | (dbTraining[k][j] == "Normal") | (dbTraining[k][j] == "Strong"):
            X[k][j] = 2
        elif (dbTraining[k][j] == "Rain") | (dbTraining[k][j] == "Cold"):
            X[k][j] = 3

#transform the original training classes to numbers and add to the vector Y. For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
#--> add your Python code here
Y = ["0"] * len(dbTraining)
for k in range(len(dbTraining)):
    if dbTraining[k][4] == "No":
        Y[k] = 2
    elif dbTraining[k][4] == "Yes":
        Y[k] = 1

#fitting the naive bayes to the data
clf = GaussianNB()
clf.fit(X, Y)

#reading the data in a csv file
#--> add your Python code here

#printing the header os the solution
print ("Day".ljust(15) + "Outlook".ljust(15) + "Temperature".ljust(15) + "Humidity".ljust(15) + "Wind".ljust(15) + "PlayTennis".ljust(15) + "Confidence".ljust(15))

#use your test samples to make probabilistic predictions.
#--> add your Python code here
#-->predicted = clf.predict_proba([[3, 1, 2, 1]])[0]


