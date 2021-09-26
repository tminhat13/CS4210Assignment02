# -------------------------------------------------------------------------
# AUTHOR: Nhat Tran
# FILENAME: knn.py
# SPECIFICATION: description of the program
# FOR: CS 4210- Assignment #2
# TIME SPENT: 1 hour
# -----------------------------------------------------------*/

# IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to
# work here only with standard vectors and arrays

# importing some Python libraries
from sklearn.neighbors import KNeighborsClassifier
import csv

db = []

# reading the data in a csv file
with open('binary_points.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for i, row in enumerate(reader):
        if i > 0:  # skipping the header
            db.append(row)
errorNum = 0
# loop your data to allow each instance to be your test set
for i, instance in enumerate(db):
    # add the training features to the 2D array X removing the instance that will be used for testing in this
    # iteration. For instance, X = [[1, 3], [2, 1,], ...]]. Convert each feature value to float to avoid warning
    # messages --> add your Python code here X =
    num_attributes = 2
    dim1, dim2 = (num_attributes, len(db))
    X = [[0 for j in range(dim1)] for k in range(dim2)]
    for k in range(len(db)):
        for j in range(len(db[k]) - 1):
            X[k][j] = float(db[k][j])


    # transform the original training classes to numbers and add to the vector Y removing the instance that will be
    # used for testing in this iteration. For instance, Y = [1, 2, ,...]. Convert each feature value to float to
    # avoid warning messages --> add your Python code here Y =
    Y = ["0"] * len(db)
    for k in range(len(db)):
        if db[k][2] == "-":
            Y[k] = float(2)
        elif db[k][2] == "+":
            Y[k] = float(1)
    del Y[i]

    # store the test sample of this iteration in the vector testSample
    # --> add your Python code here
    testSample = X.pop(i)
    # print(testSample)
    # print(X)

    # fitting the knn to the data
    clf = KNeighborsClassifier(n_neighbors=1, p=2)
    clf = clf.fit(X, Y)

    # use your test sample in this iteration to make the class prediction. For instance:
    # class_predicted = clf.predict([[1, 2]])[0]
    # --> add your Python code here
    class_predicted = clf.predict([testSample])[0]
    # print(class_predicted)

    # compare the prediction with the true label of the test instance to start calculating the error rate.
    if ((class_predicted == 1) and (db[i][num_attributes] == "-")) or ((class_predicted == 2) and (db[i][num_attributes]  == "+")):
        errorNum = errorNum + 1


# print the error rate
print("Error rate= " + str(errorNum/len(db)))
