# -------------------------------------------------------------------------
# AUTHOR: Nhat Tran
# FILENAME: decision_tree.py
# SPECIFICATION: train, test, and output the performance of the models
# FOR: CS 4210- Assignment #2
# TIME SPENT: 3
# -----------------------------------------------------------*/

# IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to
# work here only with standard vectors and arrays

# importing some Python libraries
from sklearn import tree
import csv

dataSets = ['contact_lens_training_1.csv', 'contact_lens_training_2.csv', 'contact_lens_training_3.csv']

for ds in dataSets:

    dbTraining = []
    X = []
    Y = []

    # reading the training data in a csv file
    with open(ds, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for i, row in enumerate(reader):
            if i > 0:  # skipping the header
                dbTraining.append(row)

    # transform the original training features to numbers and add to the 4D array X. For instance Young = 1,
    # Prepresbyopic = 2, Presbyopic = 3, so X = [[1, 1, 1, 1], [2, 2, 2, 2], ...]]
    num_attributes = 4
    dim1, dim2 = (num_attributes, len(dbTraining))
    X = [[0 for i in range(dim1)] for j in range(dim2)]
    for k in range(len(dbTraining)):
        for j in range(len(dbTraining[k]) - 1):
            if dbTraining[k][j] == "Young":
                X[k][j] = 1
            elif dbTraining[k][j] == "Prepresbyopic":
                X[k][j] = 2
            elif dbTraining[k][j] == "Presbyopic":
                X[k][j] = 3
            elif dbTraining[k][j] == "Myope":
                X[k][j] = 1
            elif dbTraining[k][j] == "Hypermetrope":
                X[k][j] = 2
            elif dbTraining[k][j] == "No":
                X[k][j] = 2
            elif dbTraining[k][j] == "Yes":
                X[k][j] = 1
            elif dbTraining[k][j] == "Reduced":
                X[k][j] = 2
            elif dbTraining[k][j] == "Normal":
                X[k][j] = 1
    # transform the original training classes to numbers and add to the vector Y. For instance Yes = 1, No = 2,
    # so Y = [1, 1, 2, 2, ...]
    Y = ["0"] * len(dbTraining)
    for k in range(len(dbTraining)):
        if dbTraining[k][4] == "No":
            Y[k] = 2
        elif dbTraining[k][4] == "Yes":
            Y[k] = 1

    # print("X=")
    # print(X)
    # print("Y=")
    # print(Y)
    Accuracy = []
    # loop your training and test tasks 10 times here
    for i in range(10):

        # fitting the decision tree to the data setting max_depth=3
        clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=3)
        clf = clf.fit(X, Y)

        # read the test data and add this data to dbTest
        dbTest = []
        with open('contact_lens_test.csv', 'r') as csvfile:
            reader = csv.reader(csvfile)
            for k, row in enumerate(reader):
                if k > 0:  # skipping the header
                    dbTest.append(row)

        TruePosNeg = 0
        for data in dbTest:
            # transform the features of the test instances to numbers following the same strategy done during
            # training, and then use the decision tree to make the class prediction.
            # For instance: class_predicted = clf.predict([[3, 1, 2, 1]])[0]
            # ->[0] is used to get an integer as the predicted class label so you can compare it with the true label
            row = ["0"] * (len(data) - 1)
            for e in range(len(data) - 1):
                if data[e] == "Young":
                    row[e] = 1
                elif data[e] == "Prepresbyopic":
                    row[e] = 2
                elif data[e] == "Presbyopic":
                    row[e] = 3
                elif data[e] == "Myope":
                    row[e] = 1
                elif data[e] == "Hypermetrope":
                    row[e] = 2
                elif data[e] == "No":
                    row[e] = 2
                elif data[e] == "Yes":
                    row[e] = 1
                elif data[e] == "Reduced":
                    row[e] = 2
                elif data[e] == "Normal":
                    row[e] = 1

            class_predicted = clf.predict([row])[0]
            # print(class_predicted)
            # compare the prediction with the true label (located at data[4]) of the test instance to start
            # calculating the accuracy.
            if ((class_predicted == 1) and (data[4] == "Yes")) or ((class_predicted == 2) and (data[4] == "No")):
                TruePosNeg = TruePosNeg + 1

        # find the lowest accuracy of this model during the 10 runs (training and test set)
        Accuracy.append(TruePosNeg / len(dbTest))

    # print the lowest accuracy of this model during the 10 runs (training and test set) and save it.
    # your output should be something like that: final accuracy when training on contact_lens_training_1.csv: 0.2
    # print(Accuracy)
    print("Final accuracy when training on " + ds + ": " + str(min(Accuracy)))
