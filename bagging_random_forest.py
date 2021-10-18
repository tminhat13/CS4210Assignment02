# -------------------------------------------------------------------------
# AUTHOR: Nhat Tran
# FILENAME: CS4210Assignment03
# SPECIFICATION: description of the program
# FOR: CS 4210- Assignment #3
# TIME SPENT: 3 hours
# -----------------------------------------------------------*/

# IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to
# work here only with standard vectors and arrays

# importing some Python libraries
from operator import __index__

from sklearn import tree
from sklearn.utils import resample
from sklearn.ensemble import RandomForestClassifier
import csv

dbTraining = []
dbTest = []
X_training = []
Y_training = []
classVotes = []  # this array will be used to count the votes of each classifier

# reading the training data in a csv file
with open('optdigits.tra', 'r') as trainingFile:
    reader = csv.reader(trainingFile)
    for i, row in enumerate(reader):
        dbTraining.append(row)

# reading the test data in a csv file
with open('optdigits.tes', 'r') as testingFile:
    reader = csv.reader(testingFile)
    for i, row in enumerate(reader):
        dbTest.append(row)
        classVotes.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])  # inititalizing the class votes for each test sample

    print("Started my base and ensemble classifier ...")
    accuracy = []
    for k in range(
            20):  # we will create 20 bootstrap samples here (k = 20). One classifier will be created for each
        # bootstrap sample

        bootstrapSample = resample(dbTraining, n_samples=len(dbTraining), replace=True)

        # populate the values of X_training and Y_training by using the bootstrapSample
        # --> add your Python code here
        num_attributes = 64
        dim1, dim2 = (num_attributes, len(dbTraining))
        X_training = [[0 for j in range(dim1)] for k in range(dim2)]
        for i in range(len(dbTraining)):
            for j in range(len(dbTraining[i]) - 1):
                X_training[i][j] = dbTraining[i][j]
        Y_training = ["0"] * len(dbTraining)
        for i in range(len(dbTraining)):
            Y_training[i] = dbTraining[i][num_attributes]

        # fitting the decision tree to the data
        clf = tree.DecisionTreeClassifier(criterion='entropy',
                                          max_depth=None)  # we will use a single decision tree without pruning it
        clf = clf.fit(X_training, Y_training)

        TruePosNeg = 0

        for i, testSample in enumerate(dbTest):

            # make the classifier prediction for each test sample and update the corresponding index value in
            # classVotes. For instance, if your first base classifier predicted 2 for the first test sample,
            # then classVotes[0,0,0,0,0,0,0,0,0,0] will change to classVotes[0,0,1,0,0,0,0,0,0,0]. Later,
            # if your second base classifier predicted 3 for the first test sample, then classVotes[0,0,1,0,0,0,0,0,
            # 0,0] will change to classVotes[0,0,1,1,0,0,0,0,0,0] Later, if your third base classifier predicted 3
            # for the first test sample, then classVotes[0,0,1,1,0,0,0,0,0,0] will change to classVotes[0,0,1,2,0,0,
            # 0,0,0,0] this array will consolidate the votes of all classifier for all test samples --> add your
            # Python code here
            row = ["0"] * (len(testSample) - 1)
            for e in range(len(testSample) - 1):
                row[e] = testSample[e]

            class_predicted = clf.predict([row])[0]
            predict_value = int(class_predicted)

            classVotes[i][predict_value] = classVotes[i][predict_value]+1

            if k == 0:  # for only the first base classifier, compare the prediction with the true label of the test
                # sample here to start calculating its accuracy
                if class_predicted == testSample[num_attributes]:
                    TruePosNeg = TruePosNeg + 1

        if k == 0:  # for only the first base classifier, print its accuracy here
            # --> add your Python code here
            accuracy.append(TruePosNeg / len(dbTest))
            print("Finished my base classifier (fast but relatively low accuracy) ...")
            print("My base classifier accuracy: " + str(accuracy))
            print("")

    # now, compare the final ensemble prediction (majority vote in classVotes) for each test sample with the ground
    # truth label to calculate the accuracy of the ensemble classifier (all base classifiers together) --> add your
    # Python code here
    accuracy.clear()
    TruePosNeg = 0
    for i, row in enumerate(classVotes):
        max_value = max(row)
        location = row.index(max_value)
        if location == int(dbTest[i][num_attributes]):
            TruePosNeg = TruePosNeg + 1
    accuracy.append(TruePosNeg / len(dbTest))

    # printing the ensemble accuracy here
    print("Finished my ensemble classifier (slow but higher accuracy) ...")
    print("My ensemble accuracy: " + str(accuracy))
    print("")

    print("Started Random Forest algorithm ...")

    # Create a Random Forest Classifier
    clf = RandomForestClassifier(
        n_estimators=20)  # this is the number of decision trees that will be generated by Random Forest. The sample
    # of the ensemble method used before

    # Fit Random Forest to the training data
    clf.fit(X_training, Y_training)

    # make the Random Forest prediction for each test sample. Example: class_predicted_rf = clf.predict([[3, 1, 2, 1,
    # ...]] --> add your Python code here
    TruePosNeg = 0
    accuracy.clear()
    for data in dbTest:
        row = ["0"] * (len(data) - 1)
        for e in range(len(data) - 1):
            row[e]=data[e]
        class_predicted_rf = clf.predict([row])[0]
        if class_predicted_rf == data[num_attributes]:
            TruePosNeg = TruePosNeg + 1
    accuracy.append(TruePosNeg / len(dbTest))
    # compare the Random Forest prediction for each test sample with the ground truth label to calculate its accuracy
    # --> add your Python code here

    # printing Random Forest accuracy here
    print("Random Forest accuracy: " + str(accuracy))

    print("Finished Random Forest algorithm (much faster and higher accuracy!) ...")
