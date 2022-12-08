import time
from numpy import *
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# Use a comma delimiter to parse the data in spam2.txt
rawTrainingData = loadtxt('spam2.txt', delimiter=',')

# Creating a classifier using LinearSVC
clf = LinearSVC(dual=False)

# Splitting the training data into attributes and class
X = rawTrainingData[:, 0:57]
y = rawTrainingData[:, 57]

# Splitting the data into train and test sets with a test size of 25%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

# Timing the code
start_time = time.time()

# Fitting the model using the training data
clf.fit(X_train, y_train)

# Predicting the classes using the test data
y_pred = clf.predict(X_test)

# Calculating the elapsed time
elapsed_time = time.time() - start_time

# Setting the actual classes to be the test classes
y_true = y_test

# Calculating the accuracy of the model
accuracy = clf.score(X_test, y_test)

# Printing the confusion matrix and evaluation metrics
print("Confusion matrix is: ")
print(confusion_matrix(y_true, y_pred))
print("\n", "Elapsed Time = ", elapsed_time*1000, " ms.")
print("Accuracy: ", accuracy*100, "%")
