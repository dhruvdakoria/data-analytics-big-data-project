from numpy import *
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import time

# Use a comma delimiter to parse the data in spam.txt
rawTrainingData = loadtxt('spam.txt', delimiter=',')

# Splitting the training data into attributes and class labels
X = rawTrainingData[:, 0:57]  # attributes
y = rawTrainingData[:, 57]  # class labels

# split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

# Ensure that at least 750 data points are used for training
if X_train.shape[0] < 750:
    raise ValueError("Not enough data points for training. Please use a larger dataset.")

# Initialize the linear SVM model
svm = LinearSVC(dual=False)

# Train the model on the training data
start_time = time.perf_counter()
svm.fit(X_train, y_train)
end_time = time.perf_counter()
training_time = (end_time - start_time) * 1000  # convert to milliseconds

# Test the model on the testing data
start_time = time.perf_counter()
y_pred = svm.predict(X_test)
end_time = time.perf_counter()
testing_time = (end_time - start_time) * 1000  # convert to milliseconds

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)

# Calculate the confusion matrix to evaluate the model's performance
confusion_matrix = confusion_matrix(y_test, y_pred)

# Print the computational times for training and testing, as well as the confusion matrix and accuracy
print(f"Training time: {training_time:.2f} milliseconds")
print(f"Testing time: {testing_time:.2f} milliseconds")
print(f"Confusion matrix:\n{confusion_matrix}")
print(f"Accuracy Score: {accuracy:.2f}")


# Output:

# -----Run 1------ #
# $ python linear-svm.py
# Training time: 135.87 milliseconds
# Testing time: 2.34 milliseconds
# Confusion matrix:
# [[645  28]
#  [ 54 424]]
# Accuracy Score: 0.93

# -----Run 2------ #
# $ python linear-svm.py 
# Training time: 101.40 milliseconds
# Testing time: 2.08 milliseconds
# Confusion matrix:
# [[686  31]
#  [ 48 386]]
# Accuracy Score: 0.93

# -----Run 3------ #
# $ python linear-svm.py 
# Training time: 74.96 milliseconds
# Testing time: 2.18 milliseconds
# Confusion matrix:
# [[682  28]
#  [ 52 389]]
# Accuracy Score: 0.93