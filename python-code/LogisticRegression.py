from numpy import *
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score
import time

# Use a comma delimiter to parse the data in spam.txt
rawTrainingData = loadtxt('spam.txt', delimiter=',')

# Splitting the training data into attributes and class labels
X = rawTrainingData[:, 0:57] # attributes
y = rawTrainingData[:, 57] # class labels

# split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

# Ensure that at least 750 data points are used for training
if X_train.shape[0] < 750:
    raise ValueError("Not enough data points for training. Please use a larger dataset.")

# Initialize the logistic regression model
logistic = LogisticRegression(max_iter=2000)

# Train the model on the training data
start_time = time.perf_counter()
logistic.fit(X_train, y_train)
end_time = time.perf_counter()
training_time = (end_time - start_time) * 1000 # convert to milliseconds

# Test the model on the testing data
start_time = time.perf_counter()
y_pred = logistic.predict(X_test)
end_time = time.perf_counter()
testing_time = (end_time - start_time) * 1000 # convert to milliseconds

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)

# Calculate the confusion matrix to evaluate the model's performance
confusion_matrix = confusion_matrix(y_test, y_pred)

# Calculate ROC curve and AUC
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)

# Perform Cross Validation for parameter selection
scores = cross_val_score(logistic, X_train, y_train, cv=5)
mean_score = mean(scores)

# Print the computational times for training and testing, as well as the confusion matrix, accuracy, and ROC curve
print(f"Training time: {training_time:.2f} milliseconds")
print(f"Testing time: {testing_time:.2f} milliseconds")
print(f"Confusion matrix:\n{confusion_matrix}")
print(f"Accuracy Score: {accuracy:.2f}")
print(f"ROC AUC: {roc_auc:.2f}")
print(f"Mean cross-validation score: {mean_score:.2f}")

#-------------------------- Logistic Regression --------------------------

#Run 1
# Training time: 1893.53 milliseconds
# Testing time: 0.43 milliseconds
# Confusion matrix:
# [[661  48]
#  [ 54 388]]
# Accuracy Score: 0.91
# ROC AUC: 0.91
# Mean cross-validation score: 0.93
#
# Run 2
# Training time: 1648.58 milliseconds
# Testing time: 0.46 milliseconds
# Confusion matrix:
# [[680  30]
#  [ 50 391]]
# Accuracy Score: 0.93
# ROC AUC: 0.92
# Mean cross-validation score: 0.93
#
# Run 3
# Training time: 1937.29 milliseconds
# Testing time: 0.43 milliseconds
# Confusion matrix:
# [[665  26]
#  [ 56 404]]
# Accuracy Score: 0.93
# ROC AUC: 0.92
# Mean cross-validation score: 0.93