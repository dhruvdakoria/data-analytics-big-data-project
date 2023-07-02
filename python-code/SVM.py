from numpy import *
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score
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
svm = SVC()

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

# Calculate ROC curve and AUC
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)


# Perform Cross Validation for parameter selection
scores = cross_val_score(svm, X_train, y_train, cv=5)
mean_score = mean(scores)

# Print the computational times for training and testing, as well as the confusion matrix and accuracy
print(f"Training time: {training_time:.2f} milliseconds")
print(f"Testing time: {testing_time:.2f} milliseconds")
print(f"Confusion matrix:\n{confusion_matrix}")
print(f"Accuracy Score: {accuracy:.2f}")
print(f"ROC AUC: {roc_auc:.2f}")
print(f"Mean cross-validation score: {mean_score:.2f}")
# Output:

#-------------------------- Support Vector Machine --------------------------

#Run 1
# Training time: 645.98 milliseconds
# Testing time: 445.79 milliseconds
# Confusion matrix:
# [[579  96]
#  [255 221]]
# Accuracy Score: 0.70
# ROC AUC: 0.66
# Mean cross-validation score: 0.70
#
# Run 2
# Training time: 806.77 milliseconds
# Testing time: 438.24 milliseconds
# Confusion matrix:
# [[617  96]
#  [246 192]]
# Accuracy Score: 0.70
# ROC AUC: 0.65
# Mean cross-validation score: 0.70
#
# Run 3
# Training time: 767.29 milliseconds
# Testing time: 433.90 milliseconds
# Confusion matrix:
# [[622  66]
#  [254 209]]
# Accuracy Score: 0.72
# ROC AUC: 0.68
# Mean cross-validation score: 0.71