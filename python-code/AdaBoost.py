from numpy import *
from sklearn.ensemble import AdaBoostClassifier
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

trainedClassifier = AdaBoostClassifier(n_estimators=50, learning_rate=1)

# Train Adaboost Classifer
start_time = time.perf_counter()
model = trainedClassifier.fit(X_train, y_train)
end_time = time.perf_counter()
training_time = (end_time - start_time) * 1000

#Predict the response for test dataset
start_time = time.perf_counter()
y_pred = model.predict(X_test)
end_time = time.perf_counter()
testing_time = (end_time - start_time) * 1000

# Model Accuracy
accuracy_score = accuracy_score(y_test, y_pred)
confusion_matrix = confusion_matrix(y_test, y_pred)
# Calculate ROC curve and AUC
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)
# Perform Cross Validation for parameter selection
scores = cross_val_score(trainedClassifier, X_train, y_train, cv=5)
mean_score = mean(scores)

print(f"Training time: {training_time:.2f} milliseconds")
print(f"Testing time: {testing_time:.2f} milliseconds")
print(f"Accuracy: {accuracy_score:.2f}")
print("Confusion matrix:\n{confusion_matrix}")
print(f"ROC AUC: {roc_auc:.2f}")
print(f"Mean cross-validation score: {mean_score:.2f}")

#-------------------------- AdaBoost --------------------------

#Run 1
# Training time: 469.69 milliseconds
# Testing time: 20.33 milliseconds
# Accuracy: 0.95
# Confusion Matrix:
# [[655  29]
#  [ 32 435]]
# ROC AUC: 0.94
# Mean cross-validation score: 0.94
#
# Run 2
# Training time: 504.47 milliseconds
# Testing time: 22.25 milliseconds
# Accuracy: 0.93
# Confusion Matrix:
# [[652  36]
#  [ 40 423]]
# ROC AUC: 0.93
# Mean cross-validation score: 0.94
#
# Run 3
# Training time: 525.90 milliseconds
# Testing time: 21.85 milliseconds
# Accuracy: 0.94
# Confusion Matrix:
# [[662  37]
#  [ 27 425]]
# ROC AUC: 0.94
# Mean cross-validation score: 0.93