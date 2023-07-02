from numpy import *
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import time

# Use a comma delimiter to parse the data in spam.txt
rawTrainingData = loadtxt('spam.txt', delimiter=',')

# Splitting the training data into attributes and class labels
X = rawTrainingData[:, 0:57]  # attributes
y = rawTrainingData[:, 57]  # class labels

# Split the data into training and testing sets, reserving 25% of the data for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

# Ensure that at least 750 data points are used for training
if X_train.shape[0] < 750:
    raise ValueError("Not enough data points for training. Please use a larger dataset.")

# Initialize the Fisher's linear discriminant model
lda = LinearDiscriminantAnalysis()

# Train the model on the training data
start_time = time.perf_counter()
lda.fit(X_train, y_train)
end_time = time.perf_counter()
training_time = (end_time - start_time) * 1000  # convert to milliseconds

# Test the model on the testing data
start_time = time.perf_counter()
y_pred = lda.predict(X_test)
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
# $ python fishers-linear-discriminant.py
# Training time: 200.22 milliseconds
# Testing time: 1.65 milliseconds
# Confusion matrix:
# [[674  32]
#  [ 89 356]]
# Accuracy Score: 0.89

# -----Run 2------ #
# $ python fishers-linear-discriminant.py
# Training time: 227.48 milliseconds
# Testing time: 1.77 milliseconds
# Confusion matrix:
# [[666  37]
#  [ 90 358]]
# Accuracy Score: 0.89

# -----Run 3------ #
# $ python fishers-linear-discriminant.py 
# Training time: 202.99 milliseconds
# Testing time: 2.06 milliseconds
# Confusion matrix:
# [[653  26]
#  [106 366]]
# Accuracy Score: 0.89

