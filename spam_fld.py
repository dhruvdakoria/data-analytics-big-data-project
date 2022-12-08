import time
from numpy import *
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# Use a comma delimiter to parse the data in spam2.txt
rawTrainingData = loadtxt('spam2.txt', delimiter=',')

# split the data into attributes and class labels
X = rawTrainingData[:, 0:57]  # attributes
y = rawTrainingData[:, 57]  # class labels

# split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

# create an instance of the LinearDiscriminantAnalysis class
lda = LinearDiscriminantAnalysis()

# fit and transform the training data using Fisher's linear discriminant
X_train_transformed = lda.fit_transform(X_train, y_train)

# Timing The Code
start_time = time.time()

# train a LinearSVC classifier using the transformed data
clf = LinearSVC()
clf.fit(X_train_transformed, y_train)

# predict the class labels for the test data
y_pred = clf.predict(lda.transform(X_test))

# evaluate the performance of the classifier
elapsed_time = time.time() - start_time
y_true = y_test  # actual class labels
accuracy = clf.score(lda.transform(X_test), y_test)

# print the confusion matrix and the elapsed time
print("Confusion matrix is: ")
print(confusion_matrix(y_true, y_pred))


print("\n", "Elapsed Time = ", elapsed_time*1000, " ms.")
print("Accuracy: ", accuracy*100, "%")
