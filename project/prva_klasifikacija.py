import tensorflow
import keras
import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
from sklearn import svm 
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from project.podaci_prvi import merged, y
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, roc_curve

# training and testing, merged + y
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(merged, y, test_size = 0.3)


clf = svm.SVC(kernel='linear', C=2)
clf.fit(x_train, y_train)


y_pred = clf.predict(x_test)

clf_tree = DecisionTreeClassifier()
clf_reg = LogisticRegression()
clf_tree.fit(x_train, y_train)
clf_reg.fit(x_train, y_train)

y_score1 = clf_tree.predict_proba(x_test)[:, 1]
y_score2 = clf_reg.predict_proba(x_test)[:, 1]

false_positive_rate1, true_positive_rate1, threshold1 = roc_curve(y_test, y_score1)
false_positive_rate2, true_positive_rate2, threshold2 = roc_curve(y_test, y_score2)

print('roc_auc_score for DecisionTree: ', roc_auc_score(y_test, y_score1))
print('roc_auc_score for Logistic Regression: ', roc_auc_score(y_test, y_score2))

plt.subplots(1, figsize=(10, 10))
plt.title('Receiver Operating Characteristic - DecisionTree')
plt.plot(false_positive_rate1, true_positive_rate1)
plt.plot([0, 1], ls="--")
plt.plot([0, 0], [1, 0], c=".7"), plt.plot([1, 1], c=".7")
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

plt.subplots(1, figsize=(10, 10))
plt.title('Receiver Operating Characteristic - Logistic regression')
plt.plot(false_positive_rate2, true_positive_rate2)
plt.plot([0, 1], ls="--")
plt.plot([0, 0], [1, 0], c=".7"), plt.plot([1, 1], c=".7")
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
# it doesn’t tell us anything about the errors our machine learning models make on new data we haven’t seen before
# the same accuracy metrics for two different models may indicate different model performance towards different classes
# in case of imbalanced dataset, accuracy metrics is not the most effective metrics to be used
acc = metrics.accuracy_score(y_test, y_pred)
print(acc)

# in some cases, it may be more important to have a high precision (e.g. in medical diagnosis),
# while in others, a high recall may be more important (e.g. in fraud detection)
print('Precision: %.3f' % precision_score(y_test, y_pred))
print('Recall: %.3f' % recall_score(y_test, y_pred))

# to balance precision and recall, practitioners often use the F1 score, which is a combination of the two metric
# it can be difficult to determine the optimal balance between precision and recall for a given application
# useful measure of the model in the scenarios where one tries to optimize either of precision or recall score
# and as a result, the model performance suffers
print('F1 Score: %.3f' % f1_score(y_test, y_pred))

# a graphical plot that illustrates the diagnostic ability of
# a binary classifier system as its discrimination threshold is varied


