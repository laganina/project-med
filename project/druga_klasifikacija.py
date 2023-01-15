import tensorflow
import keras
import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn import metrics
from project.podaci_drugi import df_basic, df_rankin
from sklearn.metrics import precision_score, recall_score, f1_score



# training and testing
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(df_basic, df_rankin, test_size = 0.2)


clf = svm.SVC(kernel='linear', C=2)
clf.fit(x_train, y_train)


y_pred = clf.predict(x_test)

acc = metrics.accuracy_score(y_test, y_pred)
print('Accuracy: ', acc)

print('Precision: %.3f' % precision_score(y_test, y_pred, average='micro'))
print('Recall: %.3f' % recall_score(y_test, y_pred, average='micro'))
print('F1 Score: %.3f' % f1_score(y_test, y_pred, average='micro'))

