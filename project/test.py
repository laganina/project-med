import tensorflow
import keras
import pandas as pd
import numpy as np
import sklearn
from project import dejta_frejm
from sklearn import svm 
from sklearn import metrics 
from sklearn.model_selection import train_test_split
from project.dejta_frejm import df, y

# training and testing

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(df, y, test_size = 0.2)


clf = svm.SVC(kernel='linear', C=2)
clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)

acc = metrics.accuracy_score(y_test, y_pred)
print(acc)
