import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn import metrics
# from project.podaci_drugi import df, df_rankin
from podaci_drugi import df, df_rankin
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, roc_curve

# IMAMO UKUPNO 6 KLASA - TACNOST JE NESTO VECA OD 50 POSTO
# Broj uzoraka u 0-toj klasi je: [71]
# Broj uzoraka u 1-toj klasi je: [87]
# Broj uzoraka u 2-toj klasi je: [47]
# Broj uzoraka u 3-toj klasi je: [45]
# Broj uzoraka u 4-toj klasi je: [38]
# Broj uzoraka u 5-toj klasi je: [19]
# Broj uzoraka u 6-toj klasi je: [31]

# GLEDAMO DA IH RASPODELIMO TAKO DA U SVIM BUDE PRIBLIZNO PODJEDNAK BROJ UZORAKA


# DODAJEM
obelezja = df
y = df_rankin

y = y.values.ravel()   # flattens the numpy array


# MOZEMO DA ISPITAMO PODELU NA 3 KLASE:     # za pocetak se mozemo opredeliti na 3 klase
y = np.where(y<=1, 0, y)   # 0,1 - 0                                # tacnost je 80 %
y = np.where(np.logical_and(y > 1, y < 4), 1, y)   # 2 i 3
y = np.where(np.logical_and(y >= 4, y <= 6), 2, y)   # 4, 5 i 6
# Broj uzoraka u 0-toj klasi je: 158
# Broj uzoraka u 1-toj klasi je: 92
# Broj uzoraka u 2-toj klasi je: 88



klase = np.unique(y)
print('klase: {klase}')

for i in range(len(klase)):
    broj_uzoraka = sum(y == klase[i])
    print(f'Broj uzoraka u {i}-toj klasi je: {broj_uzoraka}')

# probaj da skaliras obelezja:
from sklearn.preprocessing import StandardScaler
# select the columns to exclude from standard scaling
int_cols = obelezja.select_dtypes(include=['int'])
# create a list of columns to scale
cols_to_scale = [col for col in obelezja.columns if col not in int_cols]
# create a StandardScaler instance
scaler = StandardScaler()
# fit and transform the dataframe
obelezja[cols_to_scale] = scaler.fit_transform(obelezja[cols_to_scale])


# training and testing - ova podela nam sad ne treba
# x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(obelezja, y, test_size = 0.2)


# clf = svm.SVC(kernel='linear') #, C=2)
# clf.fit(x_train, y_train)
# y_pred = clf.predict(x_test)

from sklearn.model_selection import cross_val_score
from sklearn.multiclass import OneVsRestClassifier    # ovo je za multilabel klasifikaciju


# create an instance of the model
clf = OneVsRestClassifier(svm.SVC(probability=True))
# In the above example, we are using roc_auc_ovr as the scoring metric which
# stands for ROC AUC for one-vs-rest classification.

# It is also possible to use roc_auc_ovo as the scoring metric which stands for
#  ROC AUC for one-vs-one classification. But this will be much more computationally
#  expensive and may take longer to run.


# # perform 5-fold cross-validation
# scores = cross_val_score(clf, X, y, cv=5, scoring='roc_auc_ovr')

# # print the mean ROC AUC score
# print("ROC AUC: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


scorer = cross_val_score(clf, obelezja, y, cv=5, scoring='precision_micro')
print(f'preciznost: {scorer}')

scorer = cross_val_score(clf, obelezja, y, cv=5, scoring='recall_micro')
print(f'recall: {scorer}')

scorer = cross_val_score(clf, obelezja, y, cv=5, scoring='f1_micro')
print(f'f1: {scorer}')

scorer = cross_val_score(clf, obelezja, y, cv=5, scoring='roc_auc_ovr')
print(f'roc_auc: {scorer}')



# Ovo je dobra predstava performansi - The Classification Report
# https://towardsdatascience.com/evaluating-multi-label-classifiers-a31be83da6ea


# Neke beleske
# In multi-label classification, it's important to choose an evaluation metric that best describes
# the characteristics of the problem and the desired results. Some commonly used metrics for evaluating
# multi-label classification include:

# F1 Score: F1 score is the harmonic mean of precision and recall. It balances the trade-off between precision
# and recall and is commonly used in multi-label classification. The F1 score can be calculated using the f1_score
# function from sklearn.metrics module.

# Jaccard Score: Jaccard score measures the similarity between two sets of predicted labels and the true labels.
# It is the size of the intersection divided by the size of the union of the predicted and true labels. The Jaccard
#  score can be calculated using the jaccard_score function from sklearn.metrics module.

# Hamming Loss: Hamming loss is the fraction of the wrong labels to the total number of labels. It measures the
# number of different labels between the predicted labels and the true labels. The Hamming loss can be calculated
# using the hamming_loss function from sklearn.metrics module.

# Precision: Precision is the number of true positive predictions divided by the total number of positive predictions.
# It measures the ability of the classifier not to label as positive a sample that is negative. The Precision can be
# calculated using the precision_score function from sklearn.metrics module.

# Recall: Recall is the number of true positive predictions divided by the total number of actual positive samples.
# It measures the ability of the classifier to find all the positive samples. The Recall can be calculated using the
# recall_score function from sklearn.metrics module.

# ROC AUC: ROC AUC is a measure of the classifier's performance by plotting the true positive rate against the false
#  positive rate. It can be used for multi-label classification by treating each label as a binary classification problem
#   and then averaging the ROC AUC scores for each label.

# It is important to keep in mind that choosing the best metric may depend on the specific characteristics of the problem,
# the nature of the data, and the desired results. It is a good practice to try out different evaluation metrics and choose
#  the one that best suits the problem at hand.

