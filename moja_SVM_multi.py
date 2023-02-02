import pandas as pd
import numpy as np
from sklearn import svm
# from project.podaci_drugi import df, df_rankin
from podaci_drugi import obelezja, labela
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.multiclass import OneVsRestClassifier    # ovo je za multilabel klasifikaciju


y = labela

y = y.values.ravel()


y = np.where(y<=1, 0, y)   # 0,1 - 0
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



# select the columns to exclude from standard scaling
int_cols = obelezja.select_dtypes(include=['int'])
# create a list of columns to scale
cols_to_scale = [col for col in obelezja.columns if col not in int_cols]
# create a StandardScaler instance
scaler = StandardScaler()
# fit and transform the dataframe
obelezja[cols_to_scale] = scaler.fit_transform(obelezja[cols_to_scale])



# create an instance of the model
clf = OneVsRestClassifier(svm.SVC(probability=True))


scorer = cross_val_score(clf, obelezja, y, cv=5, scoring='accuracy')
print(f'tacnost: {scorer}')
print(f'mean acc: {np.mean(scorer)}')
print(f'*************************************************')

scorer = cross_val_score(clf, obelezja, y, cv=5, scoring='precision_macro')
print(f'preciznost: {scorer}')
print(f'mean prec: {np.mean(scorer)}')
print(f'*************************************************')

scorer = cross_val_score(clf, obelezja, y, cv=5, scoring='recall_macro')
print(f'recall: {scorer}')
print(f'mean recall: {np.mean(scorer)}')
print(f'*************************************************')

scorer = cross_val_score(clf, obelezja, y, cv=5, scoring='f1_macro')
print(f'f1: {scorer}')
print(f'mean f1: {np.mean(scorer)}')
print(f'*************************************************')

scorer = cross_val_score(clf, obelezja, y, cv=5, scoring='roc_auc_ovr')
print(f'roc_auc: {scorer}')
print(f'mean auc: {np.mean(scorer)}')
print(f'*************************************************')
# The average='macro' is used because it's multi-label classification.

# ****************************************
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score
# roc auc na drugi nacin:
print('ROC AUC NA DRUGI NACIN:')
# Assume clf is your SVM classifier
# from sklearn.preprocessing import MultiLabelBinarizer
# mlb = MultiLabelBinarizer()
# y = mlb.fit_transform(y)    # stavlja svaku klasu u posebnu kolonu

# Assume y is your current labels in a single column
# Create a dataframe from y
df = pd.DataFrame(y, columns=['labels'])

# Use the 'str.get_dummies' method to convert the 'labels' column to multiple columns
df = pd.get_dummies(df, columns=['labels'])
y = df.values



clf = OneVsRestClassifier(svm.SVC(probability=True))
predictions = cross_val_predict(clf, obelezja, y, cv=5, method='predict_proba')

# Compute AUC for each class
print('y shape:')
print(y.shape)
aucs = []
for i in range(y.shape[1]):
    auc = roc_auc_score(y[:, i], predictions[:, i])
    aucs.append(auc)

print("AUC scores:", aucs)    # ovo je za svaku klasu
print(f'mean auc: {np.mean(aucs)}')
print(f'*************************************************')






