import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn import metrics
# from project.podaci_drugi import df, df_rankin
from proba2 import df, df_rankin
from sklearn.metrics import precision_score, recall_score, f1_score

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

# MOZEMO DA ISPITAMO PODELU NA 4 KLASE:
# 0 posebno, 1 posebno, 2 i 3 zajedno, 4, 5 i 6 zajedno - tacnost je 65 % 
# y = np.where(np.logical_and(y >= 2, y <= 3), 2, y) 
# y = np.where(np.logical_and(y >= 4, y <= 6), 3, y) 
# Broj uzoraka u 0-toj klasi je: 71
# Broj uzoraka u 1-toj klasi je: 87
# Broj uzoraka u 2-toj klasi je: 92
# Broj uzoraka u 3-toj klasi je: 88


# MOZEMO DA ISPITAMO PODELU NA 3 KLASE:     # OVO JE MOZDA I NAJBOLJE, IMAMO TRI STANJA PACIJENTA   
# y = np.where(y<=1, 0, y)   # 0,1 - 0                                # tacnost je 80 % 
# y = np.where(np.logical_and(y > 1, y < 4), 1, y)   # 2 i 3
# y = np.where(np.logical_and(y >= 4, y <= 6), 2, y)   # 4, 5 i 6
# Broj uzoraka u 0-toj klasi je: 158
# Broj uzoraka u 1-toj klasi je: 92
# Broj uzoraka u 2-toj klasi je: 88


# I MOZEMO DA ISPITAMO PODELU NA 2 KLASE:      
# y = np.where(np.logical_and(y >= 0, y <= 2), 0, y)   # 0 1 2         # tacnost 91 %                       
# y = np.where(np.logical_and(y >= 3, y <= 6), 1, y)   # 3 4 5 6

# tada je broj uzoraka sledeci:
# Broj uzoraka u 0-toj klasi je: 71
# Broj uzoraka u 1-toj klasi je: 87






klase = np.unique(y)
print('klase: {klase}')

for i in range(len(klase)):
    broj_uzoraka = sum(y == klase[i])
    print(f'Broj uzoraka u {i}-toj klasi je: {broj_uzoraka}')

# skaliranje obelezja:
from sklearn.preprocessing import StandardScaler
# select the columns to exclude from standard scaling
int_cols = obelezja.select_dtypes(include=['int'])
# create a list of columns to scale
cols_to_scale = [col for col in obelezja.columns if col not in int_cols]
# create a StandardScaler instance
scaler = StandardScaler()
# fit and transform the dataframe
obelezja[cols_to_scale] = scaler.fit_transform(obelezja[cols_to_scale])


# training and testing - OVO JE AKO NE RADIMO KROSVALIDACIJU. ALI S OBZIROM NA MANJI BROJ UZORAKA, 
# KROSVALIDACIJA JE NAJBOLJE RESENJE  
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(obelezja, y, test_size = 0.2)


clf = svm.SVC(kernel='linear') #, C=2)
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)

acc = metrics.accuracy_score(y_test, y_pred)
print('Accuracy: ', acc)

# Target is multiclass but average='binary'. Please choose
# another average setting, one of [None, 'micro', 'macro', 'weighted']
# binary is for binary targets, 

# 'micro': 
# Calculates metrics globally by counting the total true positives,
# false negatives and false positives.

# 'macro': 
# Calculate metrics for each label, and find their unweighted mean.
#  This does not take label imbalance into account.

# 'weighted':
# Calculate metrics for each label, and find their average weighted by support 
# (the number of true instances for each label). This alters ‘macro’ to account
#  for label imbalance; it can result in an F-score that is not between precision
#   and recall.

# 'samples':
# Calculate metrics for each instance, and find their average (only meaningful 
# for multilabel classification where this differs from accuracy_score).



print('Precision: %.3f' % precision_score(y_test, y_pred, average='micro'))
print('Recall: %.3f' % recall_score(y_test, y_pred, average='micro'))
print('F1 Score: %.3f' % f1_score(y_test, y_pred, average='micro'))

# fali krosvalidacija 



