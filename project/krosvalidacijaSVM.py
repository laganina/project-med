import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn import metrics
# from project.podaci_drugi import df, df_rankin
# from proba2 import df, df_rankin
from proba import obelezja, labela
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier



merged = obelezja
print('ulazi: ')
print(merged)
print(merged.info())


# OVDE MOZES DA IZDVAJAS STA ULAZI U KLASIFIKATOR:
# merged = merged[['STAROST', 'NIHSS na prijemu', 'ASPECTS']]
# merged = merged[['TIP CVI', 'Clopidogrel', 'HLP']]
merged = merged[['STAROST', 'Glikemija', 'MAP', 'TIP CVI', 'ASA', 'Clopidogrel', 'AntiHTA', 'DM', 'Pušenje', 'HLP']]

# merged = merged[['NIHSS na prijemu']]  
# merged = merged[['ASPECTS']]              
# merged = merged[['STAROST']]           
# merged = merged[['STAROST', 'NIHSS na prijemu', 'ASPECTS', 'TIP CVI','TOAST']]






y = labela 
y = y.values.ravel()
klase = np.unique(y)
print(f'klase: {klase}')

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

# clf = DecisionTreeClassifier() # svm je ipak bolji 

# create the instance of the model: 
clf = svm.SVC(C = 23.563182541616264, gamma = 0.001, kernel='linear')      # isti su rezultati i bez promene C, gamma 
# clf = svm.SVC(C=1000.0, gamma=0.001, kernel='linear')   # tacnost 0.62, malo bolji f1, a rac. zahtevno 

# clf = svm.SVC(kernel='poly')   # tacnost 62 % 

# clf = svm.SVC(C = 1000, kernel='poly')    # losa tacnost  

# clf = svm.SVC(C = 1000, gamma = 0.001, kernel='poly')  # kernel mora biti poly, i gamma faktor je iskoriscen takodje 
# dobija se tacnot od 63 posto 

scorer = cross_val_score(clf, obelezja, y, cv=5, scoring='accuracy')
print(f'tacnost: {scorer}')
print(f'mean acc: {np.mean(scorer)}')
print(f'*************************************************')

scorer = cross_val_score(clf, obelezja, y, cv=5, scoring='precision')
print(f'preciznost: {scorer}')
print(f'mean prec: {np.mean(scorer)}')
print(f'*************************************************')

scorer = cross_val_score(clf, obelezja, y, cv=5, scoring='recall')
print(f'recall: {scorer}')
print(f'mean recall: {np.mean(scorer)}')
print(f'*************************************************')

scorer = cross_val_score(clf, obelezja, y, cv=5, scoring='f1')
print(f'f1: {scorer}')
print(f'mean f1: {np.mean(scorer)}')
print(f'*************************************************')

scorer = cross_val_score(clf, obelezja, y, cv=5, scoring='roc_auc')
print(f'roc_auc: {scorer}')
print(f'mean auc: {np.mean(scorer)}')
print(f'*************************************************')
# The average='macro' is used because it's multi-label classification.






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


# averaging:

# Macro-averaged: This method computes the average of the metric computed
#  for each class, independently of the others.

# Micro-averaged: This method computes the metric considering all classes 
# together, by counting the total number of true positives, false positives 
# and false negatives.

# The choice of averaging method will depend on the specific characteristics
#  of the dataset, and the goals of the classification task. For example, if
#   you want to penalize all types of errors equally, you may want to use Hamming 
#   loss or EMR. If you want to balance precision and recall, you may want to use 
#   F1-Score. If you want to have an idea of how many instances are classified correctly, 
#   you may want to use EMR. If you want to have an idea of the performance of each class
#    independently, you may want to use Macro-averaged method. If you want to have an idea
#     of the performance of the model, you may want to use Micro-averaged method.

# It's always a good idea to experiment with different metrics and averaging methods, and choose
#  the one that gives the best results for your specific use case.
