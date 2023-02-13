from podaci_prvi import obelezja, labela                # ******************************
import numpy as np
from sklearn import svm
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict

random_state = np.random.RandomState(0)

print('ulazi: ')
print(obelezja)
print(obelezja.info())

y = labela
y = y.values.ravel()
klase = np.unique(y)
print(f'klase: {klase}')

for i in range(len(klase)):
    broj_uzoraka = sum(y == klase[i])
    print(f'Broj uzoraka u {i}-toj klasi je: {broj_uzoraka}')

# probaj da skaliras obelezja:
# select the columns to exclude from standard scaling
int_cols = obelezja.select_dtypes(include=['int'])
# create a list of columns to scale
cols_to_scale = [col for col in obelezja.columns if col not in int_cols]
# create a StandardScaler instance
scaler = StandardScaler()
# fit and transform the dataframe
obelezja[cols_to_scale] = scaler.fit_transform(obelezja[cols_to_scale])



random_state = np.random.RandomState(0)
clf = svm.SVC(kernel='linear', C = 21, probability = True, random_state=random_state)

scorer = cross_val_score(clf, obelezja, y, cv=5, scoring='accuracy')
print(f'tacnost: {scorer}')
acc = np.mean(scorer)
print(f'mean acc: {np.mean(scorer)}')
print(f'*************************************************')

scorer = cross_val_score(clf, obelezja, y, cv=5, scoring='precision')
print(f'preciznost: {scorer}')
prec = np.mean(scorer)
print(f'mean prec: {np.mean(scorer)}')
print(f'*************************************************')

scorer = cross_val_score(clf, obelezja, y, cv=5, scoring='recall')
print(f'recall: {scorer}')
recall = np.mean(scorer)
print(f'mean recall: {np.mean(scorer)}')
print(f'*************************************************')

scorer = cross_val_score(clf, obelezja, y, cv=5, scoring='f1')
print(f'f1: {scorer}')
f1 = np.mean(scorer)
print(f'mean f1: {np.mean(scorer)}')
print(f'*************************************************')

scorer = cross_val_score(clf, obelezja, y, cv=5, scoring='roc_auc')
print(f'roc_auc: {scorer}')
auc = np.mean(scorer)
print(f'mean auc: {np.mean(scorer)}')
print('*************************************************')

## ****************** ADD CONFUSION MATRIX ******************** ##
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import ConfusionMatrixDisplay 
from sklearn.metrics import roc_curve
from sklearn.metrics import RocCurveDisplay


# clf = svm.SVC(kernel='linear', C = 21)

X = obelezja

# Fit your model using cross_val_predict
y_pred = cross_val_predict(clf, X, y, cv=5)
print(f'y shape: {y.shape}')
print(f'y_pred shape: {y_pred.shape}')

# Calculate the confusion matrix
conf_mat = confusion_matrix(y, y_pred)
print(conf_mat)

# Plot the confusion matrix
# plt.matshow(conf_mat, cmap=plt.cm.gray)
# plt.show()

# cm_display = ConfusionMatrixDisplay(conf_mat).plot()

# Plot the confusion matrix
disp = ConfusionMatrixDisplay(conf_mat).plot()
disp.ax_.set_xlabel('Предвиђена класа')
disp.ax_.set_ylabel('Истинита класа')

# Customize the plot
# disp.figure_.suptitle("Confusion Matrix Plot")
# disp.ax_.set_xlabel("Predicted Label")
# disp.ax_.set_ylabel("True Label")

# Show the plot
plt.title('Матрица конфузије, препознавање РНП-а, МВН класификатор')
plt.savefig('Confusion matrix SVM.png')            # radi i jedno i drugo - ovo zauzima manje memorije 
# disp.figure_.savefig('conf_mat_SVM.png',dpi=300)   # radi 
plt.show()

# fpr, tpr, _ = roc_curve(y, y_pred, pos_label=1)          # 1 NIJE DOSLO DO POBOLJSANJA 
# roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr).plot()
# plt.title('ROC curve, SVM') 
# # plt.savefig('ROC curve SVM.png')                          # radi i jedno i drugo 
# roc_display.figure_.savefig('roc_SVM.png',dpi=300)      # radi 
# plt.show()    


# ************************* DRUGI NACIN: **********************************    # NIJE DOBAR ZBOG AUC-A, OSTALI PARAMETRI JESU ISTI 
# Calculate accuracy
# acc = accuracy_score(y, y_pred)
# print(f'acc: {acc}')

# # Calculate precision
# prec = precision_score(y, y_pred)
# print(f'prec: {prec}')

# # Calculate recall
# rec = recall_score(y, y_pred)
# print(f'rec: {rec}')

# # Calculate F1 score
# f1 = f1_score(y, y_pred)
# print(f'f1: {f1}')

# # Calculate AUC
# auc = roc_auc_score(y, y_pred)
# print(f'auc: {auc}')


raw_data = {'accuracy': [acc],
                'precision': [prec],
                'recall': [recall],
                'f1': [f1],
                'auc': [auc]}

df = pd.DataFrame(raw_data, columns = ['accuracy', 'precision', 'recall', 'f1', 'auc'])
df = df.round(decimals=2)
print(df)
df.to_csv('svm_rnp.csv', index=False)


# ***************** The AUC can be used as a single scalar metric to evaluate the classifier's performance. ************************

# *************************** ROC AUC *************************************
# The predicted class probability for a Support Vector Machine (SVM) classifier is calculated 
# by making use of the decision function of the SVM. The decision function is a measure of how
#  far a sample is from the boundary that separates the classes in feature space. A sample with 
# a positive decision function score is predicted as the positive class, while a sample with a 
# negative decision function score is predicted as the negative class.

# The SVM decision function is a linear combination of the support vectors, weighted by the corresponding
#  Lagrange multipliers. By applying the sigmoid function to the decision function score, the predicted
#  class probability can be obtained. The sigmoid function maps the score to the range [0, 1], where 0 
# corresponds to the negative class and 1 corresponds to the positive class.

# In summary, the predicted class probability for an SVM classifier is obtained by transforming the decision 
# function score using the sigmoid function. The decision function score is a linear combination of the support 
# vectors, weighted by the corresponding Lagrange multipliers.