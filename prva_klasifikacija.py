import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from podaci_prvi import obelezja, labela, joined_basic
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from scipy import stats

# Increase the number of iterations (max_iter) or scale the data as shown in:
#     https://scikit-learn.org/stable/modules/preprocessing.html
# SKALIRANJE JE ODRADJENO

from sklearn.preprocessing import StandardScaler
import pandas as pd


merged = obelezja
# merged = joined_basic
y = labela

# URADJENO JE SKALIRANJE, TREBALO BI URADITI KROSVALIDACIJU. SREDITI KOD, PO FUNKCIJAMA.
# ODABRATI JEDAN KLASIFIKATOR. MOZDA NACI OPTIMALNE PARAMETRE UZ POMOC BAJESOVE OPTIMIZACIJE.

# PROBAJ DA IZVUCES SVM U NOVU SKRIPTU, URADI OPTIMIZACIJU PARAMETARA NA CELOJ
# MATRICI, ONDA NADJI OBELEZJA KOJA SU NAJBOLJA.



# select the columns to exclude from standard scaling     # necemo skalirati INT obelezja,
# jer su to diskretne varijable. FLOAT obelezja hocemo jer ce obuka klasifikatora biti brza
int_cols = merged.select_dtypes(include=['int'])
# create a list of columns to scale
cols_to_scale = [col for col in merged.columns if col not in int_cols]
# create a StandardScaler instance
scaler = StandardScaler()
# fit and transform the dataframe
merged[cols_to_scale] = scaler.fit_transform(merged[cols_to_scale])





# OVDE MOZES DA IZDVAJAS STA ULAZI U KLASIFIKATOR:
# merged = merged[['STAROST', 'NIHSS na prijemu', 'ASPECTS']]
# merged = merged[['NIHSS na prijemu']]
# merged = merged[['ASPECTS']]
# merged = merged[['STAROST']]
# merged = merged[['STAROST', 'NIHSS na prijemu', 'ASPECTS', 'TIP CVI','TOAST']]

print('ulazi: ')
print(merged)
print(merged.info())



# # calculate the Z-scores    # ovo je izbacivanje outlier-a, ali to necemo raditi
# z = np.abs(stats.zscore(merged))

# # set a threshold for the Z-score
# threshold = 3

# # remove the outliers
# data_without_outliers = merged[(z < threshold).all(axis=1)]

# print(f'pre outliera: {merged.shape}')
# print(f'posle izbacivanja outliera: {data_without_outliers.shape}')







y = y.values.ravel()
# print(f'y je: {y}')

# Explanation:
# .values will give the values in a numpy array (shape: (n,1))
# .ravel will convert that array shape to (n, ) (i.e. flatten it)


# training and testing, merged + y
merged = merged.values
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(merged, y, test_size = 0.2)

print(f'y train je: {y_train}')
print(f'y test je: {y_test}')
print(f'tip y train je: {type(y_train)}')
print(f'tip y test je: {type(y_test)}')     # 'numpy.ndarray' kod oba
print(f'tip x_train train je: {type(x_train)}')
print(f'tip x_test test je: {type(x_test)}')

print('DIMENZIJE:')
print(f'DIM y train je: {(y_train.shape)}')
print(f'DIM y test je: {(y_test.shape)}')     # 'numpy.ndarray' kod oba
print(f'DIM x_train train je: {(x_train.shape)}')
print(f'DIM x_test test je: {(x_test.shape)}')



clf = svm.SVC(kernel='poly', degree = 3)      # "linear", "poly", "rbf" and "sigmoid"
# clf = svm.SVC(kernel='linear')              # STA SU PARAMETRI?? TREBA NAPISATI. TAKODJE TREBA NAPISATI zasto je odabrano C=2
clf_tree = DecisionTreeClassifier()
# clf_reg = LogisticRegression()
clf_reg = LinearRegression()     # zamenila sam sa linearnim regresorom

clf.fit(x_train, y_train)
clf_tree.fit(x_train, y_train)
clf_reg.fit(x_train, y_train)


print('# *********** SVM **********************')
print('# **************************************')

y_pred_svm = clf.predict(x_test)
print(f'y_pred_svm: {y_pred_svm}')

# y_pred_tree = clf_tree.predict_proba(x_test)[:, 1]
# y_pred_reg = clf_reg.predict_proba(x_test)[:, 1]

# predict_proba is a method that can be used in certain machine learning models, such as logistic regression and random forests,
#  to predict the probability of each class for a given input. For example, in a binary classification problem with two classes
#  (e.g., "positive" and "negative"), predict_proba would return an array of probabilities, where the first column corresponds
#   to the probability of the input belonging to the first class (e.g., "positive"), and the second column corresponds to the
#   probability of the input belonging to the second class (e.g., "negative").


print('# *********** DecisionTree *************')
print('# **************************************')

y_pred_tree = clf_tree.predict(x_test)
print(f'y_pred_tree: {y_pred_tree}')

print('# *********** Regression ***************')
print('# **************************************')

y_pred_reg = clf_reg.predict(x_test)
print(f'y_pred_reg: {y_pred_reg}')

# GDE JE ROC KRIVA ZA SVM?
FP_svm, TP_svm, threshold_svm = roc_curve(y_test, y_pred_svm)
FP_tree, TP_tree, threshold_tree = roc_curve(y_test, y_pred_tree)
FP_reg, TP_reg, threshold_reg = roc_curve(y_test, y_pred_reg)

# # print('# *********** SVM **********************')
# # print('# **************************************')

print('roc_auc_score for SVM: ', roc_auc_score(y_test, y_pred_svm))

plt.subplots(1, figsize=(10, 10))
plt.title('Receiver Operating Characteristic - SVM')
plt.plot(FP_svm, TP_svm)
plt.plot([0, 1], ls="--")
plt.plot([0, 0], [1, 0], c=".7"), plt.plot([1, 1], c=".7")
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# # print('# *********** DecisionTree *************')
# # print('# **************************************')

print('roc_auc_score for DecisionTree: ', roc_auc_score(y_test, y_pred_tree))

plt.subplots(1, figsize=(10, 10))
plt.title('Receiver Operating Characteristic - DecisionTree')
plt.plot(FP_tree, TP_tree)
plt.plot([0, 1], ls="--")
plt.plot([0, 0], [1, 0], c=".7"), plt.plot([1, 1], c=".7")
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# # print('# *********** Regression ***************')
# # print('# **************************************')

print('roc_auc_score for Logistic Regression: ', roc_auc_score(y_test, y_pred_reg))

plt.subplots(1, figsize=(10, 10))
plt.title('Receiver Operating Characteristic - Logistic regression')
plt.plot(FP_reg, TP_reg)
plt.plot([0, 1], ls="--")
plt.plot([0, 0], [1, 0], c=".7"), plt.plot([1, 1], c=".7")
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()