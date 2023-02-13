import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn import metrics
# from project.podaci_drugi import df, df_rankin
from podaci_drugi import obelezja, labela
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score

# DODAJEM

y = labela

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

# ovo je za multilabel klasifikaciju


# create an instance of the model
# clf = OneVsRestClassifier(LogisticRegression)
clf = LogisticRegression(multi_class='ovr', solver='lbfgs')
predictions = cross_val_predict(clf, obelezja, y, cv=5)

scorer = cross_val_score(clf, obelezja, y, cv=5, scoring='accuracy')
print(f'tacnost: {scorer}')
acc = np.mean(scorer)
print(f'mean acc: {np.mean(scorer)}')
print(f'*************************************************')

scorer = cross_val_score(clf, obelezja, y, cv=5, scoring='precision_macro')
print(f'preciznost: {scorer}')
prec = np.mean(scorer)
print(f'mean prec: {np.mean(scorer)}')
print(f'*************************************************')

scorer = cross_val_score(clf, obelezja, y, cv=5, scoring='recall_macro')
print(f'recall: {scorer}')
recall = np.mean(scorer)
print(f'mean recall: {np.mean(scorer)}')
print(f'*************************************************')

scorer = cross_val_score(clf, obelezja, y, cv=5, scoring='f1_macro')
print(f'f1: {scorer}')
f1 = np.mean(scorer)
print(f'mean f1: {np.mean(scorer)}')
print(f'*************************************************')

scorer = cross_val_score(clf, obelezja, y, cv=5, scoring='roc_auc_ovr')
print(f'roc_auc: {scorer}')
auc = np.mean(scorer)
print(f'mean auc: {np.mean(scorer)}')
print(f'*************************************************')


y_pred = np.argmax(predictions, axis=0)


## ****************** ADD CONFUSION MATRIX ******************** ##
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import ConfusionMatrixDisplay 
from sklearn.metrics import roc_curve
from sklearn.metrics import RocCurveDisplay


# Calculate the confusion matrix
conf_mat = confusion_matrix(y, predictions)
print(conf_mat)


# Plot the confusion matrix
disp = ConfusionMatrixDisplay(conf_mat).plot()
disp.ax_.set_xlabel('Предвиђена класа')
disp.ax_.set_ylabel('Истинита класа')


# Show the plot
plt.title('Матрица конфузије, препознавање ПТМИ-а, логистичка регресија')
plt.savefig('Confusion matrix лог регресија мултиклас.png')            # radi i jedno i drugo - ovo zauzima manje memorije 
# disp.figure_.savefig('conf_mat_SVM.png',dpi=300)   # radi 
plt.show()

# ДОДАЈ ЧУВАЊЕ СВИХ МЕРА ПО СВАКОЈ КЛАСИ И УКУПНИХ МЕРА КОЈЕ СУ ВЕЋ ИЗРАЧУНАТЕ 


# ДОДАЈ ЧУВАЊЕ СВИХ МЕРА ПО СВАКОЈ КЛАСИ И УКУПНИХ МЕРА КОЈЕ СУ ВЕЋ ИЗРАЧУНАТЕ 
raw_data = {'Acc': [acc],
                'Precision': [prec],
                'Recall': [recall],
                'f1': [f1],
                'auc': [auc]}

df = pd.DataFrame(raw_data, columns = ['Acc', 'Precision', 'Recall', 'f1', 'auc'])
df = df.round(decimals=2)
print(df)
df.to_csv('RezultatiMultilabelLogRegUKupno.csv', index=False)


# САДА ЗА СВАКУ КЛАСУ ПОСЕБНО

import numpy as np
import pandas as pd

# Load the confusion matrix
cm = conf_mat

# Calculate the number of samples for each class
num_samples = np.sum(cm, axis=1)
print(f'num_samples: {num_samples}')

# Calculate the true positive count for each class
true_positives = np.diag(cm)
print(f'true positives: {true_positives}')

# Calculate the false positive count for each class
false_positives = np.sum(cm, axis=0) - true_positives

# Calculate the false negative count for each class
false_negatives = np.sum(cm, axis=1) - true_positives



print(f'true_positives + false_negatives: {true_positives + false_negatives}')

precision = true_positives / (true_positives + false_positives)
recall = true_positives / (true_positives + false_negatives)
f1_score = 2 * (precision * recall) / (precision + recall)

# tacnost nema poente racunati jer ce biti ista za sva tri prolaza 
# uvek se racuna kao ukupan broj tacnost klasifikovanih sa uk brojem elemenata 

# Create a data frame to store the results
results = pd.DataFrame({
    'True positives': true_positives,
    'False positives': false_positives,
    'False negatives': false_negatives,
    'Precision': precision,
    'Recall': recall,
    'F1': f1_score
})

# Save the results to a CSV file
results = results.round(decimals=2)
print(results)
results.to_csv('RezultatiMultilabelLogReg.csv', index=False)



