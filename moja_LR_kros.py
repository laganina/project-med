from sklearn.linear_model import LinearRegression, LogisticRegression
from podaci_prvi import obelezja, labela
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import pandas as pd 


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

# clf = LinearRegression()
clf = LogisticRegression()

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
print(f'*************************************************')



## ****************** ADD CONFUSION MATRIX ******************** ##
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import ConfusionMatrixDisplay 
from sklearn.metrics import roc_curve
from sklearn.metrics import RocCurveDisplay




# MATRICA KONFUZIJE
X = obelezja
y_pred = cross_val_predict(clf, X, y, cv=5)
print(f'y shape: {y.shape}')
print(f'y_pred shape: {y_pred.shape}')

# Calculate the confusion matrix
conf_mat = confusion_matrix(y, y_pred)
print(conf_mat)

# Plot the confusion matrix
disp = ConfusionMatrixDisplay(conf_mat).plot()
disp.ax_.set_xlabel('Предвиђена класа')
disp.ax_.set_ylabel('Истинита класа')


# Show the plot
plt.title('Матрица конфузије, препознавање РНП-а, логистичка регресија')
plt.savefig('Confusion matrix Log Regression.png')            # radi i jedno i drugo - ovo zauzima manje memorije 
plt.show()


fpr, tpr, _ = roc_curve(y, y_pred, pos_label=1)          # 1 NIJE DOSLO DO POBOLJSANJA 
roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr).plot()
plt.title('"ROC" kriva, prepoznavanje RNP-a, log. regresija') 
plt.savefig('ROC curve log reg.png')                          # radi i jedno i drugo 
# roc_display.figure_.savefig('roc_log_reg.png',dpi=300)      # radi 
plt.show()   


# cuvanje rezultata 

raw_data = {'accuracy': [acc],
                'precision': [prec],
                'recall': [recall],
                'f1': [f1],
                'auc': [auc]}

df = pd.DataFrame(raw_data, columns = ['accuracy', 'precision', 'recall', 'f1', 'auc'])
df = df.round(decimals=2)
print(df)
df.to_csv('log_reg_rnp.csv', index=False)








