from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from podaci_prvi import obelezja, labela
import numpy as np

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

clf = DecisionTreeClassifier()

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
