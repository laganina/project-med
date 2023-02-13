from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from podaci_prvi import obelezja, labela
import numpy as np
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

clf = DecisionTreeClassifier()

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



from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import ConfusionMatrixDisplay 


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

# Show the plot
plt.title('Confusion matrix, Decision Tree')
plt.savefig('Confusion matrix Dec Tree.png')            # radi i jedno i drugo - ovo zauzima manje memorije 
plt.show()








raw_data = {'accuracy': [acc],
                'precision': [prec],
                'recall': [recall],
                'f1': [f1],
                'auc': [auc]}

df = pd.DataFrame(raw_data, columns = ['accuracy', 'precision', 'recall', 'f1', 'auc'])
df = df.round(decimals=2)
print(df)
df.to_csv('dec_tree_rnp.csv', index=False)


# The predicted class probability for a Decision Tree classifier is calculated based on the relative
#  frequency of each class in the leaves of the tree that a sample reaches. Each internal node in a Decision 
#  Tree represents a split in the feature space, and each leaf node represents a region of the feature space
#   where the samples have similar class labels. The predicted class probability for a sample is based on the 
#   relative frequency of the classes in the leaf node that the sample reaches.

# For example, if a Decision Tree has three classes, A, B, and C, and a sample reaches a leaf node with 10 samples 
# of class A, 5 samples of class B, and 2 samples of class C, the predicted class probabilities for the sample would 
# be [0.7, 0.3, 0.1].

# In summary, the predicted class probability for a Decision Tree classifier is based on the relative frequency of 
# each class in the leaf node that a sample reaches. The class with the highest predicted probability is selected as
#  the final prediction.


