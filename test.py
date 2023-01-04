import tensorflow
import keras
import pandas as pd
from tabulate import tabulate
import numpy as np
import sklearn
from sklearn import svm 
from sklearn import metrics 
from sklearn.model_selection import train_test_split

data = pd.read_excel(r'C:\Users\Laganina\OneDrive - Univerzitet u Novom Sadu\Desktop\machine_learning\project-med\podaci.xlsx')

#osnovna obelezja: starost, nihss na prijemu, aspect score

columns = list(data.columns)

#data frame sa osnovnim obelezjima
basic = data[['STAROST', 'NIHSS na prijemu', 'ASPECTS', 'NIHSS 24h']]

#data frame [basic] sa izbacenim vrstama koje sadrze NaN value
df = pd.DataFrame(basic)

df = df.apply (pd.to_numeric, errors='coerce')
df = df.dropna()

df = df.dropna().reset_index(drop=True)

#labela za ovu klasifikaciju je pad nihss score-a za 40% nakon 24h u odnosu na inicijalnu vrednost
#napraviti varijablu nihss na prijemu, pa nihss posle 24h, labela = (nihss24h - nihssprijem)/nihssprijem,
#  ako je ta razlika kroz pocetna vrednost nihss veci ili jednak od 0.4 onda doslo je do
#poboljsanja, ako je manji od 0.4 onda je doslo do pogorsanja


#izdvojeni nihss parametri
nihssprijem = df['NIHSS na prijemu']
nihss24 = df['NIHSS 24h']

#napravljena lista koja ce predstavljati labelu
label = []
for i in range(len(nihssprijem)):
    label.append((nihss24[i]-nihssprijem[i])/nihssprijem[i])
    
#poboljsanje = 0, pogorsanje = 1
y = []
for i in range(len(label)):
    if label[i] <= 0.4:
        y.append(0)
    else:
        y.append(1)
    i+=1

#dropping nihss 24h
df = df.drop(labels='NIHSS 24h', axis = 1)

#list to data frame
df1 = pd.DataFrame(y, columns=['STANJE'])

#stvaranje matrice sa starost, nihss na prijemu, aspects, y iliti labela
merged = pd.concat([df,df1], axis=1)

#x je df, y je y
#training and testing 

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(df,y,test_size = 0.2)

classes = ['better', 'worse']

clf = svm.SVC(kernel='linear', C=2)
clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)

acc = metrics.accuracy_score(y_test, y_pred)
print(acc)