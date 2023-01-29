import tensorflow
import keras
import pandas as pd
import numpy as np
from podaci_prvi import joined, df_dodatni

# svi lekovi izbaceni iz df_dodatni

# !!!!!!!!! sredi !!!!!!!!!
# RESENJE JE DA IH UKLOPIMO SVE U JEDNO OBELEZJE CIJA CE VREDNOST DA
# BUDE DECIMALNI EKVIVALENT BINARNOG BROJA SASTAVLJENOG OD OVIH 5 OBELEZJA

df_lek = df_dodatni[['ASA', 'Clopidogrel','OAKT', 'Statini', 'AntiHTA']].copy()
df_lek = pd.DataFrame(df_lek)

print('df_lek, tek ucitano:')
print(df_lek)

# vrednosti od 0 do 5 za da (koji lek) ne
df_lek['ASA'] = df_lek['ASA'].replace('Da', 1)
df_lek['Clopidogrel'] = df_lek['Clopidogrel'].replace('Da', 1)
df_lek['OAKT'] = df_lek['OAKT'].replace('Da', 1)
df_lek['Statini'] = df_lek['Statini'].replace('Da', 1)
df_lek['AntiHTA'] = df_lek['AntiHTA'].replace('Da', 1)

df_lek['ASA'] = df_lek['ASA'].replace('da', 1)
df_lek['Clopidogrel'] = df_lek['Clopidogrel'].replace('da', 1)
df_lek['OAKT'] = df_lek['OAKT'].replace('da', 1)
df_lek['Statini'] = df_lek['Statini'].replace('da', 1)
df_lek['AntiHTA'] = df_lek['AntiHTA'].replace('da', 1)

df_lek['ASA'] = df_lek['ASA'].replace('Ne', 0)
df_lek['Clopidogrel'] = df_lek['Clopidogrel'].replace('Ne', 0)
df_lek['OAKT'] = df_lek['OAKT'].replace('Ne', 0)
df_lek['Statini'] = df_lek['Statini'].replace('Ne', 0)
df_lek['AntiHTA'] = df_lek['AntiHTA'].replace('Ne', 0)

df_lek['ASA'] = df_lek['ASA'].replace('ne', 0)
df_lek['Clopidogrel'] = df_lek['Clopidogrel'].replace('ne', 0)
df_lek['OAKT'] = df_lek['OAKT'].replace('ne', 0)
df_lek['Statini'] = df_lek['Statini'].replace('ne', 0)
df_lek['AntiHTA'] = df_lek['AntiHTA'].replace('ne', 0)

print('df_lek, zamena brojevima:')
print(df_lek)

print('*****************************************************************************')
print('*****************************************************************************')



