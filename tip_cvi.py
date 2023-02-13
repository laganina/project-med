import numpy as np
import pandas as pd

# data = pd.read_excel(r'C:\Users\Laganina\OneDrive - Univerzitet u Novom Sadu\Desktop\machine_learning\project-med\project-med\podaci.xlsx')
data = pd.read_excel(r'C:\Users\Olivera\Documents\PythonScripts\SlogOporavakProjekat2022-23\project-med\podaci.xlsx')

df_tipCVI = data[['TIP CVI']]
df_tipCVI = pd.DataFrame(df_tipCVI)

#print('df_tipCVI, tek ucitano:')
#print(df_tipCVI)
#----------------------------------------------------------------------------------------------------------------------#

#----------------------------------------------------------------------------------------------------------------------#
# zamena vrednosti brojevima
# TIP CVI

# tip cvi - pretvoreni u 0 1 2 3
df_tipCVI['TIP CVI'] = df_tipCVI['TIP CVI'].replace('TACI', 0)
df_tipCVI['TIP CVI'] = df_tipCVI['TIP CVI'].replace('TACI    ', 0)
df_tipCVI['TIP CVI'] = df_tipCVI['TIP CVI'].replace('TACI  ', 0)
df_tipCVI['TIP CVI'] = df_tipCVI['TIP CVI'].replace('TACI   ', 0)

df_tipCVI['TIP CVI'] = df_tipCVI['TIP CVI'].replace('PACI', 1)
df_tipCVI['TIP CVI'] = df_tipCVI['TIP CVI'].replace('PACI  ', 1)
df_tipCVI['TIP CVI'] = df_tipCVI['TIP CVI'].replace('PACI ', 1)

df_tipCVI['TIP CVI'] = df_tipCVI['TIP CVI'].replace('LACI', 2)
df_tipCVI['TIP CVI'] = df_tipCVI['TIP CVI'].replace('LAC', 2)
df_tipCVI['TIP CVI'] = df_tipCVI['TIP CVI'].replace('LACI ', 2)
df_tipCVI['TIP CVI'] = df_tipCVI['TIP CVI'].replace('LACI  ', 2)
df_tipCVI['TIP CVI'] = df_tipCVI['TIP CVI'].replace('LACI?', 2)

df_tipCVI['TIP CVI'] = df_tipCVI['TIP CVI'].replace('POCI', 3)

#print('df_tipCVI, tip cvi, zamena brojevima:')
#print(df_tipCVI[['TIP CVI']])
#print('*****************************************************************************')
#print('*****************************************************************************')

