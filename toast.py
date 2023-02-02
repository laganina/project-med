import numpy as np
import pandas as pd

data = pd.read_excel(r'C:\Users\Laganina\OneDrive - Univerzitet u Novom Sadu\Desktop\machine_learning\project-med\project-med\podaci.xlsx')

df_toast = data[['TOAST']]
df_toast = pd.DataFrame(df_toast)

#print('df_toast, tek ucitano:')
#print(df_toast)
#------------------------------------------------------------------------------------#

#------------------------------------------------------------------------------------#
# zamena vrednosti brojevima
# TOAST

# tip tost - pretvoreni u 0 1 2 3 4 5
df_toast['TOAST'] = df_toast['TOAST'].replace('LAA', 0)

df_toast['TOAST'] = df_toast['TOAST'].replace('CE', 1)
df_toast['TOAST'] = df_toast['TOAST'].replace('CE?', 1)

df_toast['TOAST'] = df_toast['TOAST'].replace('SVD', 2)

df_toast['TOAST'] = df_toast['TOAST'].replace('Drugi', 3)

df_toast['TOAST'] = df_toast['TOAST'].replace('Neutvrđeno', 4)
df_toast['TOAST'] = df_toast['TOAST'].replace('Neutrvđeno', 4)
df_toast['TOAST'] = df_toast['TOAST'].replace('Neutvrđen', 4)
df_toast['TOAST'] = df_toast['TOAST'].replace('Neutrvđen', 4)

df_toast['TOAST'] = df_toast['TOAST'].replace('Stroke mimic', 5)

df_toast['TOAST'] = df_toast['TOAST'].replace('na', np.nan)

#print('df_toast, TOAST, zamena brojevima:')
#print(df_toast[['TOAST']])
#print('*****************************************************************************')
#print('*****************************************************************************')

