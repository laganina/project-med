import numpy as np
import pandas as pd

# data = pd.read_excel(r'C:\Users\Laganina\OneDrive - Univerzitet u Novom Sadu\Desktop\machine_learning\project-med\project-med\podaci.xlsx')
data = pd.read_excel(r'C:\Users\Olivera\Documents\PythonScripts\SlogOporavakProjekat2022-23\project-med\podaci.xlsx')


df_tipHLP = data[['Tip HLP']]
df_tipHLP = pd.DataFrame(df_tipHLP)

#print('df_tipHLP, tek ucitano:')
#print(df_tipHLP)
#------------------------------------------------------------------------------------#

#------------------------------------------------------------------------------------#
# vrednosti od 0 do 3 za tip HLP
df_tipHLP['Tip HLP'] = df_tipHLP['Tip HLP'].replace('Bez poremećaja', 0)
df_tipHLP['Tip HLP'] = df_tipHLP['Tip HLP'].replace('Bez poremećja', 0)
df_tipHLP['Tip HLP'] = df_tipHLP['Tip HLP'].replace('Bez poremćaja', 0)
df_tipHLP['Tip HLP'] = df_tipHLP['Tip HLP'].replace('bez poremećaja', 0)
df_tipHLP['Tip HLP'] = df_tipHLP['Tip HLP'].replace('Ne', 0)

df_tipHLP['Tip HLP'] = df_tipHLP['Tip HLP'].replace('IIa', 1)
df_tipHLP['Tip HLP'] = df_tipHLP['Tip HLP'].replace('IIb', 2)
df_tipHLP['Tip HLP'] = df_tipHLP['Tip HLP'].replace('IV', 3)
df_tipHLP['Tip HLP'] = df_tipHLP['Tip HLP'].replace('HLP IV', 3)
df_tipHLP['Tip HLP'] = df_tipHLP['Tip HLP'].replace('II', np.nan)
df_tipHLP['Tip HLP'] = df_tipHLP['Tip HLP'].replace('nr', np.nan)
df_tipHLP['Tip HLP'] = df_tipHLP['Tip HLP'].replace('n/a', np.nan)
df_tipHLP['Tip HLP'] = df_tipHLP['Tip HLP'].replace('Hipo HDL holesterolemija', np.nan)

#------------------------------------------------------------------------------------#
#print('df_tipHLP, zamena brojevima:')
#print(df_tipHLP)
#print('*****************************************************************************')
#print('*****************************************************************************')



