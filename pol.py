import numpy as np
import pandas as pd



#----------------------------------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------------------#

# data = pd.read_excel(r'C:\Users\Laganina\OneDrive - Univerzitet u Novom Sadu\Desktop\machine_learning\project-med\project-med\podaci.xlsx')
data = pd.read_excel(r'C:\Users\Olivera\Documents\PythonScripts\SlogOporavakProjekat2022-23\project-med\podaci.xlsx')


pol = data[['POL']]
df_pol = pd.DataFrame(pol)

#print('df_pol, tek ucitano:')
#print(df_pol)

#----------------------------------------------------------------------------------------------------------------------#
# ženski pol - 1, muški pol - 0
df_pol['POL'] = df_pol['POL'].replace('Z', 1)
df_pol['POL'] = df_pol['POL'].replace('z', 1)
df_pol['POL'] = df_pol['POL'].replace('Ž', 1)
df_pol['POL'] = df_pol['POL'].replace('ž', 1)

df_pol['POL'] = df_pol['POL'].replace('M', 0)
df_pol['POL'] = df_pol['POL'].replace('m', 0)

#print('df_pol pretvoreno u 1 i 0: ')
#print(df_pol)