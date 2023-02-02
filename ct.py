import numpy as np
import pandas as pd
#----------------------------------------------------------------------------------------------------------------------#
data = pd.read_excel(r'C:\Users\Laganina\OneDrive - Univerzitet u Novom Sadu\Desktop\machine_learning\project-med\project-med\podaci.xlsx')

df_ct = data[['CT hiperdenzni znak']]
df_ct = pd.DataFrame(df_ct)

#print('df_ct, tek ucitano:')
#print(df_ct)
# zamena vrednosti brojevima
# CT HIPERDENZNI ZNAK

# da ne bilo koja pretvoreno u 1 0 2
df_ct['CT hiperdenzni znak'] = df_ct['CT hiperdenzni znak'].replace('Bilo koja', 2)
df_ct['CT hiperdenzni znak'] = df_ct['CT hiperdenzni znak'].replace('bilo koja', 2)

df_ct['CT hiperdenzni znak'] = df_ct['CT hiperdenzni znak'].replace('Da', 1)
df_ct['CT hiperdenzni znak'] = df_ct['CT hiperdenzni znak'].replace('da', 1)

df_ct['CT hiperdenzni znak'] = df_ct['CT hiperdenzni znak'].replace('Ne', 0)
df_ct['CT hiperdenzni znak'] = df_ct['CT hiperdenzni znak'].replace('ne', 0)

#print('df_ct, CT, zamena brojevima:')
#print(df_ct[['CT hiperdenzni znak']])
#print('*****************************************************************************')
#print('*****************************************************************************')


