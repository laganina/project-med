from podaci_prvi import df_dodatni
import numpy as np
import pandas as pd

# zamena vrednosti brojevima
# KOMORBIDITETI
# HTA->Alkohol

# napravljen data frame za komorbiditete
df_kom = df_dodatni[['HTA','DM','Pušenje','HLP','AA','CMP','Alkohol']].copy()
df_kom = pd.DataFrame(df_kom)

print('df_kom, tek ucitano:')
print(df_kom)

# vrednosti od 0 do 7 za da (komorbiditete) ne, bez tip HLP
df_kom['HTA'] = df_kom['HTA'].replace('Da', 3)
df_kom['HTA'] = df_kom['HTA'].replace('da', 3)

df_kom['DM'] = df_kom['DM'].replace('Da', 1)
df_kom['DM'] = df_kom['DM'].replace('da', 1)

df_kom['Pušenje'] = df_kom['Pušenje'].replace('Da', 2)
df_kom['Pušenje'] = df_kom['Pušenje'].replace('da', 2)

df_kom['HLP'] = df_kom['HLP'].replace('Da', 2)
df_kom['HLP'] = df_kom['HLP'].replace('da', 2)

df_kom['AA'] = df_kom['AA'].replace('Da', 3)
df_kom['AA'] = df_kom['AA'].replace('da', 3)

df_kom['CMP'] = df_kom['CMP'].replace('Da', 2)
df_kom['CMP'] = df_kom['CMP'].replace('Da ', 2)
df_kom['CMP'] = df_kom['CMP'].replace('Da (CMP hypertrophica comp)', 2)
df_kom['CMP'] = df_kom['CMP'].replace('Da (CMP ischaemica9', 2)
df_kom['CMP'] = df_kom['CMP'].replace('Da (CMP ischaemica)', 2)
df_kom['CMP'] = df_kom['CMP'].replace('Da (CMP valvulars)', 2)
df_kom['CMP'] = df_kom['CMP'].replace('Da (CMP valvularis chr. Com EF 45%)', 2)
df_kom['CMP'] = df_kom['CMP'].replace('Da (CMP valvularis chr. Com)', 2)
df_kom['CMP'] = df_kom['CMP'].replace('Da (CMP dilatativa, EF 23%)', 2)
df_kom['CMP'] = df_kom['CMP'].replace('Da (CMP hypertensiva)', 2)
df_kom['CMP'] = df_kom['CMP'].replace('Da (CMP isch)', 2)
df_kom['CMP'] = df_kom['CMP'].replace('Da (CMP hypertrophica comp)', 2)
df_kom['CMP'] = df_kom['CMP'].replace('Da (CMP dilatativa EF 35%)', 2)
df_kom['CMP'] = df_kom['CMP'].replace('Da (CMP dilatativa)', 2)
df_kom['CMP'] = df_kom['CMP'].replace('da', 2)
df_kom['CMP'] = df_kom['CMP'].replace('da (CMP hypertensiva hypertrophica comp)', 2)
df_kom['CMP'] = df_kom['CMP'].replace('da (CMP ischaemica)', 2)
df_kom['CMP'] = df_kom['CMP'].replace('da, CMP valvularis', 2)

#--------------------------------------------------------------#
#da
df_kom['Alkohol'] = df_kom['Alkohol'].replace('Da', 1)
df_kom['Alkohol'] = df_kom['Alkohol'].replace('Da', 1)
#--------------------------------------------------------------#

# ne
df_kom['HTA'] = df_kom['HTA'].replace('Ne', 0)
df_kom['HTA'] = df_kom['HTA'].replace('ne', 0)

df_kom['DM'] = df_kom['DM'].replace('Ne', 0)
df_kom['DM'] = df_kom['DM'].replace('ne', 0)

df_kom['Pušenje'] = df_kom['Pušenje'].replace('Ne', 0)
df_kom['Pušenje'] = df_kom['Pušenje'].replace('ne', 0)

df_kom['HLP'] = df_kom['HLP'].replace('Ne', 0)
df_kom['HLP'] = df_kom['HLP'].replace('ne', 0)
df_kom['HLP'] = df_kom['HLP'].replace('nr', np.nan)    # NAN

df_kom['AA'] = df_kom['AA'].replace('Ne', 0)
df_kom['AA'] = df_kom['AA'].replace('ne', 0)

df_kom['CMP'] = df_kom['CMP'].replace('Ne', 0)
df_kom['CMP'] = df_kom['CMP'].replace('ne', 0)

df_kom['Alkohol'] = df_kom['Alkohol'].replace('Ne', 0)
df_kom['Alkohol'] = df_kom['Alkohol'].replace('ne', 0)
#--------------------------------------------------------------#

print('df_kom, zamena brojevima:')
print(df_kom)
print('*****************************************************************************')
print('*****************************************************************************')
#----------------------------------------------------------------------------------------------------------------------#


