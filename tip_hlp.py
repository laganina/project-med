from podaci_prvi import df_tipHLP
from lekovi import df_dodatni
import numpy as np

# zamena vrednosti brojevima
# TIP HLP
df_tipHLP = df_dodatni[['Tip HLP']].copy()


print('df_tipHLP, tek ucitano:')
print(df_tipHLP)
# ?????????? sta raditi sa hipo hdlp holesterolemijom ??????????

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
df_tipHLP['Tip HLP'] = df_tipHLP['Tip HLP'].replace('nr', np.nan)
df_tipHLP['Tip HLP'] = df_tipHLP['Tip HLP'].replace('n/a', np.nan)
df_tipHLP['Tip HLP'] = df_tipHLP['Tip HLP'].replace('Hipo HDL holesterolemija', np.nan)

print('df_tipHLP, zamena brojevima:')
print(df_tipHLP)
print('*****************************************************************************')
print('*****************************************************************************')



