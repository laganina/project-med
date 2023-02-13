import pandas as pd
import numpy as np
from pol import df_pol
from tip_cvi import df_tipCVI
from toast import df_toast
#----------------------------------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------------------#
# zameni putanju!!!!
# data = pd.read_excel(r'C:\Users\Laganina\OneDrive - Univerzitet u Novom Sadu\Desktop\machine_learning\project-med\project-med\podaci.xlsx')    # ZAMENI PUTANJU
data = pd.read_excel(r'C:\Users\Olivera\Documents\PythonScripts\SlogOporavakProjekat2022-23\project-med\podaci.xlsx')


#komplikacije (infekcija, duboke tromboze, nema=0)
#----------------------------------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------------------#


# importovanje osnovnih obelezja
basic = data[['STAROST', 'NIHSS 24h', 'Glikemija', 'Broj dana hospitalizacije', 'RANKIN 90 dana']]
df_basic = pd.DataFrame(basic)

dodatni = data[['NIHSS na prijemu', 'NIHSS OT', 'RANKIN OT', 'Leukoarajoza', 'Pneumonija']]
df_dodatni = pd.DataFrame(dodatni)

# leukoarajoza u 1 0
df_dodatni['Leukoarajoza'] = df_dodatni['Leukoarajoza'].map({'Da': 1, 'Ne': 0, 1:1,0:0})
df_dodatni['Pneumonija'] = df_dodatni['Pneumonija'].map({'Da': 1, 'Ne': 0, 1:1,0:0})



# spajadnje u jedan df
df = pd.concat([df_basic,df_dodatni], axis=1)
df = df.join(df_pol, lsuffix='_caller', rsuffix='_other')
df = df.join(df_tipCVI, lsuffix='_caller', rsuffix='_other')
df = df.join(df_toast, lsuffix='_caller', rsuffix='_other')

# nan values
df['RANKIN 90 dana'] = df['RANKIN 90 dana'].replace('lost', np.nan)
df['RANKIN 90 dana'] = df['RANKIN 90 dana'].replace('NR', np.nan)
df['RANKIN 90 dana'] = df['RANKIN 90 dana'].replace('ne javlja se', np.nan)
df['RANKIN 90 dana'] = df['RANKIN 90 dana'].replace('nr - lost', np.nan)
df['RANKIN 90 dana'] = df['RANKIN 90 dana'].replace('NR - živi u Nemačkoj', np.nan)
df['RANKIN 90 dana'] = df['RANKIN 90 dana'].replace('lost', np.nan)
df['RANKIN 90 dana'] = df['RANKIN 90 dana'].replace('nedostupan', np.nan)
df['RANKIN 90 dana'] = df['RANKIN 90 dana'].replace('nedostupan ', np.nan)
df['RANKIN 90 dana'] = df['RANKIN 90 dana'].replace('nedostupan, proveri broj', np.nan)
df['RANKIN 90 dana'] = df['RANKIN 90 dana'].replace('nedostupan, proveri broj ', np.nan)


df['NIHSS OT'] = df['NIHSS OT'].replace('NR', np.nan)
df['NIHSS OT'] = df['NIHSS OT'].replace('na', np.nan)
df['NIHSS OT'] = df['NIHSS OT'].replace('N/a', np.nan)
df['NIHSS OT'] = df['NIHSS OT'].replace('nr', np.nan)


# brisu se vrste sa nan values
df = df.dropna()
df = df.reset_index(drop=True)

# integer
df = df.astype({'RANKIN 90 dana':'int'})
df = df.astype({'NIHSS 24h':'int'})
df = df.astype({'TOAST':'int'})
df = df.astype({'NIHSS OT':'int'})
df = df.astype({'RANKIN OT':'int'})
df = df.astype({'TIP CVI':'int'})
df = df.astype({'Leukoarajoza':'int'})

obelezja = df
# novi df, y labela
y = df[['RANKIN 90 dana']].copy()
labela = pd.DataFrame(y)
obelezja = obelezja.drop(labels='RANKIN 90 dana',axis=1)


print(obelezja.shape)
print(labela.shape)
print(obelezja.info())
# sada ide podela na manji broj klasa i krosvalidacija