import tensorflow
import keras
import pandas as pd
import numpy as np
import tip_cvi
import toast

#----------------------------------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------------------#
# zameni putanju!!!!
data = pd.read_excel(r'C:\Users\Laganina\OneDrive - Univerzitet u Novom Sadu\Desktop\machine_learning\project-med\project-med\podaci.xlsx')    # ZAMENI PUTANJU
#komplikacije (pneumonija, infekcija, duboke tromboze, nema=0)
#----------------------------------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------------------#


# importovanje osnovnih obelezja
basic = data[['STAROST', 'NIHSS 24h', 'Glikemija', 'Broj dana hospitalizacije', 'RANKIN 90 dana']]
df_basic = pd.DataFrame(basic)

dodatni = data[['NIHSS na prijemu', 'NIHSS OT', 'TIP CVI', 'TOAST', 'RANKIN OT', 'Leukoarajoza']]
df_dodatni = pd.DataFrame(dodatni)

#----------------------------------------------------------------------------------------------------------------------#
# leukoarajoza u 1 0
df_dodatni['Leukoarajoza'] = df_dodatni['Leukoarajoza'].replace('Da', np.nan)
df_dodatni['Leukoarajoza'] = df_dodatni['Leukoarajoza'].replace('da', np.nan)
df_dodatni['Leukoarajoza'] = df_dodatni['Leukoarajoza'].replace('Ne', np.nan)
df_dodatni['Leukoarajoza'] = df_dodatni['Leukoarajoza'].replace('ne', np.nan)

#----------------------------------------------------------------------------------------------------------------------#
# spajadnje u jedan df
df = pd.concat([df_basic,df_dodatni], axis=1)

#----------------------------------------------------------------------------------------------------------------------#
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
df['TOAST'] = df['TOAST'].replace('na', np.nan)

#----------------------------------------------------------------------------------------------------------------------#
# float: STAROST, NIHSS, NIHSS24, TT MASA, GLIKEMIJA, MAP, OTT, DNT


# konvertovanje u integer
df['STAROST'] = pd.to_numeric(df['STAROST'])
# integer
df = df.astype({'RANKIN 90 dana':'int'})
df = df.astype({'TOAST':'int'})
df = df.astype({'RANKIN OT':'int'})
df = df.astype({'TIP CVI':'int'})
df = df.astype({'Leukoarajoza':'int'})

#----------------------------------------------------------------------------------------------------------------------#
# prazna polja u NaN
df = df.fillna(np.nan)
# brisu se vrste sa nan values
df = df.dropna()

#resetovanje indeksa zbog izbacivanja nanova
df = df.reset_index(drop=True)
print('index kolona:')
print(df.index)
df.index.name = 'index'
print(f'index name: {df.index.name}')

print(df.info())
print(df)
print(df.shape)

#----------------------------------------------------------------------------------------------------------------------#
# novi df, y labela
df_rankin = df[['RANKIN 90 dana']]
df_rankin = pd.DataFrame(df_rankin)
# labela - ranking 90 dana
df = df.drop(labels='RANKIN 90 dana',axis=1)
print(df_rankin)

print(df.shape)
print(df_rankin.shape)

#----------------------------------------------------------------------------------------------------------------------#
# provera broja uzoraka u svakoj klasi
klase = np.unique(df_rankin.values)
print('klase: {klase}')

for i in range(len(klase)):
    broj_uzoraka = sum(df_rankin.values == klase[i])
    print(f'Broj uzoraka u {i}-toj klasi je: {broj_uzoraka}')

# IMAMO ukupno 6 KLASA
# Broj uzoraka u 0-toj klasi je: [71]
# Broj uzoraka u 1-toj klasi je: [87]
# Broj uzoraka u 2-toj klasi je: [47]
# Broj uzoraka u 3-toj klasi je: [45]
# Broj uzoraka u 4-toj klasi je: [38]
# Broj uzoraka u 5-toj klasi je: [19]
# Broj uzoraka u 6-toj klasi je: [31]

# sada ide podela na manji broj klasa i krosvalidacija