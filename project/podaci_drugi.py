import tensorflow
import keras
import pandas as pd
import numpy as np

# data = pd.read_excel(
#     r'C:\Users\Laganina\OneDrive - Univerzitet u Novom Sadu\Desktop\machine_learning\project-med\project\podaci.xlsx')
data = pd.read_excel('podaci.xlsx') 


#komplikacije (pneumonija, infekcija, duboke tromboze, nema=0)

# importovanje osnovnih obelezja
basic = data[['STAROST', 'NIHSS 24h', 'Glikemija', 'Broj dana hospitalizacije', 'RANKIN 90 dana']]
df_basic = pd.DataFrame(basic)

dodatni = data[['NIHSS na prijemu', 'NIHSS OT', 'TIP CVI', 'TOAST', 'RANKIN OT', 'Leukoarajoza']]
df_dodatni = pd.DataFrame(dodatni)

# leukoarajoza u 1 0
df_dodatni['Leukoarajoza'] = df_dodatni['Leukoarajoza'].map({'Da': 1, 'Ne': 0, 1:1,0:0})

# tip cvi - pretvoreni u 0 1 2 3
df_dodatni['TIP CVI'] = df_dodatni['TIP CVI'].replace('TACI', 0)
df_dodatni['TIP CVI'] = df_dodatni['TIP CVI'].replace('TACI    ', 0)
df_dodatni['TIP CVI'] = df_dodatni['TIP CVI'].replace('TACI  ', 0)
df_dodatni['TIP CVI'] = df_dodatni['TIP CVI'].replace('TACI   ', 0)
df_dodatni['TIP CVI'] = df_dodatni['TIP CVI'].replace('PACI', 1)
df_dodatni['TIP CVI'] = df_dodatni['TIP CVI'].replace('PACI  ', 1)
df_dodatni['TIP CVI'] = df_dodatni['TIP CVI'].replace('PACI ', 1)
df_dodatni['TIP CVI'] = df_dodatni['TIP CVI'].replace('LACI', 2)
df_dodatni['TIP CVI'] = df_dodatni['TIP CVI'].replace('LAC', 2)
df_dodatni['TIP CVI'] = df_dodatni['TIP CVI'].replace('LACI ', 2)
df_dodatni['TIP CVI'] = df_dodatni['TIP CVI'].replace('LACI  ', 2)
df_dodatni['TIP CVI'] = df_dodatni['TIP CVI'].replace('LACI?', 2)
df_dodatni['TIP CVI'] = df_dodatni['TIP CVI'].replace('POCI', 3)

# tip tost - pretvoreni u 0 1 2 3 4 5
df_dodatni['TOAST'] = df_dodatni['TOAST'].replace('LAA', 0)
df_dodatni['TOAST'] = df_dodatni['TOAST'].replace('CE', 1)
df_dodatni['TOAST'] = df_dodatni['TOAST'].replace('CE?', 1)
df_dodatni['TOAST'] = df_dodatni['TOAST'].replace('SVD', 2)
df_dodatni['TOAST'] = df_dodatni['TOAST'].replace('Drugi', 3)
df_dodatni['TOAST'] = df_dodatni['TOAST'].replace('Neutvrđeno', 4)
df_dodatni['TOAST'] = df_dodatni['TOAST'].replace('Neutrvđeno', 4)
df_dodatni['TOAST'] = df_dodatni['TOAST'].replace('Neutvrđen', 4)
df_dodatni['TOAST'] = df_dodatni['TOAST'].replace('Neutrvđen', 4)
df_dodatni['TOAST'] = df_dodatni['TOAST'].replace('Stroke mimic', 5)

# spajadnje u jedan df
df = pd.concat([df_basic,df_dodatni], axis=1)

df = df.dropna()


# konvertovanje u integer
df['STAROST'] = pd.to_numeric(df['STAROST'])
df['STAROST'] = df['STAROST'].astype(int)
df['Broj dana hospitalizacije'] = df['Broj dana hospitalizacije'].astype(int)


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

# brisu se vrste sa nan values
df = df.dropna()

# integer
df = df.astype({'RANKIN 90 dana':'int'})
df = df.astype({'NIHSS 24h':'int'})
df = df.astype({'TOAST':'int'})
df = df.astype({'NIHSS OT':'int'})
df = df.astype({'RANKIN OT':'int'})
df = df.astype({'TIP CVI':'int'})
df = df.astype({'Leukoarajoza':'int'})


# novi df, y labela
df_rankin = df[['RANKIN 90 dana']].copy()
df_rankin = pd.DataFrame(df_rankin)
df = df.drop(labels='RANKIN 90 dana',axis=1)

print(df.shape)
print(df_rankin.shape)

