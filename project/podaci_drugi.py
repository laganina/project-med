import tensorflow
import keras
import pandas as pd
import numpy as np

data = pd.read_excel(
    r'C:\Users\Laganina\OneDrive - Univerzitet u Novom Sadu\Desktop\machine_learning\project-med\project\podaci.xlsx')

#(osnovna obelezja: nihss 24h, glikemija, zivotna dob, trajanje hospitalizacije)
# rankin 90 dana - labela (vrednosti se krecu od 0 do 6)

# importovanje osnovnih obelezja
basic = data[['STAROST', 'NIHSS 24h', 'Glikemija', 'Broj dana hospitalizacije', 'RANKIN 90 dana']]
df_basic = pd.DataFrame(basic)
df_basic = df_basic.dropna()



df_basic['STAROST'] = pd.to_numeric(df_basic['STAROST'])
df_basic['STAROST'] = df_basic['STAROST'].astype(int)
df_basic['Broj dana hospitalizacije'] = df_basic['Broj dana hospitalizacije'].astype(int)

df_basic['RANKIN 90 dana'] = df_basic['RANKIN 90 dana'].replace('lost', np.nan)
df_basic['RANKIN 90 dana'] = df_basic['RANKIN 90 dana'].replace('NR', np.nan)
df_basic['RANKIN 90 dana'] = df_basic['RANKIN 90 dana'].replace('ne javlja se', np.nan)
df_basic['RANKIN 90 dana'] = df_basic['RANKIN 90 dana'].replace('nr - lost', np.nan)
df_basic['RANKIN 90 dana'] = df_basic['RANKIN 90 dana'].replace('NR - živi u Nemačkoj', np.nan)
df_basic['RANKIN 90 dana'] = df_basic['RANKIN 90 dana'].replace('lost', np.nan)
df_basic['RANKIN 90 dana'] = df_basic['RANKIN 90 dana'].replace('nedostupan', np.nan)
df_basic['RANKIN 90 dana'] = df_basic['RANKIN 90 dana'].replace('nedostupan ', np.nan)
df_basic['RANKIN 90 dana'] = df_basic['RANKIN 90 dana'].replace('nedostupan, proveri broj', np.nan)
df_basic['RANKIN 90 dana'] = df_basic['RANKIN 90 dana'].replace('nedostupan, proveri broj ', np.nan)

df_basic = df_basic.dropna()
df_basic['RANKIN 90 dana'] = df_basic['RANKIN 90 dana'].astype(int)

df_rankin = df_basic[['RANKIN 90 dana']].copy()
df_rankin = pd.DataFrame(df_rankin)

