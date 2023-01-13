import pandas as pd
from project import dejta_frejm



df_lek.fillna(0)

df_lek['ASA'] = df_lek['ASA'].fillna(0)
df_lek['Clopidogrel'] = df_lek['Clopidogrel'].fillna(0)
df_lek['OAKT'] = df_lek['OAKT'].fillna(0)
df_lek['Statini'] = df_lek['Statini'].fillna(0)
df_lek['AntiHTA'] = df_lek['AntiHTA'].fillna(0)

df_lek['ASA'] = df_lek['ASA'].map({'Da': 1, 'Ne': 0})
df_lek['Clopidogrel'] = df_lek['Clopidogrel'].map({'Da': 2, 'Ne': 0})
df_lek['OAKT'] = df_lek['OAKT'].map({'Da': 3, 'Ne': 0})
df_lek['Statini'] = df_lek['Statini'].map({'Da': 4, 'Ne': 0})
df_lek['AntiHTA'] = df_lek['AntiHTA'].map({'Da': 5, 'Ne': 0})


df_lek['Lekovi'] = df_lek[df_lek.columns[1:]].apply(
    lambda x: ','.join(x.dropna().astype(str)),
    axis=1
)



df_lek['Lekovi']


val = ' '

result = int(val.strip() or 0)

print(df_lek)





