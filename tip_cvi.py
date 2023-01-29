from podaci_prvi import df_dodatni, joined

# zamena vrednosti brojevima
# TIP CVI

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

print('df_dodatni, tip cvi, zamena brojevima:')
print(df_dodatni[['TIP CVI']])
print('*****************************************************************************')
print('*****************************************************************************')

