from podaci_prvi import df_dodatni, joined

# zamena vrednosti brojevima
# TOAST

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

joined = joined.astype({'TOAST' :int})

print('df_dodatni, TOAST, zamena brojevima:')
print(df_dodatni[['TOAST']])
print('*****************************************************************************')
print('*****************************************************************************')

