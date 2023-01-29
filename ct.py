from podaci_prvi import df_dodatni, joined
import numpy as np
#----------------------------------------------------------------------------------------------------------------------#

# zamena vrednosti brojevima
# CT HIPERDENZNI ZNAK

# da ne bilo koja pretvoreno u 1 0 2
df_dodatni['CT hiperdenzni znak'] = df_dodatni['CT hiperdenzni znak'].replace('Bilo koja', 2)
df_dodatni['CT hiperdenzni znak'] = df_dodatni['CT hiperdenzni znak'].replace('bilo koja', 2)

df_dodatni['CT hiperdenzni znak'] = df_dodatni['CT hiperdenzni znak'].replace('Da', 1)
df_dodatni['CT hiperdenzni znak'] = df_dodatni['CT hiperdenzni znak'].replace('da', 1)

df_dodatni['CT hiperdenzni znak'] = df_dodatni['CT hiperdenzni znak'].replace('Ne', 0)
df_dodatni['CT hiperdenzni znak'] = df_dodatni['CT hiperdenzni znak'].replace('ne', 0)

print('df_dodatni, CT, zamena brojevima:')
print(df_dodatni[['CT hiperdenzni znak']])
print('*****************************************************************************')
print('*****************************************************************************')


